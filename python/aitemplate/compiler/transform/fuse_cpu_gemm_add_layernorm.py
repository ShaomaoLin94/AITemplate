# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU fusion for gemm_rcr_bias_add -> LayerNorm."""

import logging
from typing import List

from aitemplate.compiler.base import IntImm, Tensor
from aitemplate.compiler.tensor_accessor import TensorAccessor
from aitemplate.compiler.transform import transform_utils
from aitemplate.compiler.transform.toposort import toposort
from aitemplate.utils import graph_utils


_LOGGER = logging.getLogger(__name__)


_VIEW_OPS = {
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
}


def _only_dst_op(tensor):
    dst_ops = list(tensor._attrs["dst_ops"])

    if len(dst_ops) != 1:
        return None

    return dst_ops[0]


def _find_layernorm_after_views(tensor):
    """
    Follow zero-copy view ops between gemm_rcr_bias_add
    and LayerNorm.
    """
    current = tensor

    while True:
        next_op = _only_dst_op(current)

        if next_op is None:
            return None, None

        if next_op._attrs["op"] == "layernorm":
            return current, next_op

        if next_op._attrs["op"] not in _VIEW_OPS:
            return None, None

        if current._attrs["is_output"]:
            return None, None

        outputs = next_op._attrs["outputs"]

        if len(outputs) != 1:
            return None, None

        current = outputs[0]


def fuse_cpu_gemm_add_layernorm(
    sorted_graph: List[Tensor],
) -> List[Tensor]:

    sorted_ops = list(
        graph_utils.get_sorted_ops(sorted_graph)
    )

    fused = 0

    for gemm_op in sorted_ops:
        if (
            gemm_op._attrs["op"]
            != "gemm_rcr_bias_add"
        ):
            continue

        if gemm_op._attrs.get(
            "fused_layernorm",
            False,
        ):
            continue

        # Original GEMM:
        # A, B, bias, residual
        if len(gemm_op._attrs["inputs"]) != 4:
            continue

        if len(gemm_op._attrs["outputs"]) != 1:
            continue

        gemm_output = (
            gemm_op._attrs["outputs"][0]
        )

        ln_input, layernorm_op = (
            _find_layernorm_after_views(
                gemm_output
            )
        )

        if layernorm_op is None:
            continue

        ln_inputs = layernorm_op._attrs[
            "inputs"
        ]

        # BERT uses:
        # x, gamma, beta
        #
        # Dynamic normalized-shape inputs are deliberately
        # excluded from the first implementation.
        if len(ln_inputs) != 3:
            continue

        if ln_inputs[0] is not ln_input:
            continue

        if (
            layernorm_op._attrs.get(
                "gamma_constant"
            )
            is not None
        ):
            continue

        if (
            layernorm_op._attrs.get(
                "beta_constant"
            )
            is not None
        ):
            continue

        normalized_shape = (
            layernorm_op._attrs.get(
                "normalized_shape"
            )
        )

        if (
            normalized_shape is None
            or len(normalized_shape) != 1
        ):
            continue

        normalized_dim = normalized_shape[0]

        if not isinstance(
            normalized_dim,
            IntImm,
        ):
            continue

        b = gemm_op._attrs["inputs"][1]

        n_dim = b._attrs["shape"][0]

        if not isinstance(n_dim, IntImm):
            continue

        if (
            normalized_dim.value()
            != n_dim.value()
        ):
            continue

        gamma = ln_inputs[1]
        beta = ln_inputs[2]

        if (
            len(gamma._attrs["shape"]) != 1
            or len(beta._attrs["shape"]) != 1
        ):
            continue

        gamma_dim = gamma._attrs["shape"][0]
        beta_dim = beta._attrs["shape"][0]

        if (
            not isinstance(gamma_dim, IntImm)
            or not isinstance(beta_dim, IntImm)
        ):
            continue

        if (
            gamma_dim.value()
            != normalized_dim.value()
            or beta_dim.value()
            != normalized_dim.value()
        ):
            continue

        ln_outputs = (
            layernorm_op._attrs["outputs"]
        )

        if len(ln_outputs) != 1:
            continue

        ln_output = ln_outputs[0]

        #
        # Important:
        #
        # Keep the ORIGINAL gemm_rcr_bias_add class and
        # registry identity.
        #
        # We only add metadata telling the CPU backend
        # to emit the fused implementation.
        #
        gemm_op._attrs[
            "fused_layernorm"
        ] = True

        gemm_op._attrs["eps"] = float(
            layernorm_op._attrs["eps"]
        )

        gemm_op._attrs[
            "normalized_shape"
        ] = list(normalized_shape)

        #
        # Original inputs:
        #
        #   A
        #   B
        #   bias
        #   residual
        #
        # Fused inputs:
        #
        #   A
        #   B
        #   bias
        #   residual
        #   gamma
        #   beta
        #
        gemm_op._attrs["inputs"].extend(
            [
                gamma,
                beta,
            ]
        )

        input_accessors = (
            gemm_op._attrs.get(
                "input_accessors"
            )
        )

        if input_accessors is None:
            gemm_op._attrs[
                "input_accessors"
            ] = [
                TensorAccessor(tensor)
                for tensor
                in gemm_op._attrs[
                    "inputs"
                ]
            ]
        else:
            input_accessors.extend(
                [
                    TensorAccessor(gamma),
                    TensorAccessor(beta),
                ]
            )

        #
        # Gamma/beta are now also consumed by
        # the GEMM fused operator.
        #
        gamma._attrs["dst_ops"].add(
            gemm_op
        )

        beta._attrs["dst_ops"].add(
            gemm_op
        )

        #
        # Disconnect standalone LayerNorm.
        #
        # LayerNorm normally owns:
        #   x -> dst layernorm
        #   gamma -> dst layernorm
        #   beta -> dst layernorm
        #
        for tensor in ln_inputs:
            tensor._attrs[
                "dst_ops"
            ].discard(
                layernorm_op
            )

        #
        # Consumers that previously read the
        # LayerNorm output now read the tensor
        # produced by the fused GEMM chain.
        #
        # replace_tensor also correctly handles
        # the case where LayerNorm was the model
        # output.
        #
        transform_utils.replace_tensor(
            ln_output,
            ln_input,
        )

        fused += 1

    if fused == 0:
        return sorted_graph

    graph_outputs = [
        tensor
        for tensor in sorted_graph
        if tensor._attrs["is_output"]
    ]

    sorted_graph = toposort(
        graph_outputs
    )

    sorted_graph = (
        transform_utils
        .sanitize_sorted_graph(
            sorted_graph
        )
    )

    _LOGGER.info(
        "CPU compiler fusion: fused %d "
        "gemm_rcr_bias_add + layernorm blocks",
        fused,
    )

    return sorted_graph
