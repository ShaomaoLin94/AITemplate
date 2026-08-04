# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias_add codegen backed by XNNPACK."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import (
    _dim_expr,
    _validate,
)
from aitemplate.compiler.dtype import normalize_dtype


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <xnnpack.h>

#include "device_functions-generated.h"

namespace {

inline void {{func_name}}_check_xnn_status(
    xnn_status status,
    const char* step) {
  if (status != xnn_status_success) {
    throw std::runtime_error(
        std::string(step) +
        " failed with XNNPACK status " +
        std::to_string(static_cast<int>(status)));
  }
}

struct {{func_name}}_xnn_operator_guard {
  xnn_operator_t op = nullptr;

  ~{{func_name}}_xnn_operator_guard() {
    if (op != nullptr) {
      xnn_delete_operator(op);
    }
  }

  {{func_name}}_xnn_operator_guard(
      const {{func_name}}_xnn_operator_guard&) = delete;
  {{func_name}}_xnn_operator_guard& operator=(
      const {{func_name}}_xnn_operator_guard&) = delete;

  {{func_name}}_xnn_operator_guard() = default;
};

}  // namespace

{{func_signature}}
{
  (void)stream;

  if (m == 0 || n == 0 || k == 0) {
    return;
  }

  static const xnn_status init_status = xnn_initialize(nullptr);
  {{func_name}}_check_xnn_status(
      init_status,
      "xnn_initialize");

  const float* a_ptr = static_cast<const float*>(a);
  const float* b_ptr = static_cast<const float*>(b);
  const float* bias_ptr = static_cast<const float*>(bias);
  const float* residual_ptr = static_cast<const float*>(residual);
  float* output_ptr = static_cast<float*>(output);

  // XNNPACK fully-connected kernels may read a few bytes past
  // the logical end of the input.
  const size_t input_elements = m * k;
  const size_t extra_elements =
      (XNN_EXTRA_BYTES + sizeof(float) - 1) / sizeof(float);

  thread_local std::vector<float> input_scratch;
  input_scratch.resize(input_elements + extra_elements);

  std::memcpy(
      input_scratch.data(),
      a_ptr,
      input_elements * sizeof(float));

  std::fill(
      input_scratch.begin() + input_elements,
      input_scratch.end(),
      0.0f);

  // Keep the GEMM result separate from the final output.
  // This also remains correct if AITemplate ever aliases
  // the residual and output buffers.
  const size_t output_elements = m * n;

  thread_local std::vector<float> gemm_scratch;
  gemm_scratch.resize(output_elements);

  {{func_name}}_xnn_operator_guard guard;

  {{func_name}}_check_xnn_status(
      xnn_create_fully_connected_nc_f32(
          k,  // input channels
          n,  // output channels
          k,  // input stride
          n,  // output stride
          b_ptr,
          bias_ptr,
          -std::numeric_limits<float>::infinity(),
          +std::numeric_limits<float>::infinity(),
          0,        // flags
          nullptr,  // weights cache
          &guard.op),
      "xnn_create_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_reshape_fully_connected_nc_f32(
          guard.op,
          m,
          nullptr),
      "xnn_reshape_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_setup_fully_connected_nc_f32(
          guard.op,
          input_scratch.data(),
          gemm_scratch.data()),
      "xnn_setup_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          guard.op,
          nullptr),
      "xnn_run_operator");

  // Residual connection:
  // output = GEMM(A, B) + bias + residual
  for (size_t i = 0; i < output_elements; ++i) {
    output_ptr[i] = gemm_scratch[i] + residual_ptr[i];
  }
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    const void* a,
    const void* b,
    const void* bias,
    const void* residual,
    void* output,
    size_t m,
    size_t n,
    size_t k,
    ait::StreamType stream)
"""
)


FUNC_DECL = jinja2.Template(
    """
{{func_signature}};
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a}},
{{indent}}    {{b}},
{{indent}}    {{bias}},
{{indent}}    {{residual}},
{{indent}}    {{output}},
{{indent}}    {{m}},
{{indent}}    {{n}},
{{indent}}    {{k}},
{{indent}}    stream);
"""
)


def _validate_bias_add(func_attrs: Dict[str, Any]) -> None:
    # Reuse all A/B/bias validation from gemm_rcr_bias.
    _validate(func_attrs)

    if len(func_attrs["inputs"]) != 4:
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add requires A, B, bias and residual"
        )

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    residual = func_attrs["inputs"][3]

    if normalize_dtype(residual._attrs["dtype"]) != "float32":
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add currently supports only float32 residual"
        )

    expected_rank = len(a._attrs["shape"])
    if len(residual._attrs["shape"]) != expected_rank:
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add requires residual rank "
            "to match GEMM output rank"
        )

    # Compiler semantics require:
    # output shape = A[:-1] + [B[0]]
    expected_shape = list(a._attrs["shape"][:-1]) + [
        b._attrs["shape"][0]
    ]

    if residual._attrs["shape"] != expected_shape:
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add requires residual shape "
            "to match GEMM output shape"
        )


@registry.reg("cpu.gemm_rcr_bias_add.config")
def gemm_rcr_bias_add_config(
    func_attrs: Dict[str, Any],
    dtype="float32",
) -> None:
    _validate_bias_add(func_attrs)

    func_attrs["op_instance"] = OrderedDict(
        [
            ("xnnpack", None),
        ]
    )


@registry.reg("cpu.gemm_rcr_bias_add.filter")
def function_filter(
    cfg,
    func_attrs,
    ab_alignment,
) -> bool:
    return cfg == "xnnpack"


@registry.reg("cpu.gemm_rcr_bias_add.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    # XNNPACK performs CPU GEMM microkernel selection internally.
    return None


@registry.reg("cpu.gemm_rcr_bias_add.gen_function")
def gen_function(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    _validate_bias_add(func_attrs)

    func_name = func_attrs["name"]

    return FUNC_TEMPLATE.render(
        func_name=func_name,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        ),
    )


@registry.reg("cpu.gemm_rcr_bias_add.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _validate_bias_add(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg("cpu.gemm_rcr_bias_add.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    _validate_bias_add(func_attrs)

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    bias = func_attrs["inputs"][2]
    residual = func_attrs["inputs"][3]
    output = func_attrs["outputs"][0]

    a_shape = a._attrs["shape"]
    b_shape = b._attrs["shape"]

    m = " * ".join(
        _dim_expr(dim)
        for dim in a_shape[:-1]
    )

    k = _dim_expr(a_shape[-1])
    n = _dim_expr(b_shape[0])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a=a._attrs["name"],
        b=b._attrs["name"],
        bias=bias._attrs["name"],
        residual=residual._attrs["name"],
        output=output._attrs["name"],
        m=m,
        n=n,
        k=k,
        indent=indent,
    )
