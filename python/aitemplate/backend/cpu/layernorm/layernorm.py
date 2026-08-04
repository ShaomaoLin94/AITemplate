# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU LayerNorm codegen for BERT-style float32 tensors."""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import normalize_dtype


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <cmath>
#include <cstddef>

#include "device_functions-generated.h"

{{func_signature}}
{
  (void)stream;

  if (m == 0 || n == 0) {
    return;
  }

  const float* input_ptr = static_cast<const float*>(input);
  const float* gamma_ptr = static_cast<const float*>(gamma);
  const float* beta_ptr = static_cast<const float*>(beta);
  float* output_ptr = static_cast<float*>(output);
 

  // naive C++ implementation of LayerNorm
  // -O3 compile optimization should vectorize the loops
  for (size_t row = 0; row < m; ++row) {
    const float* x = input_ptr + row * n;
    float* y = output_ptr + row * n;

    float sum = 0.0f;
    for (size_t col = 0; col < n; ++col) {
      sum += x[col];
    }

    const float mean = sum / static_cast<float>(n);

    float variance_sum = 0.0f;
    for (size_t col = 0; col < n; ++col) {
      const float diff = x[col] - mean;
      variance_sum += diff * diff;
    }

    const float variance =
        variance_sum / static_cast<float>(n);
    const float inv_std =
        1.0f / std::sqrt(variance + eps);

    for (size_t col = 0; col < n; ++col) {
      y[col] =
          (x[col] - mean) *
              inv_std *
              gamma_ptr[col] +
          beta_ptr[col];
    }
  }
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    size_t m,
    size_t n,
    float eps,
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
{{indent}}    {{input}},
{{indent}}    {{gamma}},
{{indent}}    {{beta}},
{{indent}}    {{output}},
{{indent}}    {{m}},
{{indent}}    {{n}},
{{indent}}    {{eps}},
{{indent}}    stream);
"""
)


def _dim_expr(dim) -> str:
    if isinstance(dim, IntImm):
        return str(dim._attrs["values"][0])
    return dim._attrs["name"]


def _validate(func_attrs: Dict[str, Any]) -> None:
    inputs = func_attrs["inputs"]

    # BERT LayerNorm always supplies input, gamma and beta.
    if len(inputs) != 3:
        raise NotImplementedError(
            "CPU LayerNorm currently requires explicit input, gamma and beta"
        )

    x, gamma, beta = inputs

    if len(x._attrs["shape"]) < 2:
        raise NotImplementedError(
            "CPU LayerNorm currently requires input rank >= 2"
        )

    normalized_shape = func_attrs["normalized_shape"]

    # BERT normalizes only the hidden dimension, e.g. [B*S, 768].
    if len(normalized_shape) != 1:
        raise NotImplementedError(
            "CPU LayerNorm currently supports only one normalized dimension"
        )

    if not isinstance(normalized_shape[0], IntImm):
        raise NotImplementedError(
            "CPU LayerNorm currently requires a static normalized dimension"
        )

    if len(gamma._attrs["shape"]) != 1:
        raise NotImplementedError(
            "CPU LayerNorm currently requires 1D gamma"
        )

    if len(beta._attrs["shape"]) != 1:
        raise NotImplementedError(
            "CPU LayerNorm currently requires 1D beta"
        )

    for tensor in (x, gamma, beta):
        dtype = normalize_dtype(tensor._attrs["dtype"])
        if dtype != "float32":
            raise NotImplementedError(
                "CPU LayerNorm currently supports only float32; "
                f"got {tensor._attrs['dtype']}"
            )


@registry.reg("cpu.layernorm.gen_function")
def gen_function(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)

    return FUNC_TEMPLATE.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    )


@registry.reg("cpu.layernorm.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg("cpu.layernorm.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    _validate(func_attrs)

    x, gamma, beta = func_attrs["inputs"]
    output = func_attrs["outputs"][0]

    x_shape = x._attrs["shape"]

    # Everything before the normalized hidden dimension is treated as M.
    # Examples:
    #   [128, 768]    -> M = 128
    #   [2, 128, 768] -> M = 2 * 128
    m = " * ".join(
        _dim_expr(dim)
        for dim in x_shape[:-1]
    )

    n = _dim_expr(func_attrs["normalized_shape"][0])
    eps = f"{float(func_attrs['eps']):.17g}f"

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=x._attrs["name"],
        gamma=gamma._attrs["name"],
        beta=beta._attrs["name"],
        output=output._attrs["name"],
        m=m,
        n=n,
        eps=eps,
        indent=indent,
    )
