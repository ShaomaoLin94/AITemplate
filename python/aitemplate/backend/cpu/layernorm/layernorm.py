# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU float32 LayerNorm codegen."""

from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import normalize_dtype


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "device_functions-generated.h"
#include "cpu_threadpool.h"

namespace {

struct {{func_name}}_layernorm_context {
  const float* input;
  const float* gamma;
  const float* beta;
  float* output;
  size_t n;
  float eps;
};

void {{func_name}}_layernorm_row(
    void* raw_context,
    size_t row) {
  auto* context =
      static_cast<{{func_name}}_layernorm_context*>(raw_context);

  const float* input_row =
      context->input + row * context->n;
  float* output_row =
      context->output + row * context->n;

  double sum = 0.0;
  double square_sum = 0.0;

  for (size_t col = 0; col < context->n; ++col) {
    const double value = static_cast<double>(input_row[col]);
    sum += value;
    square_sum += value * value;
  }

  const double inv_n = 1.0 / static_cast<double>(context->n);
  const double mean = sum * inv_n;
  double variance = square_sum * inv_n - mean * mean;
  variance = std::max(variance, 0.0);

  const float inv_std = static_cast<float>(
      1.0 / std::sqrt(variance + static_cast<double>(context->eps)));
  const float mean_f = static_cast<float>(mean);

  for (size_t col = 0; col < context->n; ++col) {
    const float gamma_value =
        context->gamma != nullptr ? context->gamma[col] : 1.0f;
    const float beta_value =
        context->beta != nullptr ? context->beta[col] : 0.0f;

    output_row[col] =
        (input_row[col] - mean_f) * inv_std * gamma_value + beta_value;
  }
}

}  // namespace

{{func_signature}}
{
  (void)stream;

  if (m == 0 || n == 0) {
    return;
  }

  {{func_name}}_layernorm_context context{
      static_cast<const float*>(input),
      static_cast<const float*>(gamma),
      static_cast<const float*>(beta),
      static_cast<float*>(output),
      n,
      eps};

  ait::parallelize_1d(
      {{func_name}}_layernorm_row,
      &context,
      m);
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
    x = func_attrs["inputs"][0]
    if normalize_dtype(x._attrs["dtype"]) != "float32":
        raise NotImplementedError("CPU LayerNorm currently supports only float32")

    normalized_shape = func_attrs["normalized_shape"]
    if len(normalized_shape) != 1:
        raise NotImplementedError(
            "CPU LayerNorm currently supports one normalized dimension"
        )
    if not isinstance(normalized_shape[0], IntImm):
        raise NotImplementedError(
            "CPU LayerNorm requires a static normalized dimension"
        )


def _get_gamma_beta(func_attrs: Dict[str, Any]):
    inputs = func_attrs["inputs"]
    index = 1

    if func_attrs.get("gamma_constant") is None:
        gamma = inputs[index]._attrs["name"]
        index += 1
    else:
        gamma = "nullptr"

    if func_attrs.get("beta_constant") is None:
        beta = inputs[index]._attrs["name"]
    else:
        beta = "nullptr"

    return gamma, beta


@registry.reg("cpu.layernorm.gen_function")
def gen_function(func_attrs: Dict[str, Any], *args, **kwargs) -> str:
    _validate(func_attrs)
    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"]),
    )


@registry.reg("cpu.layernorm.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(func_name=func_attrs["name"])
    ).strip()


@registry.reg("cpu.layernorm.func_call")
def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    _validate(func_attrs)

    x = func_attrs["inputs"][0]
    output = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]

    m = " * ".join(_dim_expr(dim) for dim in x_shape[:-1])
    n = _dim_expr(x_shape[-1])
    gamma, beta = _get_gamma_beta(func_attrs)

    eps = repr(float(func_attrs["eps"]))
    if "." not in eps and "e" not in eps.lower():
        eps += ".0"
    eps += "f"

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=x._attrs["name"],
        gamma=gamma,
        beta=beta,
        output=output._attrs["name"],
        m=m,
        n=n,
        eps=eps,
        indent=indent,
    )
