# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Backend overlay for fused gemm_rcr_bias_add + LayerNorm."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import (
    _dim_expr,
)
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias_add import (
    _needs_input_scratch,
)
from aitemplate.backend.cpu.gemm_universal.static_fc import (
    cache_id_from_tensor_name,
    render_static_fc_context,
)
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import normalize_dtype


_BASE_CONFIG = registry.get(
    "cpu.gemm_rcr_bias_add.config"
)
_BASE_GEN_PROFILER = registry.get(
    "cpu.gemm_rcr_bias_add.gen_profiler"
)
_BASE_GEN_FUNCTION = registry.get(
    "cpu.gemm_rcr_bias_add.gen_function"
)
_BASE_FUNC_DECL = registry.get(
    "cpu.gemm_rcr_bias_add.func_decl"
)
_BASE_FUNC_CALL = registry.get(
    "cpu.gemm_rcr_bias_add.func_call"
)


def _is_fused(func_attrs):
    return bool(
        func_attrs.get(
            "fused_layernorm",
            False,
        )
    )


def _static_int(dim):
    if not isinstance(dim, IntImm):
        raise NotImplementedError(
            "CPU fused residual LayerNorm "
            "currently requires static N"
        )

    return dim.value()


def _validate_fused(func_attrs):
    if len(func_attrs["inputs"]) != 6:
        raise NotImplementedError(
            "Fused gemm_rcr_bias_add+layernorm "
            "requires A, B, bias, residual, gamma, beta"
        )

    a, b, bias, residual, gamma, beta = (
        func_attrs["inputs"]
    )

    for tensor in (
        a,
        b,
        bias,
        residual,
        gamma,
        beta,
        func_attrs["outputs"][0],
    ):
        if normalize_dtype(
            tensor._attrs["dtype"]
        ) != "float32":
            raise NotImplementedError(
                "CPU fused residual LayerNorm "
                "currently supports float32 only"
            )

    n = _static_int(
        b._attrs["shape"][0]
    )

    if (
        len(gamma._attrs["shape"]) != 1
        or _static_int(
            gamma._attrs["shape"][0]
        )
        != n
    ):
        raise NotImplementedError(
            "gamma must have shape [N]"
        )

    if (
        len(beta._attrs["shape"]) != 1
        or _static_int(
            beta._attrs["shape"][0]
        )
        != n
    ):
        raise NotImplementedError(
            "beta must have shape [N]"
        )

    normalized_shape = func_attrs.get(
        "normalized_shape"
    )

    if (
        normalized_shape is None
        or len(normalized_shape) != 1
        or _static_int(normalized_shape[0])
        != n
    ):
        raise NotImplementedError(
            "LayerNorm normalized_shape must be [N]"
        )


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    const void* a,
    const void* b,
    const void* bias,
    const void* residual,
    const void* gamma,
    const void* beta,
    void* output,
    size_t m,
    size_t n,
    size_t k,
    uint64_t cache_id,
    bool use_input_scratch,
    float eps,
    ait::StreamType stream)
"""
)


FUNC_DECL = jinja2.Template(
    """
{{func_signature}};
"""
)


FUNC_CALL = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{a}},
{{indent}}    {{b}},
{{indent}}    {{bias}},
{{indent}}    {{residual}},
{{indent}}    {{gamma}},
{{indent}}    {{beta}},
{{indent}}    {{output}},
{{indent}}    {{m}},
{{indent}}    {{n}},
{{indent}}    {{k}},
{{indent}}    {{cache_id}},
{{indent}}    {{use_input_scratch}},
{{indent}}    {{eps}},
{{indent}}    stream);
"""
)


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <xnnpack.h>

#include "device_functions-generated.h"
#include "cpu_threadpool.h"
#include "model_interface.h"

namespace {

inline void {{func_name}}_check_xnn_status(
    xnn_status status,
    const char* step) {

  if (status != xnn_status_success) {
    throw std::runtime_error(
        std::string(step) +
        " failed with XNNPACK status " +
        std::to_string(
            static_cast<int>(status)));
  }
}


{{static_fc_context}}


struct {{func_name}}_layernorm_context {
  const float* residual;
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
      static_cast<
          {{func_name}}_layernorm_context*>(
              raw_context);

  const size_t offset =
      row * context->n;

  float* output_row =
      context->output + offset;

  const float* residual_row =
      context->residual + offset;

  double sum = 0.0;
  double square_sum = 0.0;

  // First pass:
  //
  // output currently contains GEMM+bias.
  // Add residual in-place and collect LayerNorm statistics.
  for (
      size_t col = 0;
      col < context->n;
      ++col) {

    const float value =
        output_row[col] +
        residual_row[col];

    output_row[col] = value;

    const double value_d =
        static_cast<double>(value);

    sum += value_d;
    square_sum += value_d * value_d;
  }

  const double inv_n =
      1.0 /
      static_cast<double>(context->n);

  const double mean =
      sum * inv_n;

  double variance =
      square_sum * inv_n -
      mean * mean;

  variance =
      std::max(
          variance,
          0.0);

  const float inv_std =
      static_cast<float>(
          1.0 /
          std::sqrt(
              variance +
              static_cast<double>(
                  context->eps)));

  const float mean_f =
      static_cast<float>(mean);

  // Second pass:
  // normalize the same buffer in-place.
  for (
      size_t col = 0;
      col < context->n;
      ++col) {

    output_row[col] =
        (output_row[col] - mean_f) *
            inv_std *
            context->gamma[col] +
        context->beta[col];
  }
}

}  // namespace


extern "C" AIT_EXPORT int
{{func_name}}_prepack(
    uint64_t cache_id,
    const void* b,
    const void* bias,
    size_t n,
    size_t k) {

  if (
      n != static_cast<size_t>(
          {{prepack_n}})
      ||
      k != static_cast<size_t>(
          {{prepack_k}})) {
    return 0;
  }

  static const xnn_status init_status =
      xnn_initialize(nullptr);

  {{func_name}}_check_xnn_status(
      init_status,
      "xnn_initialize(prepack)");

  {{func_name}}_get_cache().prepack(
      cache_id,
      static_cast<const float*>(b),
      static_cast<const float*>(bias),
      n,
      k);

  return 1;
}


{{func_signature}}
{
  (void)stream;

  if (
      m == 0 ||
      n == 0 ||
      k == 0) {
    return;
  }

  static const xnn_status init_status =
      xnn_initialize(nullptr);

  {{func_name}}_check_xnn_status(
      init_status,
      "xnn_initialize");

  const float* a_ptr =
      static_cast<const float*>(a);

  const float* b_ptr =
      static_cast<const float*>(b);

  const float* bias_ptr =
      static_cast<const float*>(bias);

  const float* residual_ptr =
      static_cast<const float*>(
          residual);

  const float* gamma_ptr =
      static_cast<const float*>(gamma);

  const float* beta_ptr =
      static_cast<const float*>(beta);

  float* output_ptr =
      static_cast<float*>(output);

  const float* xnn_input_ptr =
      a_ptr;

  thread_local std::vector<float>
      input_scratch;

  if (use_input_scratch) {
    const size_t input_elements =
        m * k;

    const size_t extra_elements =
        (XNN_EXTRA_BYTES +
         sizeof(float) - 1) /
        sizeof(float);

    input_scratch.resize(
        input_elements +
        extra_elements);

    std::memcpy(
        input_scratch.data(),
        a_ptr,
        input_elements *
            sizeof(float));

    std::fill(
        input_scratch.begin() +
            input_elements,
        input_scratch.end(),
        0.0f);

    xnn_input_ptr =
        input_scratch.data();
  }

  auto& context =
      {{func_name}}_get_cache().get(
          cache_id,
          b_ptr,
          bias_ptr,
          m,
          n,
          k);

  // XNNPACK writes GEMM+bias directly into the final tensor.
  // This remains true for both FP32 and dynamic-QD8 INT8 FC.
  context.run(
      xnn_input_ptr,
      output_ptr);

  {{func_name}}_layernorm_context
      layernorm_context{
          residual_ptr,
          gamma_ptr,
          beta_ptr,
          output_ptr,
          n,
          eps};

  ait::parallelize_1d(
      {{func_name}}_layernorm_row,
      &layernorm_context,
      m);
}
"""
)


def config(
    func_attrs: Dict[str, Any],
    dtype="float32",
):
    if not _is_fused(func_attrs):
        return _BASE_CONFIG(
            func_attrs,
            dtype,
        )

    _validate_fused(func_attrs)

    func_attrs["op_instance"] = OrderedDict(
        [
            ("xnnpack", None),
        ]
    )


def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    if not _is_fused(func_attrs):
        return _BASE_GEN_PROFILER(
            func_attrs,
            workdir,
            *args,
            **kwargs,
        )

    return None


def gen_function(
    func_attrs,
    exec_cond_template=None,
    dim_info_dict=None,
):
    if not _is_fused(func_attrs):
        return _BASE_GEN_FUNCTION(
            func_attrs,
            exec_cond_template,
            dim_info_dict,
        )

    _validate_fused(func_attrs)

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]

    func_name = func_attrs["name"]

    return FUNC_TEMPLATE.render(
        func_name=func_name,
        static_fc_context=render_static_fc_context(func_name),
        prepack_n=_dim_expr(
            b._attrs["shape"][0]
        ),
        prepack_k=_dim_expr(
            a._attrs["shape"][-1]
        ),
        func_signature=(
            FUNC_SIGNATURE.render(
                func_name=func_name
            )
        ),
    )


def func_decl(func_attrs):
    if not _is_fused(func_attrs):
        return _BASE_FUNC_DECL(
            func_attrs
        )

    _validate_fused(func_attrs)

    return FUNC_DECL.render(
        func_signature=(
            FUNC_SIGNATURE.render(
                func_name=func_attrs[
                    "name"
                ]
            )
        )
    ).strip()


def func_call(
    func_attrs,
    indent="  ",
):
    if not _is_fused(func_attrs):
        return _BASE_FUNC_CALL(
            func_attrs,
            indent,
        )

    _validate_fused(func_attrs)

    (
        a,
        b,
        bias,
        residual,
        gamma,
        beta,
    ) = func_attrs["inputs"]

    output = func_attrs["outputs"][0]

    a_shape = a._attrs["shape"]
    b_shape = b._attrs["shape"]

    m = " * ".join(
        _dim_expr(dim)
        for dim in a_shape[:-1]
    )

    return FUNC_CALL.render(
        func_name=func_attrs["name"],
        a=a._attrs["name"],
        b=b._attrs["name"],
        bias=bias._attrs["name"],
        residual=residual._attrs["name"],
        gamma=gamma._attrs["name"],
        beta=beta._attrs["name"],
        output=output._attrs["name"],
        m=m,
        n=_dim_expr(
            b_shape[0]
        ),
        k=_dim_expr(
            a_shape[-1]
        ),
        cache_id=(
            str(
                cache_id_from_tensor_name(
                    b._attrs["name"]
                )
            )
            + "ULL"
        ),
        use_input_scratch=(
            "true"
            if _needs_input_scratch(a)
            else "false"
        ),
        eps=repr(
            float(
                func_attrs["eps"]
            )
        ) + "f",
        indent=indent,
    )


# Replace only the public registry hooks.
#
# The original functions were captured above and remain the fallback for
# ordinary, non-fused gemm_rcr_bias_add operators.
registry.BACKEND_FUNCTIONS[
    "cpu.gemm_rcr_bias_add.config"
] = config

registry.BACKEND_FUNCTIONS[
    "cpu.gemm_rcr_bias_add.gen_profiler"
] = gen_profiler

registry.BACKEND_FUNCTIONS[
    "cpu.gemm_rcr_bias_add.gen_function"
] = gen_function

registry.BACKEND_FUNCTIONS[
    "cpu.gemm_rcr_bias_add.func_decl"
] = func_decl

registry.BACKEND_FUNCTIONS[
    "cpu.gemm_rcr_bias_add.func_call"
] = func_call
