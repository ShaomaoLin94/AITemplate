# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias_fast_gelu codegen backed by XNNPACK."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cpu.gemm_universal.static_fc import (
    cache_id_from_tensor_name,
)
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import (
    _dim_expr,
    _validate,
)


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
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
        std::to_string(static_cast<int>(status)));
  }
}


struct {{func_name}}_xnn_context {
  xnn_operator_t op = nullptr;

  uint64_t cache_id = 0;
  const float* weight_ptr = nullptr;
  const float* bias_ptr = nullptr;

  size_t cached_m = 0;
  size_t n = 0;
  size_t k = 0;

  {{func_name}}_xnn_context(
      uint64_t id,
      const float* weight,
      const float* bias,
      size_t output_channels,
      size_t input_channels)
      : cache_id(id),
        weight_ptr(weight),
        bias_ptr(bias),
        n(output_channels),
        k(input_channels) {

    {{func_name}}_check_xnn_status(
        xnn_create_fully_connected_nc_f32(
            k,
            n,
            k,
            n,
            weight_ptr,
            bias_ptr,
            -std::numeric_limits<float>::infinity(),
            +std::numeric_limits<float>::infinity(),
            0,
            nullptr,
            &op),
        "xnn_create_fully_connected_nc_f32");
  }

  ~{{func_name}}_xnn_context() {
    if (op != nullptr) {
      xnn_delete_operator(op);
    }
  }

  {{func_name}}_xnn_context(
      const {{func_name}}_xnn_context&) = delete;

  {{func_name}}_xnn_context& operator=(
      const {{func_name}}_xnn_context&) = delete;

  bool matches(
      uint64_t id,
      size_t output_channels,
      size_t input_channels) const {
    return cache_id == id &&
           n == output_channels &&
           k == input_channels;
  }

  void reshape(size_t m) {
    if (cached_m == m) {
      return;
    }

    {{func_name}}_check_xnn_status(
        xnn_reshape_fully_connected_nc_f32(
            op,
            m,
            ait::cpu_threadpool()),
        "xnn_reshape_fully_connected_nc_f32");

    cached_m = m;
  }
};


struct {{func_name}}_xnn_cache {
  std::list<{{func_name}}_xnn_context> contexts;

  {{func_name}}_xnn_context& get(
      uint64_t cache_id,
      const float* weight,
      const float* bias,
      size_t m,
      size_t n,
      size_t k) {

    for (auto& context : contexts) {
      if (context.matches(
              cache_id,
              n,
              k)) {
        context.reshape(m);
        return context;
      }
    }

    if (weight == nullptr || bias == nullptr) {
      throw std::runtime_error(
          "CPU static FC cache miss after raw weight release");
    }

    contexts.emplace_back(
        cache_id,
        weight,
        bias,
        n,
        k);

    auto& context = contexts.back();
    context.reshape(m);

    return context;
  }
};

thread_local {{func_name}}_xnn_cache {{func_name}}_fc_cache;

}  // namespace


extern "C" AIT_EXPORT int {{func_name}}_prepack(
    uint64_t cache_id,
    const void* b,
    const void* bias,
    size_t n,
    size_t k) {
  if (n != static_cast<size_t>({{prepack_n}}) ||
      k != static_cast<size_t>({{prepack_k}})) {
    return 0;
  }

  static const xnn_status init_status =
      xnn_initialize(nullptr);

  {{func_name}}_check_xnn_status(
      init_status,
      "xnn_initialize(prepack)");

  const float* b_ptr =
      static_cast<const float*>(b);

  const float* bias_ptr =
      static_cast<const float*>(bias);

  // m=1 is enough to force XNNPACK operator creation and weight packing.
  // The real inference path will reshape the cached operator to its actual m.
  {{func_name}}_fc_cache.get(
      cache_id,
      b_ptr,
      bias_ptr,
      1,
      n,
      k);

  return 1;
}


{{func_signature}}
{
  (void)stream;

  if (m == 0 || n == 0 || k == 0) {
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

  float* output_ptr =
      static_cast<float*>(output);

  const size_t input_elements = m * k;

  const size_t extra_elements =
      (XNN_EXTRA_BYTES + sizeof(float) - 1) /
      sizeof(float);

  // XNNPACK may read a few bytes past the logical input.
  thread_local std::vector<float> input_scratch;

  input_scratch.resize(
      input_elements + extra_elements);

  std::memcpy(
      input_scratch.data(),
      a_ptr,
      input_elements * sizeof(float));

  std::fill(
      input_scratch.begin() + input_elements,
      input_scratch.end(),
      0.0f);

  const size_t output_elements = m * n;

  // Keep the GEMM result in padded storage until we have
  // separately verified that ApproxGELU is safe in-place.
  thread_local std::vector<float> activation_scratch;

  activation_scratch.resize(
      output_elements + extra_elements);

  // Weights and bias are constant during inference.
  // Do not recreate and repack the XNNPACK FC operator
  // every time this function is called.
  auto& context = {{func_name}}_fc_cache.get(
      cache_id,
      b_ptr,
      bias_ptr,
      m,
      n,
      k);

  {{func_name}}_check_xnn_status(
      xnn_setup_fully_connected_nc_f32(
          context.op,
          input_scratch.data(),
          activation_scratch.data()),
      "xnn_setup_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          context.op,
          ait::cpu_threadpool()),
      "xnn_run_operator(fully_connected)");

  std::fill(
      activation_scratch.begin() + output_elements,
      activation_scratch.end(),
      0.0f);

  union xnn_unary_params unary_params = {};

  const struct xnn_quantization_params quantization = {
      0,
      1.0f,
  };

  {{func_name}}_check_xnn_status(
      xnn_run_unary_elementwise_nc(
          xnn_unary_approxgelu,
          xnn_datatype_fp32,
          xnn_datatype_fp32,
          &unary_params,
          &quantization,
          &quantization,
          0,
          m,
          n,
          n,
          n,
          ait::cpu_threadpool(),
          activation_scratch.data(),
          output_ptr),
      "xnn_run_unary_elementwise_nc(approxgelu)");
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    const void* a,
    const void* b,
    const void* bias,
    void* output,
    size_t m,
    size_t n,
    size_t k,
    uint64_t cache_id,
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
{{indent}}    {{output}},
{{indent}}    {{m}},
{{indent}}    {{n}},
{{indent}}    {{k}},
{{indent}}    {{cache_id}},
{{indent}}    stream);
"""
)



@registry.reg("cpu.gemm_rcr_bias_fast_gelu.config")
def gemm_rcr_bias_fast_gelu_config(
    func_attrs: Dict[str, Any],
    dtype="float32",
) -> None:
    _validate(func_attrs)

    func_attrs["op_instance"] = OrderedDict(
        [
            ("xnnpack", None),
        ]
    )


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.filter")
def function_filter(
    cfg,
    func_attrs,
    ab_alignment,
) -> bool:
    return cfg == "xnnpack"


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    return None


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.gen_function")
def gen_function(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    _validate(func_attrs)

    func_name = func_attrs["name"]

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]

    return FUNC_TEMPLATE.render(
        func_name=func_name,
        prepack_n=_dim_expr(b._attrs["shape"][0]),
        prepack_k=_dim_expr(a._attrs["shape"][-1]),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        ),
    )


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.func_decl")
def gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    _validate(func_attrs)

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    bias = func_attrs["inputs"][2]
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
        output=output._attrs["name"],
        m=m,
        n=n,
        k=k,
        cache_id=str(cache_id_from_tensor_name(b._attrs["name"])) + "ULL",
        indent=indent,
    )
