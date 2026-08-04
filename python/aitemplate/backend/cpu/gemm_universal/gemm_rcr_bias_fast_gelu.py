# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias_fast_gelu codegen backed by XNNPACK."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import (
    _dim_expr,
    _validate,
)


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
  float* output_ptr = static_cast<float*>(output);

  const size_t extra_elements =
      (XNN_EXTRA_BYTES + sizeof(float) - 1) / sizeof(float);

  // XNNPACK fully-connected input may read a few bytes past the
  // logical end, so copy A into a padded scratch buffer.
  const size_t input_elements = m * k;

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

  // The GEMM result becomes the input of approximate GELU.
  // Keep it in a padded scratch buffer because XNNPACK unary
  // since kernels may also read past the logical end of their input.
  const size_t output_elements = m * n;

  thread_local std::vector<float> activation_scratch;
  activation_scratch.resize(output_elements + extra_elements);

  {{func_name}}_xnn_operator_guard fc_guard;

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
          &fc_guard.op),
      "xnn_create_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_reshape_fully_connected_nc_f32(
          fc_guard.op,
          m,
          nullptr),
      "xnn_reshape_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_setup_fully_connected_nc_f32(
          fc_guard.op,
          input_scratch.data(),
          activation_scratch.data()),
      "xnn_setup_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          fc_guard.op,
          nullptr),
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
          0,        // flags
          m,        // batch size
          n,        // channels
          n,        // input stride
          n,        // output stride
          nullptr,  // threadpool
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
    # XNNPACK performs CPU microkernel selection internally.
    return None


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.gen_function")
def gen_function(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    _validate(func_attrs)

    func_name = func_attrs["name"]

    return FUNC_TEMPLATE.render(
        func_name=func_name,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        ),
    )


@registry.reg("cpu.gemm_rcr_bias_fast_gelu.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
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
        indent=indent,
    )
