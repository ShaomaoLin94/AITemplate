# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias codegen backed by XNNPACK."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.compiler.base import IntImm
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
  float* output_ptr = static_cast<float*>(output);

  // XNNPACK kernels may read a few bytes beyond the logical end of input.
  // Copy into a padded buffer so those reads are always safe.
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
          output_ptr),
      "xnn_setup_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          guard.op,
          nullptr),
      "xnn_run_operator");
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


def _dim_expr(dim) -> str:
    if isinstance(dim, IntImm):
        return str(dim._attrs["values"][0])
    return dim._attrs["name"]


def _validate(func_attrs: Dict[str, Any]) -> None:
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    bias = func_attrs["inputs"][2]

    a_shape = a._attrs["shape"]
    b_shape = b._attrs["shape"]
    bias_shape = bias._attrs["shape"]

    if len(a_shape) < 2:
        raise NotImplementedError(
            "CPU XNNPACK gemm_rcr_bias requires input A rank >= 2"
        )

    if len(b_shape) != 2:
        raise NotImplementedError(
            "CPU XNNPACK gemm_rcr_bias currently requires weight B to be rank 2"
        )

    if len(bias_shape) != 1:
        raise NotImplementedError(
            "CPU XNNPACK gemm_rcr_bias requires a 1D bias"
        )

    tensors = [a, b, bias]
    for tensor in tensors:
        dtype = normalize_dtype(tensor._attrs["dtype"])
        if dtype != "float32":
            raise NotImplementedError(
                "CPU XNNPACK gemm_rcr_bias currently supports only float32; "
                f"got {tensor._attrs['dtype']}"
            )


@registry.reg("cpu.gemm_rcr_bias.config")
def gemm_rcr_bias_config(
    func_attrs: Dict[str, Any],
    dtype="float32",
) -> None:
    _validate(func_attrs)

    # AITemplate normally stores many candidate GPU kernels here.
    # CPU currently delegates the implementation to XNNPACK.
    func_attrs["op_instance"] = OrderedDict(
        [
            ("xnnpack", None),
        ]
    )


@registry.reg("cpu.gemm_rcr_bias.filter")
def function_filter(
    cfg,
    func_attrs,
    ab_alignment,
) -> bool:
    return cfg == "xnnpack"


@registry.reg("cpu.gemm_rcr_bias.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    # XNNPACK performs its own CPU microkernel selection.
    return None


@registry.reg("cpu.gemm_rcr_bias.gen_function")
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


@registry.reg("cpu.gemm_rcr_bias.func_decl")
def gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg("cpu.gemm_rcr_bias.func_call")
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

    # gemm_rcr treats every A dimension except the last one as M.
    # Example:
    #   A [batch, seq, hidden]
    # becomes
    #   M = batch * seq
    #   K = hidden
    m_dims = [_dim_expr(dim) for dim in a_shape[:-1]]
    m = " * ".join(m_dims)

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
