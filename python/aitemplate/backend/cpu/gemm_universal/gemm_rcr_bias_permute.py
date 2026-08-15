# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias_permute_m2n3 backed by XNNPACK."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import (
    _dim_expr,
    _validate,
)
from aitemplate.compiler.base import IntImm


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

  {{func_name}}_xnn_operator_guard() = default;

  {{func_name}}_xnn_operator_guard(
      const {{func_name}}_xnn_operator_guard&) = delete;

  {{func_name}}_xnn_operator_guard& operator=(
      const {{func_name}}_xnn_operator_guard&) = delete;
};

}  // namespace


{{func_signature}}
{
  (void)stream;

  if (m == 0 || n == 0 || k == 0) {
    return;
  }

  if (m % t1 != 0 || n % (t2 * t3) != 0) {
    throw std::runtime_error(
        "CPU gemm_rcr_bias_permute_m2n3: invalid dimensions");
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

  // XNNPACK first produces the normal GEMM result:
  //
  // [M, N]
  //
  // We then physically permute it into m2n3 layout.
  thread_local std::vector<float> gemm_scratch;

  gemm_scratch.resize(m * n);

  {{func_name}}_xnn_operator_guard guard;

  {{func_name}}_check_xnn_status(
      xnn_create_fully_connected_nc_f32(
          k,
          n,
          k,
          n,
          b_ptr,
          bias_ptr,
          -std::numeric_limits<float>::infinity(),
          +std::numeric_limits<float>::infinity(),
          0,
          nullptr,
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

  // m2n3:
  //
  // GEMM [M, N]
  //
  // reshape:
  // [M0, M1, N0, N1, N2]
  //
  // where:
  //   M1 = t1
  //   N0 = t2
  //   N1 = t3
  //
  // permute:
  // [N0, M0, N1, M1, N2]
  //
  // BERT:
  // [B*S, 3*hidden]
  //   ->
  // [B, S, 3, heads, head_dim]
  //   ->
  // [3, B, heads, S, head_dim]

  const size_t m0_size = m / t1;
  const size_t n2_size = n / (t2 * t3);

  for (size_t m0 = 0; m0 < m0_size; ++m0) {
    for (size_t m1 = 0; m1 < t1; ++m1) {
      const size_t src_row =
          m0 * t1 + m1;

      for (size_t n0 = 0; n0 < t2; ++n0) {
        for (size_t n1 = 0; n1 < t3; ++n1) {
          const size_t src =
              src_row * n +
              (n0 * t3 + n1) * n2_size;

          const size_t dst =
              ((((n0 * m0_size + m0) * t3 + n1)
                  * t1 + m1)
                  * n2_size);

          std::memcpy(
              output_ptr + dst,
              gemm_scratch.data() + src,
              n2_size * sizeof(float));
        }
      }
    }
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
    void* output,
    size_t m,
    size_t n,
    size_t k,
    size_t t1,
    size_t t2,
    size_t t3,
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
{{indent}}    {{t1}},
{{indent}}    {{t2}},
{{indent}}    {{t3}},
{{indent}}    stream);
"""
)


def _static_int(value) -> int:
    if isinstance(value, IntImm):
        return int(value._attrs["values"][0])
    return int(value)


def _validate_permute(
    func_attrs: Dict[str, Any],
) -> None:
    _validate(func_attrs)

    a = func_attrs["inputs"][0]

    if len(a._attrs["shape"]) < 2:
        raise NotImplementedError(
            "CPU gemm_rcr_bias_permute_m2n3 "
            "requires A rank >= 2"
        )

    if func_attrs.get("layout") != "Permute5D_m2n3":
        raise NotImplementedError(
            "CPU gemm_rcr_bias_permute currently "
            "supports only m2n3"
        )

    shape = func_attrs.get("shape")

    if shape is None or len(shape) != 3:
        raise NotImplementedError(
            "CPU gemm_rcr_bias_permute_m2n3 "
            "requires shape=(t1, t2, t3)"
        )

    for dim in shape:
        if _static_int(dim) <= 0:
            raise ValueError(
                "m2n3 shape dimensions must be positive"
            )


@registry.reg(
    "cpu.gemm_rcr_bias_permute_m2n3.config"
)
def config(
    func_attrs: Dict[str, Any],
    dtype="float32",
) -> None:
    _validate_permute(func_attrs)

    func_attrs["op_instance"] = OrderedDict(
        [
            ("xnnpack", None),
        ]
    )


@registry.reg(
    "cpu.gemm_rcr_bias_permute_m2n3.filter"
)
def function_filter(
    cfg,
    func_attrs,
    ab_alignment,
) -> bool:
    return cfg == "xnnpack"


@registry.reg(
    "cpu.gemm_rcr_bias_permute_m2n3.gen_profiler"
)
def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    return None


@registry.reg(
    "cpu.gemm_rcr_bias_permute_m2n3.gen_function"
)
def gen_function(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    _validate_permute(func_attrs)

    func_name = func_attrs["name"]

    return FUNC_TEMPLATE.render(
        func_name=func_name,
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        ),
    )


@registry.reg(
    "cpu.gemm_rcr_bias_permute_m2n3.func_decl"
)
def gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    _validate_permute(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg(
    "cpu.gemm_rcr_bias_permute_m2n3.func_call"
)
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    _validate_permute(func_attrs)

    a, b, bias = func_attrs["inputs"]
    output = func_attrs["outputs"][0]

    a_shape = a._attrs["shape"]
    b_shape = b._attrs["shape"]

    t1, t2, t3 = [
        _static_int(x)
        for x in func_attrs["shape"]
    ]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a=a._attrs["name"],
        b=b._attrs["name"],
        bias=bias._attrs["name"],
        output=output._attrs["name"],
        m=" * ".join(
            _dim_expr(dim)
            for dim in a_shape[:-1]
        ),
        n=_dim_expr(b_shape[0]),
        k=_dim_expr(a_shape[-1]),
        t1=str(t1),
        t2=str(t2),
        t3=str(t3),
        indent=indent,
    )
