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
    render_static_fc_context,
)
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import (
    _dim_expr,
    _validate,
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
        std::to_string(static_cast<int>(status)));
  }
}


{{static_fc_context}}

}  // namespace


extern "C" AIT_EXPORT int
{{func_name}}_prepack(
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

  /*
   * Internal AITemplate input tensors already live inside
   * accessible model-blob storage.
   *
   * External tensors retain the padded-copy fallback.
   */
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

  /*
   * Internal fast path:
   *
   *     FC -> output
   *     GELU(output -> output)
   *
   * For externally-owned output storage we retain the
   * old padded activation buffer as a conservative fallback.
   */
  const size_t output_elements =
      m * n;

  float* gelu_input_ptr =
      output_ptr;

  thread_local std::vector<float>
      activation_scratch;

  if (use_activation_scratch) {
    const size_t extra_elements =
        (XNN_EXTRA_BYTES +
         sizeof(float) - 1) /
        sizeof(float);

    activation_scratch.resize(
        output_elements +
        extra_elements);

    gelu_input_ptr =
        activation_scratch.data();
  }

  /*
   * Static FC cache.
   *
   * cache_id rather than the raw weight pointer is used for
   * identity because the raw constants may be released after
   * XNNPACK has packed them.
   */
  auto& context =
      {{func_name}}_get_cache().get(
          cache_id,
          b_ptr,
          bias_ptr,
          m,
          n,
          k);

  context.run(
      xnn_input_ptr,
      gelu_input_ptr);

  /*
   * Only the fallback scratch needs explicit tail padding.
   * The normal BERT path does not execute this branch.
   */
  if (use_activation_scratch) {
    std::fill(
        activation_scratch.begin() +
            output_elements,
        activation_scratch.end(),
        0.0f);
  }

  union xnn_unary_params
      unary_params = {};

  const struct xnn_quantization_params
      quantization = {
          0,
          1.0f,
      };

  /*
   * Normal BERT path:
   *
   *     gelu_input_ptr == output_ptr
   *
   * Thus ApproxGELU runs in place and the 128x3072
   * activation_scratch is not allocated or touched.
   */
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
          gelu_input_ptr,
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
    bool use_input_scratch,
    bool use_activation_scratch,
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
{{indent}}    {{use_input_scratch}},
{{indent}}    {{use_activation_scratch}},
{{indent}}    stream);
"""
)


def _needs_padded_scratch(tensor) -> bool:
    """Return True if storage may not have safe internal-blob padding."""

    current = tensor
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        attrs = current._attrs

        if (
            attrs.get("is_input", False)
            or attrs.get("is_output", False)
            or attrs.get("is_param", False)
            or attrs.get("has_output_aliases", False)
            or attrs.get("external_tensor") is not None
        ):
            return True

        current = attrs.get("is_view_of")

    return False


@registry.reg(
    "cpu.gemm_rcr_bias_fast_gelu.config"
)
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


@registry.reg(
    "cpu.gemm_rcr_bias_fast_gelu.filter"
)
def function_filter(
    cfg,
    func_attrs,
    ab_alignment,
) -> bool:
    return cfg == "xnnpack"


@registry.reg(
    "cpu.gemm_rcr_bias_fast_gelu.gen_profiler"
)
def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    return None


@registry.reg(
    "cpu.gemm_rcr_bias_fast_gelu.gen_function"
)
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
        static_fc_context=render_static_fc_context(func_name),
        prepack_n=_dim_expr(
            b._attrs["shape"][0]
        ),
        prepack_k=_dim_expr(
            a._attrs["shape"][-1]
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_name,
        ),
    )


@registry.reg(
    "cpu.gemm_rcr_bias_fast_gelu.func_decl"
)
def gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
        )
    ).strip()


@registry.reg(
    "cpu.gemm_rcr_bias_fast_gelu.func_call"
)
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

    k = _dim_expr(
        a_shape[-1]
    )

    n = _dim_expr(
        b_shape[0]
    )

    cache_id = (
        str(
            cache_id_from_tensor_name(
                b._attrs["name"]
            )
        )
        + "ULL"
    )

    use_input_scratch = (
        "true"
        if _needs_padded_scratch(a)
        else "false"
    )

    use_activation_scratch = (
        "true"
        if _needs_padded_scratch(output)
        else "false"
    )

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        a=a._attrs["name"],
        b=b._attrs["name"],
        bias=bias._attrs["name"],
        output=output._attrs["name"],
        m=m,
        n=n,
        k=k,
        cache_id=cache_id,
        use_input_scratch=use_input_scratch,
        use_activation_scratch=use_activation_scratch,
        indent=indent,
    )
