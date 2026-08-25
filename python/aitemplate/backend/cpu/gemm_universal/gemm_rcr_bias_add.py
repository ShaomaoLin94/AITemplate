# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias_add codegen backed by XNNPACK."""

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
from aitemplate.compiler.dtype import normalize_dtype


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

  {{func_name}}_xnn_context(
      const {{func_name}}_xnn_context&) = delete;

  {{func_name}}_xnn_context& operator=(
      const {{func_name}}_xnn_context&) = delete;
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

  const float* residual_ptr =
      static_cast<const float*>(residual);

  float* output_ptr =
      static_cast<float*>(output);

  /*
   * Internal AITemplate tensors are stored inside the model blob.
   * The CPU model blob has extra tail padding, so XNNPACK's
   * bounded input overread is safe without making another copy.
   *
   * External tensors do not have this guarantee, so keep the
   * original padded scratch path for them.
   */
  const float* xnn_input_ptr = a_ptr;

  thread_local std::vector<float> input_scratch;

  if (use_input_scratch) {
    const size_t input_elements = m * k;

    const size_t extra_elements =
        (XNN_EXTRA_BYTES + sizeof(float) - 1) /
        sizeof(float);

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

    xnn_input_ptr = input_scratch.data();
  }

  const size_t output_elements = m * n;

  thread_local std::vector<float> gemm_scratch;

  gemm_scratch.resize(output_elements);

  // Different BERT layers share generated functions.
  // Keep one XNNPACK operator for each weight/bias pair.
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
          xnn_input_ptr,
          gemm_scratch.data()),
      "xnn_setup_fully_connected_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          context.op,
          ait::cpu_threadpool()),
      "xnn_run_operator");

  // output = GEMM(A, B) + bias + residual.
  //
  // Parallelize by GEMM row rather than by scalar element. BERT has
  // M=128 and N=768 here, which gives enough work per task while
  // avoiding thousands of tiny pthreadpool dispatches.
  struct residual_add_context {
    const float* gemm;
    const float* residual;
    float* output;
    size_t n;
  };

  residual_add_context residual_context{
      gemm_scratch.data(),
      residual_ptr,
      output_ptr,
      n};

  auto residual_add_task =
      [](void* raw_context, size_t row) {
        auto* task_context =
            static_cast<residual_add_context*>(
                raw_context);

        const size_t offset =
            row * task_context->n;

        const float* gemm_row =
            task_context->gemm + offset;

        const float* residual_row =
            task_context->residual + offset;

        float* output_row =
            task_context->output + offset;

        for (size_t col = 0;
             col < task_context->n;
             ++col) {
          output_row[col] =
              gemm_row[col] +
              residual_row[col];
        }
      };

  ait::parallelize_1d(
      residual_add_task,
      &residual_context,
      m);
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
    uint64_t cache_id,
    bool use_input_scratch,
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
{{indent}}    {{cache_id}},
{{indent}}    {{use_input_scratch}},
{{indent}}    stream);
"""
)


def _needs_input_scratch(tensor) -> bool:
    """Return True if the tensor may use storage outside the internal blob."""

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


def _validate_bias_add(
    func_attrs: Dict[str, Any],
) -> None:
    _validate(func_attrs)

    if len(func_attrs["inputs"]) != 4:
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add requires "
            "A, B, bias and residual"
        )

    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    residual = func_attrs["inputs"][3]

    if normalize_dtype(
        residual._attrs["dtype"]
    ) != "float32":
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add currently "
            "supports only float32 residual"
        )

    def static_numel(shape):
        total = 1

        for dim in shape:
            values = dim._attrs.get("values")

            if values is None or len(values) != 1:
                return None

            total *= int(values[0])

        return total

    expected_shape = list(
        a._attrs["shape"][:-1]
    ) + [
        b._attrs["shape"][0]
    ]

    expected_numel = static_numel(
        expected_shape
    )

    residual_numel = static_numel(
        residual._attrs["shape"]
    )

    if (
        expected_numel is not None
        and residual_numel is not None
        and expected_numel != residual_numel
    ):
        raise NotImplementedError(
            "CPU gemm_rcr_bias_add requires residual "
            "to have the same number of elements as "
            "the GEMM output; "
            f"got output numel={expected_numel}, "
            f"residual numel={residual_numel}"
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
    return None


@registry.reg("cpu.gemm_rcr_bias_add.gen_function")
def gen_function(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    _validate_bias_add(func_attrs)

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


@registry.reg("cpu.gemm_rcr_bias_add.func_decl")
def gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
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
        cache_id=str(cache_id_from_tensor_name(b._attrs["name"])) + "ULL",
        use_input_scratch=(
            "true"
            if _needs_input_scratch(a)
            else "false"
        ),
        indent=indent,
    )
