# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU gemm_rcr_bias_permute_m2n3 backed by XNNPACK."""

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
from aitemplate.compiler.base import IntImm


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
        xnn_initialize(nullptr),
        "xnn_initialize");

    /*
     * Static fully-connected binds and packs weight/bias here.
     * After this constructor finishes, later setup/run calls do
     * not need to read the original raw weight storage.
     */
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

  {{func_name}}_xnn_context& prepack(
      uint64_t cache_id,
      const float* weight,
      const float* bias,
      size_t n,
      size_t k) {

    for (auto& context : contexts) {
      if (context.matches(
              cache_id,
              n,
              k)) {
        return context;
      }
    }

    if (weight == nullptr || bias == nullptr) {
      throw std::runtime_error(
          "CPU QKV static FC cache miss after raw weight release");
    }

    contexts.emplace_back(
        cache_id,
        weight,
        bias,
        n,
        k);

    return contexts.back();
  }

  {{func_name}}_xnn_context& get(
      uint64_t cache_id,
      const float* weight,
      const float* bias,
      size_t m,
      size_t n,
      size_t k) {

    auto& context = prepack(
        cache_id,
        weight,
        bias,
        n,
        k);

    context.reshape(m);
    return context;
  }
};


inline {{func_name}}_xnn_cache&
{{func_name}}_get_cache() {
  thread_local {{func_name}}_xnn_cache cache;
  return cache;
}

}  // namespace


extern "C" AIT_EXPORT int {{func_name}}_prepack(
    uint64_t cache_id,
    const void* b,
    const void* bias,
    size_t n,
    size_t k) {
  if (n != static_cast<size_t>({{prepack_n}}) ||
      k != static_cast<size_t>({{prepack_k}}) ||
      b == nullptr ||
      bias == nullptr) {
    return 0;
  }

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

  if (m % t1 != 0 || n % (t2 * t3) != 0) {
    throw std::runtime_error(
        "CPU gemm_rcr_bias_permute_m2n3: invalid dimensions");
  }

  const float* a_ptr =
      static_cast<const float*>(a);

  const float* b_ptr =
      static_cast<const float*>(b);

  const float* bias_ptr =
      static_cast<const float*>(bias);

  float* output_ptr =
      static_cast<float*>(output);

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

  /*
   * Keep the existing QKV post-GEMM permute path unchanged.
   * The optimization here only replaces per-inference dynamic
   * weight packing with persistent static packed operators.
   */
  thread_local std::vector<float> gemm_scratch;
  gemm_scratch.resize(m * n);

  auto& context =
      {{func_name}}_get_cache().get(
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
      "xnn_run_operator(fully_connected)");

  // m2n3:
  //
  // [M, N]
  //   ->
  // [M0, M1, N0, N1, N2]
  //   ->
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

  // Each source GEMM row is independent. Parallelizing over M
  // gives 128 coarse tasks for BERT instead of dispatching one task
  // for every 64-float head chunk.
  struct qkv_permute_context {
    const float* input;
    float* output;
    size_t n;
    size_t t1;
    size_t t2;
    size_t t3;
    size_t m0_size;
    size_t n2_size;
  };

  qkv_permute_context permute_context{
      gemm_scratch.data(),
      output_ptr,
      n,
      t1,
      t2,
      t3,
      m0_size,
      n2_size};

  auto permute_row_task =
      [](void* raw_context, size_t src_row) {
        auto* task_context =
            static_cast<qkv_permute_context*>(
                raw_context);

        const size_t m0 =
            src_row / task_context->t1;

        const size_t m1 =
            src_row % task_context->t1;

        for (size_t n0 = 0;
             n0 < task_context->t2;
             ++n0) {
          for (size_t n1 = 0;
               n1 < task_context->t3;
               ++n1) {
            const size_t src =
                src_row * task_context->n +
                (n0 * task_context->t3 + n1) *
                    task_context->n2_size;

            const size_t dst =
                ((((n0 * task_context->m0_size + m0) *
                    task_context->t3 + n1) *
                   task_context->t1 + m1) *
                 task_context->n2_size);

            std::memcpy(
                task_context->output + dst,
                task_context->input + src,
                task_context->n2_size *
                    sizeof(float));
          }
        }
      };

  ait::parallelize_1d(
      permute_row_task,
      &permute_context,
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
    void* output,
    size_t m,
    size_t n,
    size_t k,
    size_t t1,
    size_t t2,
    size_t t3,
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
{{indent}}    {{output}},
{{indent}}    {{m}},
{{indent}}    {{n}},
{{indent}}    {{k}},
{{indent}}    {{t1}},
{{indent}}    {{t2}},
{{indent}}    {{t3}},
{{indent}}    {{cache_id}},
{{indent}}    {{use_input_scratch}},
{{indent}}    stream);
"""
)


def _static_int(value) -> int:
    if isinstance(value, IntImm):
        return int(value._attrs["values"][0])
    return int(value)


def _needs_input_scratch(tensor) -> bool:
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
        cache_id=str(cache_id_from_tensor_name(b._attrs["name"])) + "ULL",
        use_input_scratch=(
            "true"
            if _needs_input_scratch(a)
            else "false"
        ),
        indent=indent,
    )
