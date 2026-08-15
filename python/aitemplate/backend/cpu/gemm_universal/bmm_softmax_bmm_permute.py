# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""CPU bmm_softmax_bmm_permute codegen backed by XNNPACK."""

from collections import OrderedDict
from typing import Any, Dict

import jinja2

from aitemplate.backend import registry
from aitemplate.backend.cpu.gemm_universal.gemm_rcr_bias import _dim_expr
from aitemplate.compiler.base import IntImm
from aitemplate.compiler.dtype import normalize_dtype


FUNC_TEMPLATE = jinja2.Template(
    r"""
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
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

class {{func_name}}_workspace {
 public:
  ~{{func_name}}_workspace() {
    std::free(data_);
  }

  void* get(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }

    if (bytes <= capacity_) {
      return data_;
    }

    std::free(data_);
    data_ = nullptr;
    capacity_ = 0;

    constexpr size_t alignment = 64;
    const size_t rounded_bytes =
        ((bytes + alignment - 1) / alignment) * alignment;

    void* new_data = nullptr;
    if (posix_memalign(
            &new_data,
            alignment,
            rounded_bytes) != 0) {
      throw std::bad_alloc();
    }

    data_ = new_data;
    capacity_ = rounded_bytes;
    return data_;
  }

 private:
  void* data_ = nullptr;
  size_t capacity_ = 0;
};

struct {{func_name}}_xnn_context {
  xnn_operator_t qk_op = nullptr;
  xnn_operator_t softmax_op = nullptr;
  xnn_operator_t av_op = nullptr;

  {{func_name}}_xnn_context() {
    {{func_name}}_check_xnn_status(
        xnn_initialize(nullptr),
        "xnn_initialize");

    {{func_name}}_check_xnn_status(
        xnn_create_batch_matrix_multiply_nc_f32(
            XNN_FLAG_TRANSPOSE_B,
            &qk_op),
        "xnn_create_batch_matrix_multiply_nc_f32(QK)");

    {{func_name}}_check_xnn_status(
        xnn_create_softmax_nc_f32(
            0,
            &softmax_op),
        "xnn_create_softmax_nc_f32");

    {{func_name}}_check_xnn_status(
        xnn_create_batch_matrix_multiply_nc_f32(
            0,
            &av_op),
        "xnn_create_batch_matrix_multiply_nc_f32(AV)");
  }

  ~{{func_name}}_xnn_context() {
    if (qk_op != nullptr) {
      xnn_delete_operator(qk_op);
    }
    if (softmax_op != nullptr) {
      xnn_delete_operator(softmax_op);
    }
    if (av_op != nullptr) {
      xnn_delete_operator(av_op);
    }
  }

  {{func_name}}_xnn_context(
      const {{func_name}}_xnn_context&) = delete;
  {{func_name}}_xnn_context& operator=(
      const {{func_name}}_xnn_context&) = delete;
};

}  // namespace


{{func_signature}}
{
  (void)stream;

  if (batch_heads == 0 ||
      m == 0 ||
      n == 0 ||
      k_dim == 0 ||
      o == 0) {
    return;
  }

  if (num_heads == 0 || batch_heads % num_heads != 0) {
    throw std::runtime_error(
        "CPU bmm_softmax_bmm_permute: invalid number of heads");
  }

  const float* q_ptr = static_cast<const float*>(q);
  const float* k_ptr = static_cast<const float*>(k_tensor);
  const float* v_ptr = static_cast<const float*>(v);
  float* output_ptr = static_cast<float*>(output);

  const size_t batch = batch_heads / num_heads;

  const size_t q_elements =
      batch_heads * m * k_dim;
  const size_t attention_elements =
      batch_heads * m * n;
  const size_t score_elements =
      batch_heads * m * o;

  const size_t extra_elements =
      (XNN_EXTRA_BYTES + sizeof(float) - 1) /
      sizeof(float);

  thread_local {{func_name}}_xnn_context context;
  thread_local {{func_name}}_workspace workspace;

  // Scaling Q before Q*K^T is equivalent to scaling
  // the attention logits afterwards, but touches fewer elements.
  thread_local std::vector<float> q_scaled;
  q_scaled.resize(q_elements + extra_elements);

  for (size_t i = 0; i < q_elements; ++i) {
    q_scaled[i] = q_ptr[i] * scale;
  }

  std::fill(
      q_scaled.begin() + q_elements,
      q_scaled.end(),
      0.0f);

  // Q*K^T output and softmax output need XNNPACK tail padding
  // because they are used as inputs to following XNNPACK operators.
  thread_local std::vector<float> logits;
  thread_local std::vector<float> probabilities;

  logits.resize(attention_elements + extra_elements);
  probabilities.resize(attention_elements + extra_elements);

  std::fill(
      logits.begin() + attention_elements,
      logits.end(),
      0.0f);

  std::fill(
      probabilities.begin() + attention_elements,
      probabilities.end(),
      0.0f);

  const size_t batch_dims[1] = {batch_heads};

  // ------------------------------------------------------------
  // 1. Q * K^T
  // Q: [BH, M, K]
  // K: [BH, N, K]
  // output: [BH, M, N]
  // ------------------------------------------------------------
  size_t workspace_size = 0;

  {{func_name}}_check_xnn_status(
      xnn_reshape_batch_matrix_multiply_nc_f32(
          context.qk_op,
          1,
          batch_dims,
          batch_dims,
          m,
          k_dim,
          n,
          &workspace_size,
          nullptr),
      "xnn_reshape_batch_matrix_multiply_nc_f32(QK)");

  void* workspace_ptr = workspace.get(workspace_size);

  {{func_name}}_check_xnn_status(
      xnn_setup_batch_matrix_multiply_nc_f32(
          context.qk_op,
          workspace_ptr,
          q_scaled.data(),
          k_ptr,
          logits.data()),
      "xnn_setup_batch_matrix_multiply_nc_f32(QK)");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          context.qk_op,
          nullptr),
      "xnn_run_operator(QK)");

  std::fill(
      logits.begin() + attention_elements,
      logits.end(),
      0.0f);

  // ------------------------------------------------------------
  // 2. Softmax over N
  // Treat [BH, M, N] as [BH*M, N].
  // ------------------------------------------------------------
  {{func_name}}_check_xnn_status(
      xnn_reshape_softmax_nc_f32(
          context.softmax_op,
          n,
          n,
          n,
          batch_heads * m,
          nullptr),
      "xnn_reshape_softmax_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_setup_softmax_nc_f32(
          context.softmax_op,
          logits.data(),
          probabilities.data()),
      "xnn_setup_softmax_nc_f32");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          context.softmax_op,
          nullptr),
      "xnn_run_operator(softmax)");

  std::fill(
      probabilities.begin() + attention_elements,
      probabilities.end(),
      0.0f);

  // ------------------------------------------------------------
  // 3. Attention * V
  // probabilities: [BH, M, N]
  // V:             [BH, N, O]
  // raw_score:     [BH, M, O]
  // ------------------------------------------------------------
  thread_local std::vector<float> raw_score;
  raw_score.resize(score_elements);

  workspace_size = 0;

  {{func_name}}_check_xnn_status(
      xnn_reshape_batch_matrix_multiply_nc_f32(
          context.av_op,
          1,
          batch_dims,
          batch_dims,
          m,
          n,
          o,
          &workspace_size,
          nullptr),
      "xnn_reshape_batch_matrix_multiply_nc_f32(AV)");

  workspace_ptr = workspace.get(workspace_size);

  {{func_name}}_check_xnn_status(
      xnn_setup_batch_matrix_multiply_nc_f32(
          context.av_op,
          workspace_ptr,
          probabilities.data(),
          v_ptr,
          raw_score.data()),
      "xnn_setup_batch_matrix_multiply_nc_f32(AV)");

  {{func_name}}_check_xnn_status(
      xnn_run_operator(
          context.av_op,
          nullptr),
      "xnn_run_operator(AV)");

  // ------------------------------------------------------------
  // 4. Physical layout:
  //
  // raw_score:
  //   [B, H, M, O]
  //
  // AITemplate bmm_softmax_bmm_permute expects:
  //   [B, M, H, O]
  //
  // The compiler represents the underlying tensor as [BH, M, O]
  // and then applies a reshape, so this permutation must happen
  // physically inside the backend implementation.
  // ------------------------------------------------------------
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < num_heads; ++h) {
      for (size_t row = 0; row < m; ++row) {
        const size_t src =
            (((b * num_heads + h) * m + row) * o);

        const size_t dst =
            (((b * m + row) * num_heads + h) * o);

        std::memcpy(
            output_ptr + dst,
            raw_score.data() + src,
            o * sizeof(float));
      }
    }
  }
}
"""
)


FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(
    const void* q,
    const void* k_tensor,
    const void* v,
    void* output,
    size_t batch_heads,
    size_t m,
    size_t n,
    size_t k_dim,
    size_t o,
    size_t num_heads,
    float scale,
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
{{indent}}    {{q}},
{{indent}}    {{k_tensor}},
{{indent}}    {{v}},
{{indent}}    {{output}},
{{indent}}    {{batch_heads}},
{{indent}}    {{m}},
{{indent}}    {{n}},
{{indent}}    {{k_dim}},
{{indent}}    {{o}},
{{indent}}    {{num_heads}},
{{indent}}    {{scale}},
{{indent}}    stream);
"""
)


def _static_int(value) -> int:
    if isinstance(value, IntImm):
        return int(value._attrs["values"][0])
    return int(value)


def _float_literal(value) -> str:
    text = repr(float(value))
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def _input_ptr(tensor, accessor) -> str:
    name = tensor._attrs["name"]

    if (
        accessor.is_from_strided_tensor
        and not accessor.is_contiguous
    ):
        raise NotImplementedError(
            "CPU bmm_softmax_bmm_permute currently "
            "requires contiguous fused inputs"
        )

    offset = accessor.offset

    if offset == 0:
        return name

    return (
        f"(static_cast<const float*>({name}) "
        f"+ {offset})"
    )


def _validate(func_attrs: Dict[str, Any]) -> None:
    if len(func_attrs["inputs"]) != 3:
        raise NotImplementedError(
            "CPU bmm_softmax_bmm_permute requires Q, K and V"
        )

    q, k_tensor, v = func_attrs["inputs"]
    accessors = func_attrs["input_accessors"]

    for name, tensor, accessor in (
        ("Q", q, accessors[0]),
        ("K", k_tensor, accessors[1]),
        ("V", v, accessors[2]),
    ):
        logical_shape = accessor.original_shapes

        if len(logical_shape) != 3:
            raise NotImplementedError(
                "CPU bmm_softmax_bmm_permute requires "
                f"logical {name} rank 3"
            )

        dtype = normalize_dtype(tensor._attrs["dtype"])
        if dtype != "float32":
            raise NotImplementedError(
                "CPU bmm_softmax_bmm_permute currently supports "
                f"only float32; got {name} dtype={tensor._attrs['dtype']}"
            )

        if (
            accessor.is_from_strided_tensor
            and not accessor.is_contiguous
        ):
            raise NotImplementedError(
                "CPU bmm_softmax_bmm_permute currently "
                "requires contiguous fused inputs"
            )

    if func_attrs.get("layout") != "Permute4DBMM_0213":
        raise NotImplementedError(
            "CPU bmm_softmax_bmm_permute currently supports "
            "only Permute4DBMM_0213"
        )

    shape = func_attrs.get("shape")
    if shape is None or len(shape) != 1:
        raise NotImplementedError(
            "CPU bmm_softmax_bmm_permute requires shape=(num_heads,)"
        )

    num_heads = _static_int(shape[0])
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")


@registry.reg("cpu.bmm_softmax_bmm_permute.config")
def config(
    func_attrs: Dict[str, Any],
    dtype="float32",
) -> None:
    _validate(func_attrs)

    func_attrs["op_instance"] = OrderedDict(
        [
            ("xnnpack", None),
        ]
    )


@registry.reg("cpu.bmm_softmax_bmm_permute.filter")
def function_filter(
    cfg,
    func_attrs,
    ab_alignment,
) -> bool:
    return cfg == "xnnpack"


@registry.reg("cpu.bmm_softmax_bmm_permute.gen_profiler")
def gen_profiler(
    func_attrs,
    workdir,
    *args,
    **kwargs,
):
    # XNNPACK performs microkernel selection internally.
    return None


@registry.reg("cpu.bmm_softmax_bmm_permute.gen_function")
def gen_function(
    func_attrs: Dict[str, Any],
    exec_cond_template=None,
    dim_info_dict=None,
) -> str:
    _validate(func_attrs)

    return FUNC_TEMPLATE.render(
        func_name=func_attrs["name"],
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"]
        ),
    )


@registry.reg("cpu.bmm_softmax_bmm_permute.func_decl")
def gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    _validate(func_attrs)

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"]
        )
    ).strip()


@registry.reg("cpu.bmm_softmax_bmm_permute.func_call")
def gen_function_call(
    func_attrs: Dict[str, Any],
    indent="  ",
) -> str:
    _validate(func_attrs)

    q, k_tensor, v = func_attrs["inputs"]
    output = func_attrs["outputs"][0]

    q_accessor = func_attrs["input_accessors"][0]
    k_accessor = func_attrs["input_accessors"][1]
    v_accessor = func_attrs["input_accessors"][2]

    q_shape = q_accessor.original_shapes
    k_shape = k_accessor.original_shapes
    v_shape = v_accessor.original_shapes

    num_heads = _static_int(func_attrs["shape"][0])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        q=_input_ptr(q, q_accessor),
        k_tensor=_input_ptr(k_tensor, k_accessor),
        v=_input_ptr(v, v_accessor),
        output=output._attrs["name"],
        batch_heads=_dim_expr(q_shape[0]),
        m=_dim_expr(q_shape[1]),
        n=_dim_expr(k_shape[1]),
        k_dim=_dim_expr(q_shape[2]),
        o=_dim_expr(v_shape[2]),
        num_heads=str(num_heads),
        scale=_float_literal(func_attrs["scale"]),
        indent=indent,
    )
