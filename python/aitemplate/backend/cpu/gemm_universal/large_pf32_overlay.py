# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

# Experimental BERT-large packed-LHS dispatch for CPU XNNPACK.
#
# Conservative behavior:
# - only QKV / FFN1 / FFN2 BERT-large shapes are changed;
# - runtime use is opt-in with AIT_CPU_LARGE_PF32=1;
# - missing PF32 weak symbols automatically keep F32;
# - BERT-base and 1024->1024 projection remain unchanged.

from aitemplate.backend import registry


_FAST_KEY = "cpu.gemm_rcr_bias_fast_gelu.gen_function"
_ADD_KEY = "cpu.gemm_rcr_bias_add.gen_function"
_QKV_KEY = "cpu.gemm_rcr_bias_permute_m2n3.gen_function"

_BASE_FAST_GEN = registry.get(_FAST_KEY)
_BASE_ADD_GEN = registry.get(_ADD_KEY)
_BASE_QKV_GEN = registry.get(_QKV_KEY)


_CPP_DISPATCH = r'''
// ---- AIT CPU BERT-large PF32 experimental dispatch -----------------
#include <cstdlib>
#include <cstdint>
#include <vector>

extern "C" {
enum xnn_status xnn_create_fully_connected_nc_pf32(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out)
    __attribute__((weak));

enum xnn_status xnn_reshape_fully_connected_nc_pf32(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    size_t* workspace_size,
    pthreadpool_t threadpool)
    __attribute__((weak));

enum xnn_status xnn_setup_fully_connected_nc_pf32(
    xnn_operator_t fully_connected_op,
    const float* input,
    float* output,
    void* workspace)
    __attribute__((weak));
}

namespace ait_cpu_large_pf32 {

struct FcState {
  xnn_operator_t op = nullptr;
  size_t workspace_size = 0;
};

inline thread_local std::vector<FcState> g_states;
inline thread_local std::vector<uint8_t> g_workspace;

inline bool requested() {
  const char* value = std::getenv("AIT_CPU_LARGE_PF32");
  return value != nullptr &&
         value[0] != '\0' &&
         value[0] != '0';
}

inline bool symbols_available() {
  return xnn_create_fully_connected_nc_pf32 != nullptr &&
         xnn_reshape_fully_connected_nc_pf32 != nullptr &&
         xnn_setup_fully_connected_nc_pf32 != nullptr;
}

inline bool target_shape(
    size_t input_channels,
    size_t output_channels) {
  return
      (input_channels == 1024 && output_channels == 3072) ||
      (input_channels == 1024 && output_channels == 4096) ||
      (input_channels == 4096 && output_channels == 1024);
}

inline FcState* find_state(xnn_operator_t op) {
  for (auto& state : g_states) {
    if (state.op == op) {
      return &state;
    }
  }
  return nullptr;
}

inline void* workspace_ptr(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  constexpr uintptr_t alignment = 64;
  g_workspace.resize(bytes + alignment - 1);

  const uintptr_t raw =
      reinterpret_cast<uintptr_t>(g_workspace.data());

  const uintptr_t aligned =
      (raw + alignment - 1) & ~(alignment - 1);

  return reinterpret_cast<void*>(aligned);
}

inline enum xnn_status create_fc(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {

  if (requested() &&
      symbols_available() &&
      target_shape(input_channels, output_channels)) {

    xnn_operator_t packed_op = nullptr;

    const enum xnn_status status =
        xnn_create_fully_connected_nc_pf32(
            input_channels,
            output_channels,
            input_stride,
            output_stride,
            kernel,
            bias,
            output_min,
            output_max,
            flags,
            weights_cache,
            &packed_op);

    if (status == xnn_status_success) {
      *fully_connected_op_out = packed_op;
      g_states.push_back(FcState{packed_op, 0});
      return status;
    }
  }

  return ::xnn_create_fully_connected_nc_f32(
      input_channels,
      output_channels,
      input_stride,
      output_stride,
      kernel,
      bias,
      output_min,
      output_max,
      flags,
      weights_cache,
      fully_connected_op_out);
}

inline enum xnn_status reshape_fc(
    xnn_operator_t op,
    size_t batch_size,
    pthreadpool_t threadpool) {

  FcState* state = find_state(op);

  if (state == nullptr) {
    return ::xnn_reshape_fully_connected_nc_f32(
        op,
        batch_size,
        threadpool);
  }

  size_t workspace_size = 0;

  const enum xnn_status status =
      xnn_reshape_fully_connected_nc_pf32(
          op,
          batch_size,
          &workspace_size,
          threadpool);

  if (status == xnn_status_success) {
    state->workspace_size = workspace_size;
  }

  return status;
}

inline enum xnn_status setup_fc(
    xnn_operator_t op,
    const float* input,
    float* output) {

  FcState* state = find_state(op);

  if (state == nullptr) {
    return ::xnn_setup_fully_connected_nc_f32(
        op,
        input,
        output);
  }

  return xnn_setup_fully_connected_nc_pf32(
      op,
      input,
      output,
      workspace_ptr(state->workspace_size));
}

}  // namespace ait_cpu_large_pf32

#define xnn_create_fully_connected_nc_f32 \
    ait_cpu_large_pf32::create_fc
#define xnn_reshape_fully_connected_nc_f32 \
    ait_cpu_large_pf32::reshape_fc
#define xnn_setup_fully_connected_nc_f32 \
    ait_cpu_large_pf32::setup_fc
// -------------------------------------------------------------------
'''


def _static_dim(dim):
    if hasattr(dim, "value"):
        try:
            return int(dim.value())
        except Exception:
            pass

    attrs = getattr(dim, "_attrs", {})
    values = attrs.get("values")
    if values is not None and len(values) == 1:
        return int(values[0])

    return None


def _kn(func_attrs):
    a = func_attrs["inputs"][0]
    b = func_attrs["inputs"][1]
    k = _static_dim(a._attrs["shape"][-1])
    n = _static_dim(b._attrs["shape"][0])
    return k, n


def _inject(code):
    marker = "#include <xnnpack.h>"

    if "AIT CPU BERT-large PF32 experimental dispatch" in code:
        return code

    if marker not in code:
        raise RuntimeError(
            "large_pf32_overlay: generated CPU source has no xnnpack include"
        )

    return code.replace(
        marker,
        marker + "\n" + _CPP_DISPATCH,
        1,
    )


def _fast_gen(
    func_attrs,
    exec_cond_template=None,
    dim_info_dict=None,
):
    code = _BASE_FAST_GEN(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    )

    if _kn(func_attrs) == (1024, 4096):
        return _inject(code)

    return code


def _add_gen(
    func_attrs,
    exec_cond_template=None,
    dim_info_dict=None,
):
    code = _BASE_ADD_GEN(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    )

    if (
        bool(func_attrs.get("fused_layernorm", False))
        and _kn(func_attrs) == (4096, 1024)
    ):
        return _inject(code)

    return code


def _qkv_gen(
    func_attrs,
    exec_cond_template=None,
    dim_info_dict=None,
):
    code = _BASE_QKV_GEN(
        func_attrs,
        exec_cond_template,
        dim_info_dict,
    )

    if _kn(func_attrs) == (1024, 3072):
        return _inject(code)

    return code


registry.BACKEND_FUNCTIONS[_FAST_KEY] = _fast_gen
registry.BACKEND_FUNCTIONS[_ADD_KEY] = _add_gen
registry.BACKEND_FUNCTIONS[_QKV_KEY] = _qkv_gen
