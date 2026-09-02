# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Helpers for persistent CPU XNNPACK static-FC caches."""

import os

import jinja2


def cache_id_from_tensor_name(name: str) -> int:
    """Return a deterministic non-zero 64-bit cache id for a tensor name."""

    # FNV-1a 64-bit. We embed the result as a literal in generated C++,
    # so Python's randomized hash() must not be used here.
    value = 14695981039346656037

    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    return value or 1


def use_int8_static_fc() -> bool:
    """Return whether CPU static FC codegen should use dynamic-QD8 INT8 GEMM.

    The switch is intentionally compile-time: each generated model packs only
    the selected FP32 or INT8 weights, rather than keeping both representations
    in memory.  No matrix dimensions are encoded here, so the same path works
    for BERT-base, BERT-large, and other compatible static FC shapes.
    """

    value = os.environ.get("AIT_CPU_INT8_FC", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


_STATIC_FC_CONTEXT_TEMPLATE = jinja2.Template(
    r"""
static constexpr bool {{func_name}}_use_int8_static_fc =
    {{use_int8}};


struct {{func_name}}_xnn_context {
  xnn_operator_t fc_op = nullptr;
  xnn_operator_t convert_op = nullptr;

  uint64_t cache_id = 0;
  size_t cached_m = 0;
  size_t n = 0;
  size_t k = 0;

  std::vector<int8_t> qinput;
  std::vector<xnn_quantization_params> qparams;

  void* workspace = nullptr;
  size_t workspace_capacity = 0;

  {{func_name}}_xnn_context(
      uint64_t id,
      const float* weight,
      const float* bias,
      size_t output_channels,
      size_t input_channels)
      : cache_id(id),
        n(output_channels),
        k(input_channels) {

    if (weight == nullptr || bias == nullptr) {
      throw std::runtime_error(
          "CPU static FC creation requires raw weight and bias");
    }

    if constexpr ({{func_name}}_use_int8_static_fc) {
      /*
       * Static weight quantization is done exactly once while the model
       * constants are still available.  XNNPACK synchronously packs the
       * quantized weight/bias during operator creation, so the temporary
       * qweight/scale vectors can be released when this constructor returns.
       *
       * Per-output-channel symmetric INT8 keeps this path shape-agnostic:
       * BERT-base (768/3072), BERT-large (1024/4096), and other N/K values
       * all use the same code.
       */
      std::vector<int8_t> qweight(n * k);
      std::vector<float> weight_scale(n);

      for (size_t oc = 0; oc < n; ++oc) {
        float max_abs = 0.0f;

        const float* weight_row =
            weight + oc * k;

        for (size_t ic = 0; ic < k; ++ic) {
          max_abs = std::max(
              max_abs,
              std::abs(weight_row[ic]));
        }

        const float scale =
            max_abs > 0.0f
                ? max_abs / 127.0f
                : 1.0f;

        weight_scale[oc] = scale;

        int8_t* qweight_row =
            qweight.data() + oc * k;

        for (size_t ic = 0; ic < k; ++ic) {
          int value = static_cast<int>(
              std::nearbyint(
                  weight_row[ic] / scale));

          value = std::max(
              -127,
              std::min(127, value));

          qweight_row[ic] =
              static_cast<int8_t>(value);
        }
      }

      {{func_name}}_check_xnn_status(
          xnn_create_convert_nc_f32_qd8(
              0,
              &convert_op),
          "xnn_create_convert_nc_f32_qd8");

      {{func_name}}_check_xnn_status(
          xnn_create_fully_connected_nc_qd8_f32_qc8w(
              k,
              n,
              k,
              n,
              weight_scale.data(),
              qweight.data(),
              bias,
              -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity(),
              0,
              nullptr,
              &fc_op),
          "xnn_create_fully_connected_nc_qd8_f32_qc8w");
    } else {
      {{func_name}}_check_xnn_status(
          xnn_create_fully_connected_nc_f32(
              k,
              n,
              k,
              n,
              weight,
              bias,
              -std::numeric_limits<float>::infinity(),
              +std::numeric_limits<float>::infinity(),
              0,
              nullptr,
              &fc_op),
          "xnn_create_fully_connected_nc_f32");
    }
  }

  ~{{func_name}}_xnn_context() {
    if (fc_op != nullptr) {
      xnn_delete_operator(fc_op);
    }

    if (convert_op != nullptr) {
      xnn_delete_operator(convert_op);
    }

    std::free(workspace);
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

  void ensure_workspace(size_t required_size) {
    if (required_size <= workspace_capacity) {
      return;
    }

    const size_t alignment = 64;
    const size_t rounded_size =
        (required_size + alignment - 1) &
        ~(alignment - 1);

    void* new_workspace = nullptr;

    if (posix_memalign(
            &new_workspace,
            alignment,
            rounded_size) != 0) {
      throw std::bad_alloc();
    }

    std::free(workspace);
    workspace = new_workspace;
    workspace_capacity = rounded_size;
  }

  void reshape(size_t m) {
    if (cached_m == m) {
      return;
    }

    if constexpr ({{func_name}}_use_int8_static_fc) {
      const size_t qinput_elements =
          m * k + XNN_EXTRA_BYTES;

      qinput.resize(qinput_elements);
      qparams.resize(
          m + XNN_EXTRA_QUANTIZATION_PARAMS);

      {{func_name}}_check_xnn_status(
          xnn_reshape_convert_nc_f32_qd8(
              convert_op,
              m,
              k,
              k,
              k,
              ait::cpu_threadpool()),
          "xnn_reshape_convert_nc_f32_qd8");

      size_t required_workspace_size = 0;

      {{func_name}}_check_xnn_status(
          xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
              fc_op,
              m,
              &required_workspace_size,
              ait::cpu_threadpool()),
          "xnn_reshape_fully_connected_nc_qd8_f32_qc8w");

      ensure_workspace(required_workspace_size);
    } else {
      {{func_name}}_check_xnn_status(
          xnn_reshape_fully_connected_nc_f32(
              fc_op,
              m,
              ait::cpu_threadpool()),
          "xnn_reshape_fully_connected_nc_f32");
    }

    cached_m = m;
  }

  void run(
      const float* input,
      float* output) {
    if constexpr ({{func_name}}_use_int8_static_fc) {
      {{func_name}}_check_xnn_status(
          xnn_setup_convert_nc_f32_qd8(
              convert_op,
              input,
              qinput.data(),
              nullptr,
              qparams.data()),
          "xnn_setup_convert_nc_f32_qd8");

      {{func_name}}_check_xnn_status(
          xnn_run_operator(
              convert_op,
              ait::cpu_threadpool()),
          "xnn_run_operator(convert_f32_qd8)");

      {{func_name}}_check_xnn_status(
          xnn_setup_fully_connected_nc_qd8_f32_qc8w(
              fc_op,
              qinput.data(),
              output,
              workspace,
              qparams.data()),
          "xnn_setup_fully_connected_nc_qd8_f32_qc8w");

      {{func_name}}_check_xnn_status(
          xnn_run_operator(
              fc_op,
              ait::cpu_threadpool()),
          "xnn_run_operator(fully_connected_qd8_qc8w)");
    } else {
      {{func_name}}_check_xnn_status(
          xnn_setup_fully_connected_nc_f32(
              fc_op,
              input,
              output),
          "xnn_setup_fully_connected_nc_f32");

      {{func_name}}_check_xnn_status(
          xnn_run_operator(
              fc_op,
              ait::cpu_threadpool()),
          "xnn_run_operator(fully_connected_f32)");
    }
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
          "CPU static FC cache miss after raw weight release");
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
"""
)


def render_static_fc_context(func_name: str) -> str:
    """Render the shared FP32/INT8 persistent XNNPACK FC context."""

    return _STATIC_FC_CONTEXT_TEMPLATE.render(
        func_name=func_name,
        use_int8="true" if use_int8_static_fc() else "false",
    )
