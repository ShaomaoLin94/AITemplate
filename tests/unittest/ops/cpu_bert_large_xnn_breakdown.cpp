#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <xnnpack.h>

namespace {

volatile float g_sink = 0.0f;

void check_xnn(xnn_status status, const char* step) {
  if (status != xnn_status_success) {
    std::cerr << step << " failed with XNNPACK status "
              << static_cast<int>(status) << std::endl;
    std::exit(2);
  }
}

struct Stats {
  double median_ms;
  double mean_ms;
  double min_ms;
  double max_ms;
};

template <class Fn>
Stats benchmark(Fn&& fn, int warmup = 5, int iterations = 30) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }

  std::vector<double> times;
  times.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    times.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());

  double median = 0.0;
  if (iterations % 2 == 0) {
    median = 0.5 * (sorted[iterations / 2 - 1] + sorted[iterations / 2]);
  } else {
    median = sorted[iterations / 2];
  }

  const double sum = std::accumulate(times.begin(), times.end(), 0.0);
  return {
      median,
      sum / static_cast<double>(iterations),
      *std::min_element(times.begin(), times.end()),
      *std::max_element(times.begin(), times.end()),
  };
}

void fill_data(std::vector<float>& values, int seed) {
  for (size_t i = 0; i < values.size(); ++i) {
    const int x = static_cast<int>((i * 17 + seed * 13) % 101) - 50;
    values[i] = static_cast<float>(x) * 0.0004f;
  }
}

struct FcBench {
  size_t m;
  size_t n;
  size_t k;
  std::vector<float> weight;
  std::vector<float> bias;
  xnn_operator_t op = nullptr;

  FcBench(size_t m_, size_t n_, size_t k_, int seed)
      : m(m_),
        n(n_),
        k(k_),
        weight(n * k),
        bias(n) {
    fill_data(weight, seed);
    fill_data(bias, seed + 1);

    check_xnn(
        xnn_create_fully_connected_nc_f32(
            k,
            n,
            k,
            n,
            weight.data(),
            bias.data(),
            -std::numeric_limits<float>::infinity(),
            +std::numeric_limits<float>::infinity(),
            0,
            nullptr,
            &op),
        "xnn_create_fully_connected_nc_f32");

    check_xnn(
        xnn_reshape_fully_connected_nc_f32(
            op,
            m,
            nullptr),
        "xnn_reshape_fully_connected_nc_f32");
  }

  ~FcBench() {
    if (op != nullptr) {
      xnn_delete_operator(op);
    }
  }

  void run(const float* input, float* output) {
    check_xnn(
        xnn_setup_fully_connected_nc_f32(
            op,
            input,
            output),
        "xnn_setup_fully_connected_nc_f32");

    check_xnn(
        xnn_run_operator(op, nullptr),
        "xnn_run_operator(fully_connected)");
  }
};

void approx_gelu_inplace(float* data, size_t m, size_t n) {
  union xnn_unary_params unary_params = {};
  const struct xnn_quantization_params quantization = {0, 1.0f};

  check_xnn(
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
          nullptr,
          data,
          data),
      "xnn_run_unary_elementwise_nc(approxgelu)");
}

void residual_layernorm_inplace(
    float* output,
    const float* residual,
    const float* gamma,
    const float* beta,
    size_t m,
    size_t n,
    float eps) {
  for (size_t row = 0; row < m; ++row) {
    const size_t offset = row * n;
    float* output_row = output + offset;
    const float* residual_row = residual + offset;

    double sum = 0.0;
    double square_sum = 0.0;

    for (size_t col = 0; col < n; ++col) {
      const float value = output_row[col] + residual_row[col];
      output_row[col] = value;
      const double value_d = static_cast<double>(value);
      sum += value_d;
      square_sum += value_d * value_d;
    }

    const double inv_n = 1.0 / static_cast<double>(n);
    const double mean = sum * inv_n;
    double variance = square_sum * inv_n - mean * mean;
    variance = std::max(variance, 0.0);

    const float inv_std = static_cast<float>(
        1.0 / std::sqrt(variance + static_cast<double>(eps)));
    const float mean_f = static_cast<float>(mean);

    for (size_t col = 0; col < n; ++col) {
      output_row[col] =
          (output_row[col] - mean_f) * inv_std * gamma[col] + beta[col];
    }
  }
}

void qkv_permute(
    const float* input,
    float* output,
    size_t m,
    size_t n,
    size_t t1,
    size_t t2,
    size_t t3) {
  const size_t m0_size = m / t1;
  const size_t n2_size = n / (t2 * t3);

  for (size_t src_row = 0; src_row < m; ++src_row) {
    const size_t m0 = src_row / t1;
    const size_t m1 = src_row % t1;

    for (size_t n0 = 0; n0 < t2; ++n0) {
      for (size_t n1 = 0; n1 < t3; ++n1) {
        const size_t src =
            src_row * n + (n0 * t3 + n1) * n2_size;

        const size_t dst =
            ((((n0 * m0_size + m0) * t3 + n1) * t1 + m1) * n2_size);

        std::memcpy(
            output + dst,
            input + src,
            n2_size * sizeof(float));
      }
    }
  }
}

void print_stats(const std::string& name, const Stats& s) {
  std::cout << name
            << " median=" << s.median_ms
            << " mean=" << s.mean_ms
            << " min=" << s.min_ms
            << " max=" << s.max_ms
            << std::endl;
}

}  // namespace

int main() {
  check_xnn(xnn_initialize(nullptr), "xnn_initialize");

  constexpr size_t M = 128;
  constexpr size_t H = 1024;
  constexpr size_t FFN = 4096;
  constexpr size_t QKV = 3072;
  constexpr size_t HEADS = 16;
  constexpr size_t HEAD_DIM = 64;
  constexpr float EPS = 1.0e-12f;

  const size_t extra =
      (XNN_EXTRA_BYTES + sizeof(float) - 1) / sizeof(float);

  std::vector<float> input_h(M * H + extra);
  std::vector<float> input_ffn(M * FFN + extra);
  std::vector<float> out_ffn1(M * FFN + extra);
  std::vector<float> out_h(M * H + extra);
  std::vector<float> qkv_scratch(M * QKV + extra);
  std::vector<float> qkv_output(M * QKV + extra);
  std::vector<float> residual(M * H + extra);
  std::vector<float> gamma(H, 1.0f);
  std::vector<float> beta(H, 0.0f);

  fill_data(input_h, 1);
  fill_data(input_ffn, 2);
  fill_data(out_ffn1, 3);
  fill_data(out_h, 4);
  fill_data(qkv_scratch, 5);
  fill_data(qkv_output, 6);
  fill_data(residual, 7);

  FcBench qkv_fc(M, QKV, H, 10);
  FcBench proj_fc(M, H, H, 20);
  FcBench ffn1_fc(M, FFN, H, 30);
  FcBench ffn2_fc(M, H, FFN, 40);

  const Stats qkv_core = benchmark([&]() {
    qkv_fc.run(input_h.data(), qkv_scratch.data());
    g_sink += qkv_scratch[0];
  });

  const Stats qkv_post = benchmark([&]() {
    qkv_permute(
        qkv_scratch.data(),
        qkv_output.data(),
        M,
        QKV,
        128,
        3,
        HEADS);
    g_sink += qkv_output[HEAD_DIM];
  });

  const Stats qkv_full = benchmark([&]() {
    qkv_fc.run(input_h.data(), qkv_scratch.data());
    qkv_permute(
        qkv_scratch.data(),
        qkv_output.data(),
        M,
        QKV,
        128,
        3,
        HEADS);
    g_sink += qkv_output[HEAD_DIM];
  });

  const Stats proj_core = benchmark([&]() {
    proj_fc.run(input_h.data(), out_h.data());
    g_sink += out_h[0];
  });

  const Stats residual_ln_post = benchmark([&]() {
    residual_layernorm_inplace(
        out_h.data(),
        residual.data(),
        gamma.data(),
        beta.data(),
        M,
        H,
        EPS);
    g_sink += out_h[0];
  });

  const Stats proj_full = benchmark([&]() {
    proj_fc.run(input_h.data(), out_h.data());
    residual_layernorm_inplace(
        out_h.data(),
        residual.data(),
        gamma.data(),
        beta.data(),
        M,
        H,
        EPS);
    g_sink += out_h[0];
  });

  const Stats ffn1_core = benchmark([&]() {
    ffn1_fc.run(input_h.data(), out_ffn1.data());
    g_sink += out_ffn1[0];
  });

  const Stats gelu_post = benchmark([&]() {
    approx_gelu_inplace(out_ffn1.data(), M, FFN);
    g_sink += out_ffn1[0];
  });

  const Stats ffn1_full = benchmark([&]() {
    ffn1_fc.run(input_h.data(), out_ffn1.data());
    approx_gelu_inplace(out_ffn1.data(), M, FFN);
    g_sink += out_ffn1[0];
  });

  const Stats ffn2_core = benchmark([&]() {
    ffn2_fc.run(input_ffn.data(), out_h.data());
    g_sink += out_h[0];
  });

  const Stats ffn2_full = benchmark([&]() {
    ffn2_fc.run(input_ffn.data(), out_h.data());
    residual_layernorm_inplace(
        out_h.data(),
        residual.data(),
        gamma.data(),
        beta.data(),
        M,
        H,
        EPS);
    g_sink += out_h[0];
  });

  std::cout << "===== XNNPACK BERT-large one-thread breakdown =====" << std::endl;
  print_stats("qkv_core", qkv_core);
  print_stats("qkv_permute", qkv_post);
  print_stats("qkv_full", qkv_full);
  print_stats("projection_core", proj_core);
  print_stats("residual_layernorm", residual_ln_post);
  print_stats("projection_full", proj_full);
  print_stats("ffn1_core", ffn1_core);
  print_stats("fast_gelu", gelu_post);
  print_stats("ffn1_full", ffn1_full);
  print_stats("ffn2_core", ffn2_core);
  print_stats("ffn2_full", ffn2_full);

  std::cout
      << "RESULT_JSON: {"
      << "\"qkv\":{"
      << "\"core_ms\":" << qkv_core.median_ms << ","
      << "\"post_ms\":" << qkv_post.median_ms << ","
      << "\"full_ms\":" << qkv_full.median_ms << "},"
      << "\"projection\":{"
      << "\"core_ms\":" << proj_core.median_ms << ","
      << "\"post_ms\":" << residual_ln_post.median_ms << ","
      << "\"full_ms\":" << proj_full.median_ms << "},"
      << "\"ffn1\":{"
      << "\"core_ms\":" << ffn1_core.median_ms << ","
      << "\"post_ms\":" << gelu_post.median_ms << ","
      << "\"full_ms\":" << ffn1_full.median_ms << "},"
      << "\"ffn2\":{"
      << "\"core_ms\":" << ffn2_core.median_ms << ","
      << "\"post_ms\":" << residual_ln_post.median_ms << ","
      << "\"full_ms\":" << ffn2_full.median_ms << "}"
      << "}" << std::endl;

  if (g_sink == 123456.0f) {
    std::cerr << "sink" << std::endl;
  }

  return 0;
}
