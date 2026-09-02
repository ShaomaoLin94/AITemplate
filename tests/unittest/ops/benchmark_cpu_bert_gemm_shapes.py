import os
import subprocess
import textwrap
from pathlib import Path


# Match the current isolated/full benchmark environment:
# pin the whole process, and therefore the child benchmark, to one CPU.
allowed_cpus = sorted(os.sched_getaffinity(0))
cpu = allowed_cpus[0]
os.sched_setaffinity(0, {cpu})

CPP_PATH = Path("/tmp/ait_cpu_bert_gemm_shapes.cpp")
BIN_PATH = Path("/tmp/ait_cpu_bert_gemm_shapes")


CPP_SOURCE = r'''
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <xnnpack.h>


static void check_xnn(
    xnn_status status,
    const char* step) {

  if (status != xnn_status_success) {
    throw std::runtime_error(
        std::string(step) +
        " failed, status=" +
        std::to_string(
            static_cast<int>(status)));
  }
}


static void fill_data(
    std::vector<float>& data,
    float scale) {

  for (size_t i = 0; i < data.size(); ++i) {
    const int value =
        static_cast<int>(i % 251) - 125;

    data[i] =
        static_cast<float>(value) *
        scale;
  }
}


struct Stats {
  double median_ms;
  double mean_ms;
  double min_ms;
  double max_ms;
};


static Stats summarize(
    std::vector<double> times) {

  std::sort(
      times.begin(),
      times.end());

  const size_t n = times.size();

  double median;

  if (n % 2 == 0) {
    median =
        0.5 *
        (
            times[n / 2 - 1] +
            times[n / 2]
        );
  } else {
    median = times[n / 2];
  }

  const double mean =
      std::accumulate(
          times.begin(),
          times.end(),
          0.0) /
      static_cast<double>(n);

  return {
      median,
      mean,
      times.front(),
      times.back(),
  };
}


static void print_stats(
    const char* label,
    const Stats& stats,
    size_t m,
    size_t n,
    size_t k) {

  const double flops =
      2.0 *
      static_cast<double>(m) *
      static_cast<double>(n) *
      static_cast<double>(k);

  const double seconds =
      stats.median_ms / 1000.0;

  const double gflops =
      flops /
      seconds /
      1.0e9;

  std::printf(
      "  %-12s median=%8.4f ms  "
      "mean=%8.4f  min=%8.4f  max=%8.4f  "
      "%8.2f GFLOP/s\n",
      label,
      stats.median_ms,
      stats.mean_ms,
      stats.min_ms,
      stats.max_ms,
      gflops);
}


static void benchmark_shape(
    const char* name,
    size_t m,
    size_t n,
    size_t k) {

  constexpr size_t warmup = 20;
  constexpr size_t iterations = 100;

  const size_t input_elements =
      m * k;

  const size_t output_elements =
      m * n;

  const size_t extra_elements =
      (XNN_EXTRA_BYTES +
       sizeof(float) - 1) /
      sizeof(float);

  // Persistent padded input:
  // this represents the current BERT internal-blob fast path,
  // not the old input_scratch memcpy path.
  std::vector<float> input(
      input_elements +
      extra_elements,
      0.0f);

  std::vector<float> weight(
      n * k);

  std::vector<float> bias(
      n);

  std::vector<float> output(
      output_elements +
      extra_elements,
      0.0f);

  fill_data(input, 0.001f);
  fill_data(weight, 0.0001f);
  fill_data(bias, 0.001f);

  xnn_operator_t op = nullptr;

  const auto create_begin =
      std::chrono::steady_clock::now();

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

  const auto create_end =
      std::chrono::steady_clock::now();

  const double create_ms =
      std::chrono::duration<double, std::milli>(
          create_end -
          create_begin)
          .count();

  check_xnn(
      xnn_setup_fully_connected_nc_f32(
          op,
          input.data(),
          output.data()),
      "xnn_setup_fully_connected_nc_f32");

  for (size_t i = 0; i < warmup; ++i) {
    check_xnn(
        xnn_run_operator(
            op,
            nullptr),
        "xnn_run_operator(warmup)");
  }

  // ----------------------------------------------------------
  // A. Pure xnn_run_operator()
  //
  // setup is done once outside the timed region.
  // ----------------------------------------------------------

  std::vector<double> run_only_times;
  run_only_times.reserve(iterations);

  for (size_t i = 0; i < iterations; ++i) {
    const auto begin =
        std::chrono::steady_clock::now();

    check_xnn(
        xnn_run_operator(
            op,
            nullptr),
        "xnn_run_operator");

    const auto end =
        std::chrono::steady_clock::now();

    run_only_times.push_back(
        std::chrono::duration<double, std::milli>(
            end - begin)
            .count());
  }

  // ----------------------------------------------------------
  // B. setup + run
  //
  // This is closer to the current generated BERT backend,
  // which calls xnn_setup_* on every inference.
  // ----------------------------------------------------------

  std::vector<double> setup_run_times;
  setup_run_times.reserve(iterations);

  for (size_t i = 0; i < iterations; ++i) {
    const auto begin =
        std::chrono::steady_clock::now();

    check_xnn(
        xnn_setup_fully_connected_nc_f32(
            op,
            input.data(),
            output.data()),
        "xnn_setup_fully_connected_nc_f32");

    check_xnn(
        xnn_run_operator(
            op,
            nullptr),
        "xnn_run_operator");

    const auto end =
        std::chrono::steady_clock::now();

    setup_run_times.push_back(
        std::chrono::duration<double, std::milli>(
            end - begin)
            .count());
  }

  const Stats run_only =
      summarize(run_only_times);

  const Stats setup_run =
      summarize(setup_run_times);

  // Prevent an optimizer from treating the result as dead.
  volatile float checksum =
      output[
          (m * n) / 2];

  std::printf("\n");
  std::printf(
      "===== %-10s M=%zu N=%zu K=%zu =====\n",
      name,
      m,
      n,
      k);

  std::printf(
      "  create+reshape once: %.4f ms\n",
      create_ms);

  print_stats(
      "run only",
      run_only,
      m,
      n,
      k);

  print_stats(
      "setup+run",
      setup_run,
      m,
      n,
      k);

  std::printf(
      "  checksum     : %.9f\n",
      static_cast<double>(checksum));

  xnn_delete_operator(op);
}


int main() {
  check_xnn(
      xnn_initialize(nullptr),
      "xnn_initialize");

  std::printf(
      "===== Pure XNNPACK BERT GEMM shape benchmark =====\n");

  std::printf(
      "threadpool   : nullptr (single-thread)\n");

  std::printf(
      "warmup       : 20\n");

  std::printf(
      "iterations   : 100\n");

  // BERT batch=1, sequence=128.
  benchmark_shape(
      "QKV",
      128,
      2304,
      768);

  benchmark_shape(
      "Projection",
      128,
      768,
      768);

  benchmark_shape(
      "FFN1",
      128,
      3072,
      768);

  benchmark_shape(
      "FFN2",
      128,
      768,
      3072);

  return 0;
}
'''


def main():
    print("===== Build pure XNNPACK GEMM benchmark =====")
    print("pinned CPU :", cpu)

    CPP_PATH.write_text(
        textwrap.dedent(CPP_SOURCE)
    )

    compile_cmd = [
        "/usr/bin/g++",
        "-O3",
        "-std=c++17",
        "-march=native",
        "-mtune=native",
        "-DNDEBUG",
        "-pthread",
        str(CPP_PATH),
        "-I/usr/local/include",
        "-L/usr/local/lib",
        "-Wl,-rpath,/usr/local/lib",
        "-lXNNPACK",
        "-o",
        str(BIN_PATH),
    ]

    print(
        "compile    :",
        " ".join(compile_cmd),
    )

    subprocess.run(
        compile_cmd,
        check=True,
    )

    print()
    subprocess.run(
        [str(BIN_PATH)],
        check=True,
    )


if __name__ == "__main__":
    main()
