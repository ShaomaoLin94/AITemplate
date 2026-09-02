import os
import statistics
import time

import torch
import torch.nn.functional as F


allowed_cpus = sorted(os.sched_getaffinity(0))
cpu = allowed_cpus[0]
os.sched_setaffinity(0, {cpu})

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

WARMUP = 20
ITERS = 100


SHAPES = [
    ("QKV",        128, 2304, 768),
    ("Projection", 128, 768,  768),
    ("FFN1",       128, 3072, 768),
    ("FFN2",       128, 768,  3072),
]


def benchmark(name, m, n, k):
    torch.manual_seed(0)

    x = torch.randn(
        m,
        k,
        dtype=torch.float32,
    )

    weight = torch.randn(
        n,
        k,
        dtype=torch.float32,
    )

    bias = torch.randn(
        n,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            y = F.linear(
                x,
                weight,
                bias,
            )

        times = []

        for _ in range(ITERS):
            begin = time.perf_counter_ns()

            y = F.linear(
                x,
                weight,
                bias,
            )

            end = time.perf_counter_ns()

            times.append(
                (end - begin) / 1.0e6
            )

    median_ms = statistics.median(times)
    mean_ms = statistics.mean(times)

    flops = 2.0 * m * n * k
    gflops = (
        flops /
        (median_ms / 1000.0) /
        1.0e9
    )

    print()
    print(
        f"===== {name:<10} "
        f"M={m} N={n} K={k} ====="
    )

    print(
        f"median ms: {median_ms:.4f}"
    )
    print(
        f"mean ms  : {mean_ms:.4f}"
    )
    print(
        f"min ms   : {min(times):.4f}"
    )
    print(
        f"max ms   : {max(times):.4f}"
    )
    print(
        f"GFLOP/s  : {gflops:.2f}"
    )

    print(
        "checksum :",
        float(y.flatten()[y.numel() // 2]),
    )


def main():
    print(
        "===== PyTorch BERT GEMM shape benchmark ====="
    )

    print("pinned CPU :", cpu)
    print(
        "threads    :",
        torch.get_num_threads(),
    )

    for args in SHAPES:
        benchmark(*args)


if __name__ == "__main__":
    main()
