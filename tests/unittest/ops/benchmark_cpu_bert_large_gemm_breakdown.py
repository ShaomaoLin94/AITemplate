import json
import os
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F


SOURCE = "tests/unittest/ops/cpu_bert_large_xnn_breakdown.cpp"
BINARY = "./tmp/cpu_bert_large_xnn_breakdown"

M = 128
HIDDEN = 1024
INTERMEDIATE = 4096
QKV = 3072
HEADS = 16
HEAD_DIM = 64
EPS = 1e-12


def pin_single_cpu():
    cpus = sorted(os.sched_getaffinity(0))
    if not cpus:
        raise RuntimeError("No CPU available in affinity mask")
    cpu = cpus[0]
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return cpu


def benchmark(fn, warmup=5, iterations=30):
    with torch.no_grad():
        for _ in range(warmup):
            fn()

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def compile_xnn_helper():
    os.makedirs(os.path.dirname(BINARY), exist_ok=True)

    cmd = [
        os.environ.get("CXX", "g++"),
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-pthread",
        SOURCE,
        "-o",
        BINARY,
        "-L/usr/local/lib",
        "-Wl,-rpath,/usr/local/lib",
        "-lXNNPACK",
    ]

    print("===== Build XNNPACK breakdown helper =====")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_xnn_helper():
    proc = subprocess.run(
        [BINARY],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="")

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line.split(":", 1)[1].strip())

    raise RuntimeError("No RESULT_JSON from XNNPACK helper")


def make_data(m, k, n, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(m, k, generator=gen, dtype=torch.float32) * 0.02
    weight = torch.randn(n, k, generator=gen, dtype=torch.float32) * 0.02
    bias = torch.randn(n, generator=gen, dtype=torch.float32) * 0.02
    return x, weight, bias


def torch_breakdown():
    result = {}

    gamma = torch.ones(HIDDEN, dtype=torch.float32)
    beta = torch.zeros(HIDDEN, dtype=torch.float32)
    residual = torch.randn(M, HIDDEN, dtype=torch.float32) * 0.02

    x, weight, bias = make_data(M, HIDDEN, QKV, 10)
    qkv_base = F.linear(x, weight, bias)

    result["qkv"] = {
        "core": benchmark(lambda: F.linear(x, weight, bias)),
        "post": benchmark(
            lambda: qkv_base.view(1, 128, 3, HEADS, HEAD_DIM)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        ),
        "full": benchmark(
            lambda: F.linear(x, weight, bias)
            .view(1, 128, 3, HEADS, HEAD_DIM)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        ),
    }
    del x, weight, bias, qkv_base

    x, weight, bias = make_data(M, HIDDEN, HIDDEN, 20)
    proj_base = F.linear(x, weight, bias)

    result["projection"] = {
        "core": benchmark(lambda: F.linear(x, weight, bias)),
        "post": benchmark(
            lambda: F.layer_norm(
                proj_base + residual,
                (HIDDEN,),
                gamma,
                beta,
                EPS,
            )
        ),
        "full": benchmark(
            lambda: F.layer_norm(
                F.linear(x, weight, bias) + residual,
                (HIDDEN,),
                gamma,
                beta,
                EPS,
            )
        ),
    }
    del x, weight, bias, proj_base

    x, weight, bias = make_data(M, HIDDEN, INTERMEDIATE, 30)
    ffn1_base = F.linear(x, weight, bias)

    result["ffn1"] = {
        "core": benchmark(lambda: F.linear(x, weight, bias)),
        "post": benchmark(
            lambda: F.gelu(ffn1_base, approximate="tanh")
        ),
        "full": benchmark(
            lambda: F.gelu(
                F.linear(x, weight, bias),
                approximate="tanh",
            )
        ),
    }
    del x, weight, bias, ffn1_base

    x, weight, bias = make_data(M, INTERMEDIATE, HIDDEN, 40)
    ffn2_base = F.linear(x, weight, bias)

    result["ffn2"] = {
        "core": benchmark(lambda: F.linear(x, weight, bias)),
        "post": benchmark(
            lambda: F.layer_norm(
                ffn2_base + residual,
                (HIDDEN,),
                gamma,
                beta,
                EPS,
            )
        ),
        "full": benchmark(
            lambda: F.layer_norm(
                F.linear(x, weight, bias) + residual,
                (HIDDEN,),
                gamma,
                beta,
                EPS,
            )
        ),
    }

    return result


def print_table(xnn, pt):
    print()
    print("===== Exact-shape median breakdown =====")
    print(
        f"{'path':<12} {'XNN core':>10} {'XNN post':>10} {'XNN full':>10} "
        f"{'PT core':>10} {'PT post':>10} {'PT full':>10} {'XNN/PT':>9}"
    )

    for name in ("qkv", "projection", "ffn1", "ffn2"):
        x = xnn[name]
        p = pt[name]
        pt_core = p["core"]["median_ms"]
        pt_post = p["post"]["median_ms"]
        pt_full = p["full"]["median_ms"]
        ratio = pt_full / x["full_ms"]

        print(
            f"{name:<12} "
            f"{x['core_ms']:10.4f} {x['post_ms']:10.4f} {x['full_ms']:10.4f} "
            f"{pt_core:10.4f} {pt_post:10.4f} {pt_full:10.4f} {ratio:9.4f}"
        )

    xnn_layer = sum(xnn[name]["full_ms"] for name in xnn)
    pt_layer = sum(pt[name]["full"]["median_ms"] for name in pt)

    print()
    print("XNN four-path sum / layer ms:", xnn_layer)
    print("PT  four-path sum / layer ms:", pt_layer)
    print("Estimated XNN 24-layer ms   :", xnn_layer * 24.0)
    print("Estimated PT  24-layer ms   :", pt_layer * 24.0)

    print()
    print("===== Post-op shares =====")
    for name in ("qkv", "projection", "ffn1", "ffn2"):
        x = xnn[name]
        share = x["post_ms"] / x["full_ms"] * 100.0
        print(f"{name:<12}: {share:6.2f}% of XNN full microbenchmark")


def main():
    cpu = pin_single_cpu()
    print("===== BERT-large GEMM/post-op breakdown =====")
    print("pinned CPU   :", cpu)
    print("threads      :", torch.get_num_threads())
    print("M            :", M)
    print("hidden       :", HIDDEN)
    print("intermediate :", INTERMEDIATE)
    print("QKV N        :", QKV)

    compile_xnn_helper()
    xnn = run_xnn_helper()

    print()
    print("===== PyTorch exact-shape breakdown =====")
    pt = torch_breakdown()
    for name in ("qkv", "projection", "ffn1", "ffn2"):
        print(
            f"{name:<12} "
            f"core={pt[name]['core']['median_ms']:.4f} ms  "
            f"post={pt[name]['post']['median_ms']:.4f} ms  "
            f"full={pt[name]['full']['median_ms']:.4f} ms"
        )

    print_table(xnn, pt)

    print(
        "RESULT_JSON:",
        json.dumps(
            {
                "xnnpack": xnn,
                "pytorch": pt,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
