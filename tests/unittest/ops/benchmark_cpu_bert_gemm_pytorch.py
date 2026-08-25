import os
import statistics
import time

import torch
import torch.nn.functional as F


CPU = 0
WARMUP = 50
ITERS = 200

BATCH = 1
SEQ = 128
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
INTERMEDIATE = 3072


os.sched_setaffinity(0, {CPU})

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(0)


def bench(name, fn):
    with torch.inference_mode():
        for _ in range(WARMUP):
            fn()

        times = []

        for _ in range(ITERS):
            start = time.perf_counter_ns()
            fn()
            end = time.perf_counter_ns()

            times.append(
                (end - start) / 1_000_000.0
            )

    print(
        f"{name:<26} "
        f"median={statistics.median(times):8.4f} ms  "
        f"mean={statistics.mean(times):8.4f} ms  "
        f"min={min(times):8.4f} ms"
    )


# --------------------------------------------------
# FFN1
# [128, 768] x [3072, 768]^T
# + bias + approximate GELU
# --------------------------------------------------

ffn1_x = torch.randn(
    SEQ,
    HIDDEN,
    dtype=torch.float32,
)

ffn1_w = torch.randn(
    INTERMEDIATE,
    HIDDEN,
    dtype=torch.float32,
)

ffn1_b = torch.randn(
    INTERMEDIATE,
    dtype=torch.float32,
)


def ffn1():
    y = F.linear(
        ffn1_x,
        ffn1_w,
        ffn1_b,
    )

    return F.gelu(
        y,
        approximate="tanh",
    )


# --------------------------------------------------
# FFN2
# [128, 3072] x [768, 3072]^T
# + bias + residual
# --------------------------------------------------

ffn2_x = torch.randn(
    SEQ,
    INTERMEDIATE,
    dtype=torch.float32,
)

ffn2_w = torch.randn(
    HIDDEN,
    INTERMEDIATE,
    dtype=torch.float32,
)

ffn2_b = torch.randn(
    HIDDEN,
    dtype=torch.float32,
)

ffn2_residual = torch.randn(
    SEQ,
    HIDDEN,
    dtype=torch.float32,
)


def ffn2():
    return F.linear(
        ffn2_x,
        ffn2_w,
        ffn2_b,
    ) + ffn2_residual


# --------------------------------------------------
# QKV
# [128, 768] x [2304, 768]^T
# + bias
# + physical layout permutation
#
# [B, S, 3, H, D]
# ->
# [3, B, H, S, D]
# --------------------------------------------------

qkv_x = torch.randn(
    BATCH * SEQ,
    HIDDEN,
    dtype=torch.float32,
)

qkv_w = torch.randn(
    3 * HIDDEN,
    HIDDEN,
    dtype=torch.float32,
)

qkv_b = torch.randn(
    3 * HIDDEN,
    dtype=torch.float32,
)


def qkv():
    y = F.linear(
        qkv_x,
        qkv_w,
        qkv_b,
    )

    y = y.view(
        BATCH,
        SEQ,
        3,
        HEADS,
        HEAD_DIM,
    )

    return y.permute(
        2,
        0,
        3,
        1,
        4,
    ).contiguous()


# --------------------------------------------------
# Attention projection
# [128, 768] x [768, 768]^T
# + bias + residual
# --------------------------------------------------

proj_x = torch.randn(
    SEQ,
    HIDDEN,
    dtype=torch.float32,
)

proj_w = torch.randn(
    HIDDEN,
    HIDDEN,
    dtype=torch.float32,
)

proj_b = torch.randn(
    HIDDEN,
    dtype=torch.float32,
)

proj_residual = torch.randn(
    SEQ,
    HIDDEN,
    dtype=torch.float32,
)


def projection():
    return F.linear(
        proj_x,
        proj_w,
        proj_b,
    ) + proj_residual


print("===== PyTorch BERT GEMM benchmark =====")
print("pinned CPU :", CPU)
print("threads    :", torch.get_num_threads())
print("mkldnn     :", torch.backends.mkldnn.is_available())
print()

bench("FFN1 + GELU", ffn1)
bench("FFN2 + residual", ffn2)
bench("QKV + permute", qkv)
bench("Projection + residual", projection)

print()
print("===== Current AIT reference =====")
print("FFN1 + GELU              ~4.99 ms")
print("FFN2 + residual          ~4.80 ms")
print("QKV + permute            ~4.01 ms")
print("Projection + residual    ~1.17 ms")
