# AITemplate x86 CPU Backend

This branch extends AITemplate with an experimental **x86 CPU backend** for efficient BERT-family inference.

The implementation preserves AITemplate's graph compilation and code-generation model while generating CPU C++ code and using **XNNPACK** for optimized FP32 computation.

## Project Scope

The current backend targets complete inference for:

- BERT-base
- BERT-large
- Megatron-BERT 1.3B

The implementation is currently focused on FP32 BERT-family workloads rather than complete CPU operator coverage for all AITemplate models.

## Architecture

The primary CPU backend implementation is located under:

    python/aitemplate/backend/cpu/
    ├── embedding/
    │   └── bert_embeddings.py
    ├── gemm_universal/
    │   ├── attention_memory_qscale_overlay.py
    │   ├── bmm_softmax_bmm_permute.py
    │   ├── gemm_rcr_bias.py
    │   ├── gemm_rcr_bias_add.py
    │   ├── gemm_rcr_bias_add_layernorm_overlay.py
    │   ├── gemm_rcr_bias_fast_gelu.py
    │   ├── gemm_rcr_bias_permute.py
    │   └── static_fc.py
    ├── layernorm/
    │   └── layernorm.py
    ├── softmax/
    │   └── softmax.py
    ├── tensor/
    ├── lib_template.py
    └── target_def.py

CPU thread-pool support is implemented in:

    static/include/cpu_threadpool.h

The high-level compilation flow is:

    AITemplate graph
            ↓
    CPU backend code generation
            ↓
    Generated C++ operators
            ↓
    XNNPACK / CPU kernels
            ↓
    Compiled shared library

## Implemented BERT Operators

The CPU backend currently covers the main operations required by the tested BERT-family models:

- BERT embeddings
- GEMM + bias
- GEMM + bias + residual add
- GEMM + bias + ApproxGELU
- GEMM + bias + permutation
- LayerNorm
- Softmax
- QKV permutation
- Attention Q scaling
- QK^T → Softmax → AV attention path
- Tensor identity
- Tensor split

## Static Weight Prepacking

Constant fully connected weights are packed through XNNPACK and reused across inference calls.

The implementation uses a stable cache identity instead of depending directly on the raw weight pointer.

Once a static weight has been successfully packed, the original constant storage can be released since it is no longer needed.

This can significantly reduce memory usage.

## Memory Optimizations

Several BERT execution paths were modified to reduce intermediate allocations and memory traffic:

- residual buffer aliasing
- in-place ApproxGELU
- residual + LayerNorm buffer overlay
- removal of unnecessary input scratch buffers
- Q scaling fused into QKV permutation/copy
- release of raw constants after static weight packing

These optimizations become especially important for BERT-large and Megatron-BERT 1.3B.

## CPU Multithreading

The backend uses a persistent CPU thread pool.

The thread count can be configured with:

    AIT_CPU_NUM_THREADS=<N>

XNNPACK operators use the shared backend thread pool, while hand-written CPU loops use the same parallel execution infrastructure.

If `AIT_CPU_NUM_THREADS` is not specified, execution defaults to one thread.

## Tested Models

All final benchmark configurations use batch size 1, sequence length 128, and FP32.

| Model | Layers | Hidden | Heads | Intermediate |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 3072 |
| BERT-large | 24 | 1024 | 16 | 4096 |
| Megatron-BERT 1.3B | 24 | 2048 | 32 | 8192 |

## Benchmark Methodology

Final public benchmarks use:

- Intel Core i9-12900H (all tests run on assigned P-cores)
- batch size 1
- sequence length 128
- FP32
- 1 / 2 / 4 CPU threads
- repeated isolated benchmark rounds
- median-of-medians latency

Benchmark parameters use deterministic synthetic FP32 weights rather than downloaded pretrained checkpoints.

AITemplate and PyTorch reconstruct the same deterministic tensors for numerical and performance comparison.

## BERT-base Performance

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 207.29 ms | 230.94 ms | 1.114x |
| 2 | 112.58 ms | 124.50 ms | 1.106x |
| 4 | 69.07 ms | 75.24 ms | 1.089x |

AITemplate provides approximately **8–10% lower latency** than the PyTorch CPU reference.

Measured resident memory is approximately equivalent for BERT-base.

## BERT-large Performance

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 730.85 ms | 764.61 ms | 1.046x |
| 2 | 405.92 ms | 424.53 ms | 1.046x |
| 4 | 255.09 ms | 270.90 ms | 1.062x |

AITemplate provides approximately **4–6% lower latency**.

Measured resident memory is approximately **20% lower** than the PyTorch reference.

## Megatron-BERT 1.3B Performance

| Threads | AITemplate | PyTorch | AIT / PyTorch |
|---:|---:|---:|---:|
| 1 | 2835.89 ms | 2889.43 ms | 1.019x |
| 2 | 1589.29 ms | 1597.28 ms | 1.005x |
| 4 | 974.21 ms | 992.63 ms | 1.018x |

For Megatron-BERT 1.3B, AITemplate and PyTorch provide approximately equivalent latency within ±2%.

AITemplate uses approximately **6–7% less resident memory**.

As hidden size increases, runtime becomes increasingly dominated by large GEMM operations. Since both XNNPACK and the PyTorch CPU backend provide highly optimized GEMM implementations, the relative advantage from AITemplate-side memory and fusion optimizations becomes smaller.

## Thread Scaling

AITemplate scaling relative to one thread:

| Model | 2 Threads | 4 Threads |
|---|---:|---:|
| BERT-base | 1.84x | 3.00x |
| BERT-large | 1.80x | 2.87x |
| Megatron-BERT 1.3B | 1.78x | 2.91x |

## Benchmark Results

Processed benchmark results are stored under:

    benchmark_results/
    ├── all_models.json
    ├── bert_base.json
    ├── bert_large.json
    └── megatron_bert_1_3b.json

Benchmark implementations are located under:

    tests/unittest/ops/

Raw benchmark logs are excluded from version control.

## Current Limitations

The backend remains an experimental CPU extension.

Current limitations include:

- FP32-focused implementation
- BERT-family operator coverage rather than complete CPU backend coverage
- some upstream frontend paths remain GPU-oriented
- `run_with_tensors()` still assumes GPU tensors in some paths
- CPU tests use lower-level runtime interfaces where necessary
- some frontend modules still require CPU-specific configuration to avoid CUDA target detection

The implementation is therefore intended as a focused x86 CPU backend for BERT-family inference rather than a replacement for AITemplate's CUDA and ROCm backends.

## Upstream Project

This repository is based on Meta's AITemplate project:

https://github.com/facebookincubator/AITemplate

The original upstream README is preserved in [README_UPSTREAM.md](README_UPSTREAM.md).
