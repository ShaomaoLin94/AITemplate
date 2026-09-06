# AITemplate x86 CPU Backend

An experimental **x86 CPU backend for AITemplate**, targeting efficient BERT-family inference with XNNPACK.

This project extends Meta's original [AITemplate](https://github.com/facebookincubator/AITemplate) compiler with CPU code generation, XNNPACK-backed operators, static weight prepacking, memory optimizations, and multithreaded execution.

## Highlights

- New x86 CPU backend for AITemplate
- XNNPACK-backed FP32 computation
- Full BERT embeddings and attention execution
- Static fully connected weight prepacking
- Raw weight release after prepacking
- Reduced intermediate memory usage
- Persistent CPU thread pool
- Configurable multithreading with `AIT_CPU_NUM_THREADS`
- Full inference support for BERT-base, BERT-large, and Megatron-BERT 1.3B

## Performance

Benchmarks were measured on an Intel Core i9-12900H with batch size 1, sequence length 128, FP32, using 1 / 2 / 4 CPU threads.

### BERT-base

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 207.29 ms | 230.94 ms | 1.114x |
| 2 | 112.58 ms | 124.50 ms | 1.106x |
| 4 | 69.07 ms | 75.24 ms | 1.089x |

AITemplate reduces latency by approximately **8–10%**.

### BERT-large

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 730.85 ms | 764.61 ms | 1.046x |
| 2 | 405.92 ms | 424.53 ms | 1.046x |
| 4 | 255.09 ms | 270.90 ms | 1.062x |

AITemplate reduces latency by approximately **4–6%** and uses approximately **20% less resident memory**.

### Megatron-BERT 1.3B

| Threads | AITemplate | PyTorch | AIT / PyTorch |
|---:|---:|---:|---:|
| 1 | 2835.89 ms | 2889.43 ms | 1.019x |
| 2 | 1589.29 ms | 1597.28 ms | 1.005x |
| 4 | 974.21 ms | 962.71 ms | 0.988x |

AITemplate and PyTorch provide approximately equivalent latency within ±2%, while AITemplate uses approximately **6–7% less resident memory**.

## CPU Backend

Main implementation:

    python/aitemplate/backend/cpu/
    ├── embedding/
    ├── gemm_universal/
    ├── layernorm/
    ├── softmax/
    ├── tensor/
    ├── lib_template.py
    └── target_def.py

CPU multithreading support:

    static/include/cpu_threadpool.h

Execution flow:

    AITemplate graph
        ↓
    CPU backend code generation
        ↓
    Generated C++ model
        ↓
    XNNPACK / CPU kernels
        ↓
    Compiled shared library

## Tested Models

| Model | Layers | Hidden | Heads | FFN |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 3072 |
| BERT-large | 24 | 1024 | 16 | 4096 |
| Megatron-BERT 1.3B | 24 | 2048 | 32 | 8192 |

## Key Optimizations

- XNNPACK static fully connected weight prepacking
- Release of raw constant weights after packing
- Stable static-weight cache identity
- Residual buffer aliasing
- In-place ApproxGELU
- Residual + LayerNorm memory overlay
- Reduced unnecessary scratch buffers
- Attention Q scaling fused into QKV permutation
- Persistent shared CPU thread pool

## Benchmark Data

Processed benchmark results are available under:

    benchmark_results/
    ├── all_models.json
    ├── bert_base.json
    ├── bert_large.json
    └── megatron_bert_1_3b.json

For implementation details and limitations, see [CPU_BACKEND.md](CPU_BACKEND.md).

## Scope

This project focuses on FP32 BERT-family CPU inference and does not yet provide complete AITemplate CPU operator coverage.

Some original AITemplate frontend and runtime paths remain GPU-oriented, so CPU-specific execution paths are currently used where upstream interfaces assume CUDA execution.

## Upstream AITemplate

This repository is a fork of Meta's [AITemplate](https://github.com/facebookincubator/AITemplate).

The original AITemplate README is preserved in [README_UPSTREAM.md](README_UPSTREAM.md).
