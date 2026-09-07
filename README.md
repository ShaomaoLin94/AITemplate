# Towards Efficient BERT Inference on x86 CPUs with AITemplate

An experimental **x86 CPU backend for AITemplate**, investigating how a GPU-centric deep learning compiler can be extended to provide efficient BERT-family inference on general-purpose CPUs.

The backend preserves AITemplate's ahead-of-time graph compilation and code-generation model while introducing CPU execution with [XNNPACK](https://github.com/google/XNNPACK), static weight prepacking, memory-aware execution, operator-level optimizations, and multithreading.

## Motivation

AITemplate was originally designed around GPU execution, with its backend and runtime primarily targeting CUDA and ROCm.

Simply replacing GPU kernels with CPU implementations is not sufficient to obtain competitive CPU inference performance. On CPUs, performance is also affected by repeated weight preparation, temporary buffers, intermediate memory traffic, operator boundaries, and thread-management overhead.

This project therefore explores two questions:

- Can AITemplate's compilation model be effectively extended from GPUs to x86 CPUs for Transformer inference?
- Which compiler- and runtime-level bottlenecks limit BERT inference on CPUs, and how much can static graph information reduce them?

BERT-family models are used as the main workload because they combine large GEMMs with attention, normalization, activation, residual connections, and significant intermediate memory traffic.

## Main Bottlenecks

Several bottlenecks emerged during the CPU backend development:

- **Static weight preparation**  
  Fully connected weights remain constant across inference calls, yet repeatedly preparing or retaining multiple weight representations adds unnecessary overhead and memory usage.

- **Intermediate memory traffic**  
  Temporary buffers and materialized intermediate tensors between GEMM, activation, residual, LayerNorm, and attention operations can become costly on CPUs.

- **Operator boundary overhead**  
  Lightweight operations such as Q scaling, activation, and layout transformation may still require additional memory passes when executed separately.

- **CPU parallel execution**  
  Efficient multithreading requires persistent worker management rather than repeatedly creating parallel execution resources.

- **GEMM dominance at larger model sizes**  
  As hidden dimensions increase, runtime becomes increasingly dominated by large GEMMs, reducing the relative impact of compiler-side memory and fusion optimizations.

## Key Optimizations

The CPU backend introduces:

- XNNPACK-backed FP32 operators
- Static fully connected weight prepacking
- Raw constant release after prepacking
- Stable static-weight cache identity
- Scratch buffer reuse and elimination
- In-place ApproxGELU
- Residual + LayerNorm memory overlay
- Attention Q scaling fused into QKV permutation
- Persistent CPU thread pool
- Configurable multithreading with `AIT_CPU_NUM_THREADS`

The goal is not only to optimize individual kernels, but also to reduce overhead **between** kernels.

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

Compilation flow:

    AITemplate graph
            ↓
    CPU backend code generation
            ↓
    Generated C++ operators
            ↓
    XNNPACK / CPU kernels
            ↓
    Compiled shared library

## Supported BERT-family Models

| Model | Layers | Hidden | Heads | Intermediate |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 3072 |
| BERT-large | 24 | 1024 | 16 | 4096 |
| Megatron-BERT 1.3B | 24 | 2048 | 32 | 8192 |

The backend currently supports the main operations required by these models, including embeddings, GEMM, attention, Softmax, LayerNorm, ApproxGELU, residual operations, permutation, identity, and split.

## Performance

Benchmarks were measured on an **Intel Core i9-12900H**, using assigned P-cores with:

- Batch size: 1
- Sequence length: 128
- Precision: FP32
- CPU threads: 1 / 2 / 4

### BERT-base

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 207.29 ms | 230.94 ms | 1.114x |
| 2 | 112.58 ms | 124.50 ms | 1.106x |
| 4 | 69.07 ms | 75.24 ms | 1.089x |

AITemplate provides approximately **8–10% lower latency** than the PyTorch CPU reference.

### BERT-large

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 730.85 ms | 764.61 ms | 1.046x |
| 2 | 405.92 ms | 424.53 ms | 1.046x |
| 4 | 255.09 ms | 270.90 ms | 1.062x |

AITemplate provides approximately **4–6% lower latency** and uses approximately **20% less resident memory**.

### Megatron-BERT 1.3B

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 2835.89 ms | 2889.43 ms | 1.019x |
| 2 | 1589.29 ms | 1618.48 ms | 1.018x |
| 4 | 974.21 ms | 992.63 ms | 1.019x |

For Megatron-BERT 1.3B, AITemplate and PyTorch provide approximately equivalent latency within **±2%**, while AITemplate uses approximately **6–7% less resident memory**.

The smaller relative speedup at this scale reflects the increasing dominance of large GEMM operations.

## Benchmark Data

Processed benchmark results are available under:

    benchmark_results/
    ├── all_models.json
    ├── bert_base.json
    ├── bert_large.json
    └── megatron_bert_1_3b.json

Benchmark implementations are located under:

    tests/unittest/ops/

For architecture, implementation details, bottleneck analysis, and limitations, see [CPU_BACKEND.md](CPU_BACKEND.md).

## Scope

This project focuses on **FP32 BERT-family inference on x86 CPUs** rather than complete CPU operator coverage for AITemplate.

Some original AITemplate frontend and runtime paths remain GPU-oriented, so CPU-specific execution paths are currently required where upstream interfaces assume CUDA execution.

## Upstream AITemplate

This repository is based on Meta's original [AITemplate](https://github.com/facebookincubator/AITemplate).

The original upstream README is preserved in [README_UPSTREAM.md](README_UPSTREAM.md).
