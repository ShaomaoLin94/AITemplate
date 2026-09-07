# AITemplate x86 CPU Backend

This document describes the design, implementation, bottlenecks, and optimization decisions of the experimental **x86 CPU backend for AITemplate**.

The project investigates whether AITemplate's GPU-oriented compilation model can be extended to efficient CPU Transformer inference while preserving its ahead-of-time graph compilation and generated C++ execution model.

The current implementation focuses on FP32 BERT-family workloads and uses **XNNPACK** as the primary optimized CPU kernel library.

## Research Motivation

AITemplate was originally designed primarily for GPU inference through CUDA and ROCm.

Its original design assumptions therefore differ from CPU execution in several important ways. GPU-oriented execution commonly prioritizes kernel selection, device workspace management, and accelerator-specific code generation, while CPU inference is highly sensitive to memory traffic, weight preparation, thread management, and repeated operator-level overhead.

A direct translation from GPU operators to CPU kernels can therefore produce a functionally correct backend without necessarily providing competitive end-to-end inference performance.

The project focuses on two main questions:

1. How can AITemplate's compilation and runtime model be extended to support x86 CPU execution?
2. After basic CPU execution is available, which bottlenecks dominate BERT inference and which of them can be reduced using compile-time or static-model information?

## Backend Architecture

The primary implementation is located under:

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

The backend therefore preserves AITemplate's original ahead-of-time compilation model instead of introducing a separate CPU runtime framework.

## Implemented BERT Operators

The current CPU backend supports the main operators required by the tested BERT-family models:

- BERT embeddings
- GEMM + bias
- GEMM + bias + residual add
- GEMM + bias + ApproxGELU
- GEMM + bias + permutation
- LayerNorm
- Softmax
- QKV permutation
- Attention Q scaling
- QKᵀ → Softmax → AV attention path
- Tensor identity
- Tensor split

XNNPACK provides optimized FP32 kernels where appropriate, while additional generated CPU code handles BERT-specific execution, fusion, memory management, and data movement.

## Bottleneck Analysis

### 1. Static Weight Preparation

BERT contains a large number of fully connected layers whose weights remain constant during inference.

A naive execution path can introduce two forms of unnecessary overhead:

- repeated preparation or packing of constant weights
- retaining both raw and packed representations in memory

These costs become increasingly important as model size grows.

### Optimization

Constant fully connected weights are prepacked through XNNPACK and reused across inference calls.

The implementation also uses a stable cache identity rather than relying directly on raw weight pointers.

After successful prepacking, the original raw constant storage can be released when it is no longer required.

This shifts work from repeated inference-time processing into model initialization and reduces memory footprint.

## 2. Intermediate Memory Traffic

Transformer execution contains many intermediate values between:

- GEMM
- activation
- residual addition
- LayerNorm
- QKV transformation
- attention

On CPUs, these operations are not only limited by arithmetic throughput. Additional temporary buffers and memory copies can increase cache pressure and memory bandwidth usage.

A computationally inexpensive operator can therefore still contribute measurable latency if it requires a complete additional pass over a large tensor.

### Optimization

Several execution paths were redesigned to reduce temporary memory usage:

- residual buffer aliasing
- in-place ApproxGELU
- residual + LayerNorm buffer overlay
- removal of unnecessary input scratch buffers
- release of raw constants after static weight packing

These changes reduce both intermediate memory footprint and data movement.

## 3. Operator Boundary Overhead

AITemplate graphs are composed of individual operators, but on a CPU, keeping every transformation as an independent memory pass can be inefficient.

For example, Q scaling is mathematically inexpensive but still requires reading and writing the Q tensor when implemented as a separate operation.

The same issue appears with activation, residual handling, and layout transformation.

### Optimization

Where compatible, lightweight operations are folded into existing data movement or compute paths.

One example is:

    Q scaling
        +
    QKV permutation / copy

Instead of materializing a separately scaled Q tensor, scaling is performed while Q is already being copied into the attention layout.

The objective is therefore not only kernel optimization, but also reducing the number of times intermediate tensors must be materialized.

## 4. Scratch Buffer Allocation

Some CPU operator implementations initially required temporary input or output buffers to adapt tensor layouts or satisfy kernel interfaces.

Repeated allocation or oversized scratch storage increases both runtime overhead and memory usage.

### Optimization

Scratch buffers are reused where possible, and unnecessary buffers are removed entirely when input layouts can be consumed directly.

Persistent or reusable storage is preferred over repeated allocation during inference.

## 5. Thread Management

Multithreading improves CPU throughput only when thread-management overhead remains small relative to operator execution time.

Creating independent worker resources for individual operators would introduce additional overhead and produce inconsistent execution behavior.

### Optimization

The backend uses a persistent CPU thread pool:

    static/include/cpu_threadpool.h

The number of threads can be configured through:

    AIT_CPU_NUM_THREADS=<N>

XNNPACK operators use the shared backend thread pool, while hand-written backend loops use the same parallel execution infrastructure.

If `AIT_CPU_NUM_THREADS` is not specified, execution defaults to one thread.

## 6. GEMM-Dominated Execution

Profiling and model scaling reveal an important limitation.

As hidden dimensions increase, an increasing fraction of total inference time is spent inside large GEMM operations.

This is especially visible in Megatron-BERT 1.3B.

At that point, optimizations such as scratch reuse, fusion, and reduced intermediate memory traffic still reduce overhead, but they affect a smaller fraction of total execution time.

Both XNNPACK and the PyTorch CPU backend already contain highly optimized matrix multiplication implementations.

As a result:

- BERT-base shows the largest relative benefit from backend/runtime optimization
- BERT-large remains moderately faster while also reducing memory usage
- Megatron-BERT becomes close to PyTorch in latency because large GEMMs dominate execution

This suggests that further performance gains for larger Transformer models increasingly depend on GEMM implementation and scheduling rather than only graph-level or runtime overhead.

## Static Weight Prepacking

Constant fully connected weights are packed once through XNNPACK.

The backend maintains stable packed-weight cache identities so that the packed representation does not depend directly on the lifetime or address of the original raw tensor.

Once packing is complete and the raw constant is no longer needed, the original storage can be released.

This optimization provides two benefits:

1. repeated inference calls avoid weight preparation
2. memory usage is reduced by eliminating redundant weight representations

The effect becomes more important for BERT-large and Megatron-BERT 1.3B because their parameter storage is substantially larger.

## Memory Optimizations

The current backend includes:

- residual buffer aliasing
- in-place ApproxGELU
- residual + LayerNorm buffer overlay
- scratch buffer reuse
- removal of unnecessary temporary input buffers
- Q scaling fused into QKV permutation/copy
- release of raw constants after static weight packing

These optimizations target both latency and resident memory usage.

## Attention Path

The BERT attention path performs:

    QKV projection
        ↓
    QKV permutation
        ↓
    Q scaling
        ↓
    QKᵀ
        ↓
    Softmax
        ↓
    AV
        ↓
    output permutation

The CPU implementation reduces intermediate work by combining compatible transformation stages.

In particular, Q scaling is performed during QKV permutation/copy instead of through a separate tensor pass.

The attention implementation therefore attempts to reduce memory movement in addition to using optimized matrix operations.

## Benchmark Methodology

Final benchmark configurations use:

- Intel Core i9-12900H
- assigned P-cores
- batch size 1
- sequence length 128
- FP32
- 1 / 2 / 4 CPU threads
- repeated isolated benchmark rounds
- median-of-medians latency

Benchmark parameters use deterministic synthetic FP32 weights rather than downloaded pretrained checkpoints.

AITemplate and PyTorch reconstruct the same deterministic tensors for numerical and performance comparison.

## Tested Models

| Model | Layers | Hidden | Heads | Intermediate |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 3072 |
| BERT-large | 24 | 1024 | 16 | 4096 |
| Megatron-BERT 1.3B | 24 | 2048 | 32 | 8192 |

Using models with increasing hidden dimensions also provides a way to observe how the dominant performance bottleneck changes with model scale.

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

| Threads | AITemplate | PyTorch | Speedup |
|---:|---:|---:|---:|
| 1 | 2835.89 ms | 2889.43 ms | 1.019x |
| 2 | 1589.29 ms | 1618.48 ms | 1.018x |
| 4 | 974.21 ms | 992.63 ms | 1.019x |

AITemplate and PyTorch provide approximately equivalent latency within **±2%**.

AITemplate uses approximately **6–7% less resident memory**.

The reduced relative latency advantage is consistent with the increasing dominance of GEMM computation at larger hidden dimensions.

## Thread Scaling

AITemplate scaling relative to single-thread execution:

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

The current implementation should therefore be viewed as a focused investigation of x86 CPU Transformer inference within AITemplate rather than a replacement for the existing CUDA and ROCm backends.

## Current Performance Interpretation

The current results suggest three main observations:

1. AITemplate's ahead-of-time compilation model can be extended to execute complete BERT-family inference on x86 CPUs.
2. Static graph information can be used to reduce weight preparation, intermediate memory usage, and operator-level overhead.
3. As model size grows, large GEMMs increasingly dominate execution, limiting the relative benefit of surrounding compiler/runtime optimizations.

This distinction is important because the remaining bottleneck is no longer simply "CPU backend overhead"; for larger Transformer models, further gains increasingly require improving or better scheduling the GEMM computation itself.

## Upstream Project

This repository is based on Meta's AITemplate project:

https://github.com/facebookincubator/AITemplate

The original upstream README is preserved in [README_UPSTREAM.md](README_UPSTREAM.md).
