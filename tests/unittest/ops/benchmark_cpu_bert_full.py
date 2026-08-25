import importlib.util
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

# Current AITemplate/XNNPACK path uses no pthreadpool.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Keep the in-process sanity benchmark on the same CPU used by the
# isolated benchmark. This reduces migration/frequency noise.
_allowed_cpus = sorted(os.sched_getaffinity(0))
BENCH_CPU = _allowed_cpus[0]
os.sched_setaffinity(0, {BENCH_CPU})

from aitemplate.backend.cpu import CPU
from aitemplate.backend.cpu.gemm_universal.static_fc import (
    cache_id_from_tensor_name,
)
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


# Reuse the already validated full-BERT definitions.
spec = importlib.util.spec_from_file_location(
    "cpu_bert_full_test",
    "tests/unittest/ops/test_cpu_bert_full_numerical.py",
)
full_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_test)

CPUEncoder = full_test.CPUEncoder
pytorch_encoder_layer = full_test.pytorch_encoder_layer


def benchmark(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()

    times_ms = []

    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()

        times_ms.append(
            (end - start) * 1000.0
        )

    return {
        "median": statistics.median(times_ms),
        "mean": statistics.mean(times_ms),
        "min": min(times_ms),
        "max": max(times_ms),
    }



def write_static_fc_prepack_manifest(
    module,
    encoder,
    output_dir,
):
    """Write the exact constant -> generated prepack mapping for this model."""

    supported_static_fc_ops = {
        "gemm_rcr_bias_add",
        "gemm_rcr_bias_fast_gelu",
        "gemm_rcr_bias_permute",
        "gemm_rcr_bias_permute_m2n3",
    }

    compiled_tensors = {
        tensor._attrs["name"]: tensor
        for tensor in module.debug_sorted_graph
    }

    groups = {}

    for logical_name, param in encoder.named_parameters():
        if logical_name.endswith("mha.cu_length"):
            continue

        param_tensor = param.tensor()
        ait_name = param_tensor._attrs["name"]

        compiled_tensor = compiled_tensors.get(
            ait_name
        )

        if compiled_tensor is None:
            raise RuntimeError(
                "Compiled graph is missing parameter "
                f"{ait_name}"
            )

        static_dst_ops = [
            op
            for op in compiled_tensor.dst_ops()
            if op._attrs.get("op")
            in supported_static_fc_ops
        ]

        if not static_dst_ops:
            continue

        if len(static_dst_ops) != 1:
            raise RuntimeError(
                "Expected one static-FC destination op "
                f"for {logical_name}, got "
                f"{len(static_dst_ops)}"
            )

        if logical_name.endswith(".weight"):
            base_name = logical_name[:-len(".weight")]
            kind = "weight"
        elif logical_name.endswith(".bias"):
            base_name = logical_name[:-len(".bias")]
            kind = "bias"
        else:
            raise RuntimeError(
                "Static FC parameter must be a weight "
                f"or bias: {logical_name}"
            )

        shape = [
            int(dim._attrs["values"][0])
            for dim in param_tensor._attrs["shape"]
        ]

        op = static_dst_ops[0]

        groups.setdefault(
            base_name,
            {},
        )[kind] = {
            "logical_name": logical_name,
            "ait_name": ait_name,
            "shape": shape,
            "op_name": op._attrs["name"],
            "op_type": op._attrs["op"],
        }

    pairs = []

    for base_name in sorted(groups):
        group = groups[base_name]

        if set(group) != {"weight", "bias"}:
            raise RuntimeError(
                f"Incomplete static-FC pair: {base_name}"
            )

        weight = group["weight"]
        bias = group["bias"]

        if weight["op_name"] != bias["op_name"]:
            raise RuntimeError(
                "Weight and bias map to different generated "
                f"functions for {base_name}"
            )

        weight_shape = weight["shape"]

        if len(weight_shape) != 2:
            raise RuntimeError(
                f"Expected 2D FC weight for {base_name}, "
                f"got {weight_shape}"
            )

        cache_id = cache_id_from_tensor_name(
            weight["ait_name"]
        )

        pairs.append(
            {
                "logical_base": base_name,
                "weight_name": weight["ait_name"],
                "weight_shape": weight_shape,
                "bias_name": bias["ait_name"],
                "bias_shape": bias["shape"],
                "prepack_symbol":
                    weight["op_name"] + "_prepack",
                "cache_id": cache_id,
                "n": int(weight_shape[0]),
                "k": int(weight_shape[1]),
                "op_type": weight["op_type"],
            }
        )

    expected_pairs = 12 * 4

    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} static FC pairs, "
            f"got {len(pairs)}"
        )

    cache_ids = [
        pair["cache_id"]
        for pair in pairs
    ]

    if len(cache_ids) != len(set(cache_ids)):
        raise RuntimeError(
            "Static FC cache-id collision detected"
        )

    raw_bytes = 0

    for pair in pairs:
        weight_numel = 1
        bias_numel = 1

        for dim in pair["weight_shape"]:
            weight_numel *= dim

        for dim in pair["bias_shape"]:
            bias_numel *= dim

        raw_bytes += (
            weight_numel + bias_numel
        ) * 4

    manifest = {
        "version": 1,
        "dtype": "float32",
        "pairs": pairs,
        "total_raw_bytes": raw_bytes,
    }

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    manifest_path = os.path.join(
        output_dir,
        "static_fc_prepack_manifest.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print("===== Static FC prepack manifest =====")
    print("pairs        :", len(pairs))
    print(
        "raw MiB      :",
        raw_bytes / 1024.0 / 1024.0,
    )
    print("manifest     :", manifest_path)


def main():
    batch = 1
    seq = 128
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    intermediate = 3072
    num_layers = 12

    vocab_size = 30522
    max_position_embeddings = 512
    type_vocab_size = 2

    eps = 1e-12

    Linear.USE_CUDA = False
    MultiheadAttention.USE_CUDA = False

    # ------------------------------------------------------------
    # Build AITemplate graph
    # ------------------------------------------------------------

    input_ids = Tensor(
        shape=[batch, seq],
        dtype="int64",
        name="input_ids",
        is_input=True,
    )

    token_type_ids = Tensor(
        shape=[batch, seq],
        dtype="int64",
        name="token_type_ids",
        is_input=True,
    )

    position_ids = Tensor(
        shape=[batch, seq],
        dtype="int64",
        name="position_ids",
        is_input=True,
    )

    word_embeddings = Tensor(
        shape=[vocab_size, hidden],
        dtype="float32",
        name="word_embeddings",
    )

    token_type_embeddings = Tensor(
        shape=[type_vocab_size, hidden],
        dtype="float32",
        name="token_type_embeddings",
    )

    position_embeddings = Tensor(
        shape=[max_position_embeddings, hidden],
        dtype="float32",
        name="position_embeddings",
    )

    embedding_gamma = Tensor(
        shape=[hidden],
        dtype="float32",
        name="embedding_gamma",
    )

    embedding_beta = Tensor(
        shape=[hidden],
        dtype="float32",
        name="embedding_beta",
    )

    x = ops.bert_embeddings()(
        input_ids,
        token_type_ids,
        position_ids,
        word_embeddings,
        token_type_embeddings,
        position_embeddings,
        embedding_gamma,
        embedding_beta,
        eps,
    )

    encoder = CPUEncoder(
        num_layers=num_layers,
        batch=batch,
        seq=seq,
        hidden=hidden,
        heads=heads,
        intermediate=intermediate,
        eps=eps,
    )

    encoder.name_parameter_tensor()

    packed_gemm_suffixes = (
        "mha.qkv.weight",
        "mha.qkv.bias",
        "mha.proj.weight",
        "mha.proj.bias",
        "ffn1.weight",
        "ffn1.bias",
        "ffn2.weight",
        "ffn2.bias",
    )

    for name, param in encoder.named_parameters():
        if name.endswith("mha.cu_length"):
            continue

        tensor = param.tensor()

        # Encoder parameters are model constants, not runtime inputs.
        tensor._attrs["is_input"] = False

        # These constants are consumed only by persistent static XNNPACK
        # FC operators. Once their operator is prepacked, the raw pointer
        # may legally become nullptr. codegen.py uses this flag only to
        # omit the generic constant-null guard; the GEMM cache itself is
        # still keyed by the compile-time cache_id.
        if name.endswith(packed_gemm_suffixes):
            tensor._attrs["allow_null_after_pack"] = True

    output_tensor = encoder(x)
    output_tensor._attrs["name"] = "output"
    output_tensor._attrs["is_output"] = True

    print("===== Compile =====")

    module = compile_model(
        output_tensor,
        CPU(),
        "./tmp",
        "benchmark_cpu_bert_full",
    )

    write_static_fc_prepack_manifest(
        module,
        encoder,
        "./tmp/benchmark_cpu_bert_full",
    )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    torch.manual_seed(0)

    input_ids_pt = torch.randint(
        0,
        vocab_size,
        (batch, seq),
        dtype=torch.int64,
    )

    token_type_ids_pt = torch.zeros(
        batch,
        seq,
        dtype=torch.int64,
    )

    position_ids_pt = (
        torch.arange(seq, dtype=torch.int64)
        .reshape(1, seq)
        .expand(batch, -1)
        .contiguous()
    )

    word_embeddings_pt = (
        torch.randn(vocab_size, hidden) * 0.02
    )

    token_type_embeddings_pt = (
        torch.randn(type_vocab_size, hidden) * 0.02
    )

    position_embeddings_pt = (
        torch.randn(max_position_embeddings, hidden) * 0.02
    )

    embedding_gamma_pt = torch.ones(hidden)
    embedding_beta_pt = torch.zeros(hidden)

    logical_params = {}

    ait_inputs = {
        "input_ids":
            torch_to_ait_data(input_ids_pt),
        "token_type_ids":
            torch_to_ait_data(token_type_ids_pt),
        "position_ids":
            torch_to_ait_data(position_ids_pt),
    }

    ait_constants = {
        "word_embeddings":
            torch_to_ait_data(word_embeddings_pt),
        "token_type_embeddings":
            torch_to_ait_data(token_type_embeddings_pt),
        "position_embeddings":
            torch_to_ait_data(position_embeddings_pt),
        "embedding_gamma":
            torch_to_ait_data(embedding_gamma_pt),
        "embedding_beta":
            torch_to_ait_data(embedding_beta_pt),
    }

    for name, param in encoder.named_parameters():
        if name.endswith("mha.cu_length"):
            continue

        tensor = param.tensor()

        shape = [
            dim._attrs["values"][0]
            for dim in tensor._attrs["shape"]
        ]

        if (
            "ln1.weight" in name
            or "ln2.weight" in name
        ):
            value = torch.ones(shape)

        elif (
            "ln1.bias" in name
            or "ln2.bias" in name
        ):
            value = torch.zeros(shape)

        else:
            value = torch.randn(shape) * 0.02

        logical_params[name] = value

        ait_constants[tensor._attrs["name"]] = (
            torch_to_ait_data(value)
        )

    ait_output = torch.empty(
        batch,
        seq,
        hidden,
        dtype=torch.float32,
    )

    ait_outputs = {
        "output": torch_to_ait_data(ait_output),
    }

    # ------------------------------------------------------------
    # AITemplate forward
    # ------------------------------------------------------------

    module.set_many_constants(ait_constants)

    def ait_forward():
        module.run(
            ait_inputs,
            ait_outputs,
        )

    # ------------------------------------------------------------
    # PyTorch reference forward
    # ------------------------------------------------------------

    @torch.no_grad()
    def pytorch_forward():
        x_pt = (
            F.embedding(
                input_ids_pt,
                word_embeddings_pt,
            )
            + F.embedding(
                token_type_ids_pt,
                token_type_embeddings_pt,
            )
            + F.embedding(
                position_ids_pt,
                position_embeddings_pt,
            )
        )

        x_pt = F.layer_norm(
            x_pt,
            (hidden,),
            embedding_gamma_pt,
            embedding_beta_pt,
            eps,
        )

        for layer_idx in range(num_layers):
            x_pt = pytorch_encoder_layer(
                x_pt,
                logical_params,
                "layers." + str(layer_idx) + ".",
                batch,
                seq,
                hidden,
                heads,
                head_dim,
                eps,
            )

        return x_pt

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    print()
    print("===== Benchmark configuration =====")
    print("PyTorch threads :", torch.get_num_threads())
    print("pinned CPU      :", BENCH_CPU)
    print("layers          :", num_layers)
    print("batch           :", batch)
    print("sequence        :", seq)
    print("hidden          :", hidden)

    print()
    print("===== AITemplate CPU =====")

    ait_stats = benchmark(
        ait_forward,
    )

    print("median ms:", ait_stats["median"])
    print("mean ms  :", ait_stats["mean"])
    print("min ms   :", ait_stats["min"])
    print("max ms   :", ait_stats["max"])

    print()
    print("===== PyTorch CPU =====")

    torch_stats = benchmark(
        pytorch_forward,
    )

    print("median ms:", torch_stats["median"])
    print("mean ms  :", torch_stats["mean"])
    print("min ms   :", torch_stats["min"])
    print("max ms   :", torch_stats["max"])

    speedup = (
        torch_stats["median"]
        / ait_stats["median"]
    )

    print()
    print("===== Result =====")
    print("AITemplate median:", ait_stats["median"], "ms")
    print("PyTorch median   :", torch_stats["median"], "ms")
    print("AIT/PyTorch speedup:", speedup, "x")

    if speedup > 1.0:
        print("AITemplate is faster")
    else:
        print("PyTorch is faster")


if __name__ == "__main__":
    main()
