import importlib.util
import json
import os

import torch

from aitemplate.backend.cpu import CPU
from aitemplate.backend.cpu.gemm_universal.static_fc import (
    cache_id_from_tensor_name,
)
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


# Reuse the already validated generic CPU BERT encoder implementation.
spec = importlib.util.spec_from_file_location(
    "cpu_bert_full_test",
    "tests/unittest/ops/test_cpu_bert_full_numerical.py",
)
full_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_test)
CPUEncoder = full_test.CPUEncoder


BATCH = 1
SEQ = 128
HIDDEN = 1024
HEADS = 16
INTERMEDIATE = 4096
NUM_LAYERS = 24
VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2
EPS = 1e-12

OUTPUT_DIR = "./tmp/benchmark_cpu_bert_large_full"
MODEL_NAME = "benchmark_cpu_bert_large_full"

PACKED_GEMM_SUFFIXES = (
    "mha.qkv.weight",
    "mha.qkv.bias",
    "mha.proj.weight",
    "mha.proj.bias",
    "ffn1.weight",
    "ffn1.bias",
    "ffn2.weight",
    "ffn2.bias",
)

SUPPORTED_STATIC_FC_OPS = {
    "gemm_rcr_bias_add",
    "gemm_rcr_bias_fast_gelu",
    "gemm_rcr_bias_permute",
    "gemm_rcr_bias_permute_m2n3",
}


def _shape(tensor):
    return [
        int(dim._attrs["values"][0])
        for dim in tensor._attrs["shape"]
    ]


def write_static_fc_prepack_manifest(module, encoder, output_dir):
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
        compiled_tensor = compiled_tensors.get(ait_name)

        if compiled_tensor is None:
            raise RuntimeError(
                f"Compiled graph is missing parameter {ait_name}"
            )

        static_fc_uses = []

        for op in compiled_tensor.dst_ops():
            if op._attrs.get("op") not in SUPPORTED_STATIC_FC_OPS:
                continue

            op_inputs = op._attrs.get("inputs", [])
            if len(op_inputs) < 3:
                continue

            if op_inputs[1] is compiled_tensor:
                static_fc_uses.append((op, "weight"))
            elif op_inputs[2] is compiled_tensor:
                static_fc_uses.append((op, "bias"))

        # LayerNorm gamma/beta can be attached to a fused GEMM op, but they
        # are not FC weight/bias and therefore must not enter the manifest.
        if not static_fc_uses:
            continue

        if len(static_fc_uses) != 1:
            raise RuntimeError(
                "Expected one static-FC use for "
                f"{logical_name}, got {len(static_fc_uses)}"
            )

        op, slot_kind = static_fc_uses[0]

        if logical_name.endswith(".weight"):
            base_name = logical_name[: -len(".weight")]
            logical_kind = "weight"
        elif logical_name.endswith(".bias"):
            base_name = logical_name[: -len(".bias")]
            logical_kind = "bias"
        else:
            raise RuntimeError(
                f"Static FC parameter is not weight/bias: {logical_name}"
            )

        if logical_kind != slot_kind:
            raise RuntimeError(
                "Static FC slot mismatch for "
                f"{logical_name}: logical={logical_kind}, slot={slot_kind}"
            )

        groups.setdefault(base_name, {})[logical_kind] = {
            "logical_name": logical_name,
            "ait_name": ait_name,
            "shape": _shape(param_tensor),
            "op_name": op._attrs["name"],
            "op_type": op._attrs["op"],
        }

    pairs = []

    for base_name in sorted(groups):
        group = groups[base_name]
        if set(group) != {"weight", "bias"}:
            raise RuntimeError(f"Incomplete static-FC pair: {base_name}")

        weight = group["weight"]
        bias = group["bias"]

        if weight["op_name"] != bias["op_name"]:
            raise RuntimeError(
                f"Weight/bias map to different ops for {base_name}"
            )

        weight_shape = weight["shape"]
        bias_shape = bias["shape"]

        if len(weight_shape) != 2 or len(bias_shape) != 1:
            raise RuntimeError(
                f"Unexpected FC shapes for {base_name}: "
                f"weight={weight_shape}, bias={bias_shape}"
            )

        if bias_shape[0] != weight_shape[0]:
            raise RuntimeError(
                f"FC bias mismatch for {base_name}: "
                f"weight={weight_shape}, bias={bias_shape}"
            )

        pairs.append(
            {
                "logical_base": base_name,
                "weight_name": weight["ait_name"],
                "weight_shape": weight_shape,
                "bias_name": bias["ait_name"],
                "bias_shape": bias_shape,
                "prepack_symbol": weight["op_name"] + "_prepack",
                "cache_id": cache_id_from_tensor_name(weight["ait_name"]),
                "n": int(weight_shape[0]),
                "k": int(weight_shape[1]),
                "op_type": weight["op_type"],
            }
        )

    expected_pairs = NUM_LAYERS * 4
    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} static FC pairs, got {len(pairs)}"
        )

    cache_ids = [pair["cache_id"] for pair in pairs]
    if len(cache_ids) != len(set(cache_ids)):
        raise RuntimeError("Static FC cache-id collision detected")

    raw_bytes = 0
    for pair in pairs:
        weight_numel = 1
        bias_numel = 1
        for dim in pair["weight_shape"]:
            weight_numel *= dim
        for dim in pair["bias_shape"]:
            bias_numel *= dim
        raw_bytes += (weight_numel + bias_numel) * 4

    manifest = {
        "version": 1,
        "dtype": "float32",
        "model": "bert-large",
        "layers": NUM_LAYERS,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "pairs": pairs,
        "total_raw_bytes": raw_bytes,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "static_fc_prepack_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print()
    print("===== Static FC prepack manifest =====")
    print("pairs   :", len(pairs))
    print("raw MiB :", raw_bytes / 1024.0 / 1024.0)
    print("manifest:", path)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    Linear.USE_CUDA = False
    MultiheadAttention.USE_CUDA = False

    input_ids = Tensor(
        shape=[BATCH, SEQ],
        dtype="int64",
        name="input_ids",
        is_input=True,
    )
    token_type_ids = Tensor(
        shape=[BATCH, SEQ],
        dtype="int64",
        name="token_type_ids",
        is_input=True,
    )
    position_ids = Tensor(
        shape=[BATCH, SEQ],
        dtype="int64",
        name="position_ids",
        is_input=True,
    )

    word_embeddings = Tensor(
        shape=[VOCAB_SIZE, HIDDEN],
        dtype="float32",
        name="word_embeddings",
    )
    token_type_embeddings = Tensor(
        shape=[TYPE_VOCAB_SIZE, HIDDEN],
        dtype="float32",
        name="token_type_embeddings",
    )
    position_embeddings = Tensor(
        shape=[MAX_POSITION_EMBEDDINGS, HIDDEN],
        dtype="float32",
        name="position_embeddings",
    )
    embedding_gamma = Tensor(
        shape=[HIDDEN],
        dtype="float32",
        name="embedding_gamma",
    )
    embedding_beta = Tensor(
        shape=[HIDDEN],
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
        EPS,
    )

    encoder = CPUEncoder(
        num_layers=NUM_LAYERS,
        batch=BATCH,
        seq=SEQ,
        hidden=HIDDEN,
        heads=HEADS,
        intermediate=INTERMEDIATE,
        eps=EPS,
    )
    encoder.name_parameter_tensor()

    for name, param in encoder.named_parameters():
        if name.endswith("mha.cu_length"):
            continue

        tensor = param.tensor()
        tensor._attrs["is_input"] = False

        if name.endswith(PACKED_GEMM_SUFFIXES):
            tensor._attrs["allow_null_after_pack"] = True

    output = encoder(x)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True

    print("===== Compile BERT-large CPU =====")
    print("layers       :", NUM_LAYERS)
    print("batch        :", BATCH)
    print("sequence     :", SEQ)
    print("hidden       :", HIDDEN)
    print("heads        :", HEADS)
    print("head dim     :", HIDDEN // HEADS)
    print("intermediate :", INTERMEDIATE)

    module = compile_model(
        output,
        CPU(),
        "./tmp",
        MODEL_NAME,
    )

    write_static_fc_prepack_manifest(
        module,
        encoder,
        OUTPUT_DIR,
    )

    print()
    print("PASS: BERT-large graph compiled")


if __name__ == "__main__":
    main()
