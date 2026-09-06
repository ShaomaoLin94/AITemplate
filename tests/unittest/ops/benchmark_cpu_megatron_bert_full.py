import importlib.util
import json
import os

from aitemplate.backend.cpu import CPU
from aitemplate.backend.cpu.gemm_universal.static_fc import (
    cache_id_from_tensor_name,
)
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


BATCH = 1
SEQ = 128
HIDDEN = 2048
HEADS = 32
INTERMEDIATE = 8192
NUM_LAYERS = 24

VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2

EPS = 1e-12

MODEL_NAME = "benchmark_cpu_megatron_bert_1_3b"
OUTPUT_DIR = f"./tmp/{MODEL_NAME}"


spec = importlib.util.spec_from_file_location(
    "cpu_bert_full_test",
    "tests/unittest/ops/test_cpu_bert_full_numerical.py",
)

full_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_test)

CPUEncoder = full_test.CPUEncoder


def write_static_fc_prepack_manifest(
    module,
    encoder,
):
    supported_ops = {
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
        if logical_name.endswith(
            "mha.cu_length"
        ):
            continue

        tensor = param.tensor()
        ait_name = tensor._attrs["name"]

        compiled_tensor = compiled_tensors.get(
            ait_name
        )

        if compiled_tensor is None:
            raise RuntimeError(
                "Compiled graph missing parameter "
                f"{ait_name}"
            )

        uses = []

        for op in compiled_tensor.dst_ops():
            if (
                op._attrs.get("op")
                not in supported_ops
            ):
                continue

            inputs = op._attrs.get(
                "inputs",
                [],
            )

            if len(inputs) < 3:
                continue

            if inputs[1] is compiled_tensor:
                uses.append(
                    (op, "weight")
                )

            elif inputs[2] is compiled_tensor:
                uses.append(
                    (op, "bias")
                )

        if not uses:
            continue

        if len(uses) != 1:
            raise RuntimeError(
                f"Expected one static-FC use for "
                f"{logical_name}, got {len(uses)}"
            )

        op, slot_kind = uses[0]

        if logical_name.endswith(
            ".weight"
        ):
            base_name = (
                logical_name[
                    :-len(".weight")
                ]
            )

            logical_kind = "weight"

        elif logical_name.endswith(
            ".bias"
        ):
            base_name = (
                logical_name[
                    :-len(".bias")
                ]
            )

            logical_kind = "bias"

        else:
            raise RuntimeError(
                "Unexpected static-FC "
                f"parameter: {logical_name}"
            )

        if logical_kind != slot_kind:
            raise RuntimeError(
                "Static-FC slot mismatch: "
                f"{logical_name}"
            )

        shape = [
            int(
                dim._attrs[
                    "values"
                ][0]
            )
            for dim
            in tensor._attrs["shape"]
        ]

        groups.setdefault(
            base_name,
            {},
        )[logical_kind] = {
            "ait_name":
                ait_name,
            "shape":
                shape,
            "op_name":
                op._attrs["name"],
            "op_type":
                op._attrs["op"],
        }

    pairs = []

    for base_name in sorted(groups):
        group = groups[base_name]

        if set(group) != {
            "weight",
            "bias",
        }:
            raise RuntimeError(
                "Incomplete FC pair: "
                f"{base_name}"
            )

        weight = group["weight"]
        bias = group["bias"]

        if (
            weight["op_name"]
            != bias["op_name"]
        ):
            raise RuntimeError(
                "Weight/bias generated "
                "function mismatch: "
                f"{base_name}"
            )

        if len(
            weight["shape"]
        ) != 2:
            raise RuntimeError(
                "FC weight must be 2D: "
                f"{base_name}"
            )

        if len(
            bias["shape"]
        ) != 1:
            raise RuntimeError(
                "FC bias must be 1D: "
                f"{base_name}"
            )

        cache_id = (
            cache_id_from_tensor_name(
                weight["ait_name"]
            )
        )

        pairs.append(
            {
                "logical_base":
                    base_name,

                "weight_name":
                    weight["ait_name"],

                "weight_shape":
                    weight["shape"],

                "bias_name":
                    bias["ait_name"],

                "bias_shape":
                    bias["shape"],

                "prepack_symbol":
                    (
                        weight["op_name"]
                        + "_prepack"
                    ),

                "cache_id":
                    cache_id,

                "n":
                    int(
                        weight[
                            "shape"
                        ][0]
                    ),

                "k":
                    int(
                        weight[
                            "shape"
                        ][1]
                    ),

                "op_type":
                    weight["op_type"],
            }
        )

    expected_pairs = (
        NUM_LAYERS * 4
    )

    if (
        len(pairs)
        != expected_pairs
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_pairs} "
            "static-FC pairs, "
            f"got {len(pairs)}"
        )

    cache_ids = [
        pair["cache_id"]
        for pair in pairs
    ]

    if (
        len(cache_ids)
        != len(set(cache_ids))
    ):
        raise RuntimeError(
            "Static-FC cache-id "
            "collision detected"
        )

    raw_bytes = 0

    for pair in pairs:
        weight_numel = 1
        bias_numel = 1

        for dim in pair[
            "weight_shape"
        ]:
            weight_numel *= dim

        for dim in pair[
            "bias_shape"
        ]:
            bias_numel *= dim

        raw_bytes += (
            weight_numel
            + bias_numel
        ) * 4

    manifest = {
        "version": 1,
        "model":
            "Megatron-BERT 1.3B",
        "dtype":
            "float32",
        "pairs":
            pairs,
        "total_raw_bytes":
            raw_bytes,
    }

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        OUTPUT_DIR,
        "static_fc_prepack_manifest.json",
    )

    with open(
        path,
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
    print(
        "===== Static FC manifest ====="
    )

    print(
        "pairs      :",
        len(pairs),
    )

    print(
        "raw FC GiB :",
        raw_bytes
        / 1024**3,
    )

    print(
        "manifest   :",
        path,
    )


def main():
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
        shape=[
            VOCAB_SIZE,
            HIDDEN,
        ],
        dtype="float32",
        name="word_embeddings",
    )

    token_type_embeddings = Tensor(
        shape=[
            TYPE_VOCAB_SIZE,
            HIDDEN,
        ],
        dtype="float32",
        name="token_type_embeddings",
    )

    position_embeddings = Tensor(
        shape=[
            MAX_POSITION_EMBEDDINGS,
            HIDDEN,
        ],
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

    packed_suffixes = (
        "mha.qkv.weight",
        "mha.qkv.bias",
        "mha.proj.weight",
        "mha.proj.bias",
        "ffn1.weight",
        "ffn1.bias",
        "ffn2.weight",
        "ffn2.bias",
    )

    for name, param in (
        encoder.named_parameters()
    ):
        if name.endswith(
            "mha.cu_length"
        ):
            continue

        tensor = param.tensor()

        tensor._attrs[
            "is_input"
        ] = False

        if name.endswith(
            packed_suffixes
        ):
            tensor._attrs[
                "allow_null_after_pack"
            ] = True

    output = encoder(x)

    output._attrs["name"] = (
        "output"
    )

    output._attrs[
        "is_output"
    ] = True

    print(
        "===== Megatron-BERT "
        "1.3B compile ====="
    )

    print(
        "layers       :",
        NUM_LAYERS,
    )
    print(
        "hidden       :",
        HIDDEN,
    )
    print(
        "heads        :",
        HEADS,
    )
    print(
        "head dim     :",
        HIDDEN // HEADS,
    )
    print(
        "intermediate :",
        INTERMEDIATE,
    )

    module = compile_model(
        output,
        CPU(),
        "./tmp",
        MODEL_NAME,
    )

    write_static_fc_prepack_manifest(
        module,
        encoder,
    )

    print()
    print("COMPILE PASS")


if __name__ == "__main__":
    main()
