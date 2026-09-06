import ctypes
import gc
import hashlib
import importlib.util
import json
import os
import resource

import torch
import torch.nn.functional as F

from aitemplate.compiler.model import (
    AITData,
    Model,
    torch_to_ait_data,
)
from aitemplate.frontend.nn.attention import (
    MultiheadAttention,
)
from aitemplate.frontend.nn.linear import (
    Linear,
)


BATCH = 1
SEQ = 128
HIDDEN = 2048
HEADS = 32
HEAD_DIM = 64
INTERMEDIATE = 8192
NUM_LAYERS = 24

VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2

EPS = 1e-12

MODEL_DIR = (
    "./tmp/"
    "benchmark_cpu_megatron_bert_1_3b"
)

SO_PATH = os.path.join(
    MODEL_DIR,
    "test.so",
)

MANIFEST_PATH = os.path.join(
    MODEL_DIR,
    "static_fc_prepack_manifest.json",
)


spec = importlib.util.spec_from_file_location(
    "cpu_bert_full_test",
    "tests/unittest/ops/"
    "test_cpu_bert_full_numerical.py",
)

full_test = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    full_test
)

CPUEncoder = full_test.CPUEncoder
pytorch_encoder_layer = (
    full_test.pytorch_encoder_layer
)


def memory_mib():
    rss = None
    peak = None

    with open(
        "/proc/self/status",
        "r",
    ) as f:
        for line in f:
            if line.startswith(
                "VmRSS:"
            ):
                rss = (
                    int(
                        line.split()[1]
                    )
                    / 1024.0
                )

            elif line.startswith(
                "VmHWM:"
            ):
                peak = (
                    int(
                        line.split()[1]
                    )
                    / 1024.0
                )

    return rss, peak


def trim_allocator():
    gc.collect()

    try:
        libc = ctypes.CDLL(
            "libc.so.6"
        )

        libc.malloc_trim(0)

    except Exception:
        pass


def seed_for(name):
    digest = hashlib.sha256(
        name.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
    ) & 0x7FFFFFFFFFFFFFFF


def random_tensor(
    name,
    shape,
):
    generator = (
        torch.Generator(
            device="cpu"
        )
    )

    generator.manual_seed(
        seed_for(name)
    )

    return (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    )


def make_parameter(
    name,
    shape,
):
    if (
        name.endswith(
            "ln1.weight"
        )
        or name.endswith(
            "ln2.weight"
        )
    ):
        return torch.ones(
            shape,
            dtype=torch.float32,
        )

    if (
        name.endswith(
            "ln1.bias"
        )
        or name.endswith(
            "ln2.bias"
        )
    ):
        return torch.zeros(
            shape,
            dtype=torch.float32,
        )

    return random_tensor(
        name,
        shape,
    )


def parameter_metadata():
    Linear.USE_CUDA = False
    MultiheadAttention.USE_CUDA = False

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

    result = {}

    for name, param in (
        encoder.named_parameters()
    ):
        if name.endswith(
            "mha.cu_length"
        ):
            continue

        tensor = param.tensor()

        result[name] = {
            "ait_name":
                tensor._attrs[
                    "name"
                ],

            "shape": [
                int(
                    dim._attrs[
                        "values"
                    ][0]
                )
                for dim
                in tensor._attrs[
                    "shape"
                ]
            ],
        }

    return result


def create_common_data():
    ids_generator = torch.Generator(
        device="cpu"
    )

    ids_generator.manual_seed(
        20260904
    )

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH, SEQ),
        generator=ids_generator,
        dtype=torch.int64,
    )

    token_type_ids = torch.zeros(
        BATCH,
        SEQ,
        dtype=torch.int64,
    )

    position_ids = (
        torch.arange(
            SEQ,
            dtype=torch.int64,
        )
        .reshape(
            1,
            SEQ,
        )
        .expand(
            BATCH,
            -1,
        )
        .contiguous()
    )

    return {
        "input_ids":
            input_ids,

        "token_type_ids":
            token_type_ids,

        "position_ids":
            position_ids,

        "word_embeddings":
            random_tensor(
                "word_embeddings",
                (
                    VOCAB_SIZE,
                    HIDDEN,
                ),
            ),

        "token_type_embeddings":
            random_tensor(
                "token_type_embeddings",
                (
                    TYPE_VOCAB_SIZE,
                    HIDDEN,
                ),
            ),

        "position_embeddings":
            random_tensor(
                "position_embeddings",
                (
                    MAX_POSITION_EMBEDDINGS,
                    HIDDEN,
                ),
            ),

        "embedding_gamma":
            torch.ones(
                HIDDEN,
                dtype=torch.float32,
            ),

        "embedding_beta":
            torch.zeros(
                HIDDEN,
                dtype=torch.float32,
            ),
    }


def load_manifest():
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    if len(
        manifest["pairs"]
    ) != NUM_LAYERS * 4:
        raise RuntimeError(
            "Expected 96 static-FC "
            f"pairs, got "
            f"{len(manifest['pairs'])}"
        )

    return manifest


def set_persistent_constants(
    module,
    common,
    metadata,
):
    persistent = {
        "word_embeddings":
            common[
                "word_embeddings"
            ],

        "token_type_embeddings":
            common[
                "token_type_embeddings"
            ],

        "position_embeddings":
            common[
                "position_embeddings"
            ],

        "embedding_gamma":
            common[
                "embedding_gamma"
            ],

        "embedding_beta":
            common[
                "embedding_beta"
            ],
    }

    constants = {
        name:
            torch_to_ait_data(
                value
            )
        for name, value
        in persistent.items()
    }

    for logical_name, info in (
        metadata.items()
    ):
        if not (
            logical_name.endswith(
                "ln1.weight"
            )
            or logical_name.endswith(
                "ln1.bias"
            )
            or logical_name.endswith(
                "ln2.weight"
            )
            or logical_name.endswith(
                "ln2.bias"
            )
        ):
            continue

        value = make_parameter(
            logical_name,
            info["shape"],
        )

        persistent[
            info["ait_name"]
        ] = value

        constants[
            info["ait_name"]
        ] = torch_to_ait_data(
            value
        )

    module.set_many_constants(
        constants
    )

    return persistent


def prepack_all_fc(
    module,
    manifest,
):
    library = ctypes.CDLL(
        SO_PATH
    )

    total = len(
        manifest["pairs"]
    )

    print()
    print(
        "===== Incremental static-FC "
        "prepack ====="
    )

    print(
        "pairs            :",
        total,
    )

    print(
        "raw FC total GiB :",
        manifest[
            "total_raw_bytes"
        ]
        / 1024**3,
    )

    for index, pair in enumerate(
        manifest["pairs"],
        start=1,
    ):
        base = pair[
            "logical_base"
        ]

        weight = make_parameter(
            base + ".weight",
            pair[
                "weight_shape"
            ],
        )

        bias = make_parameter(
            base + ".bias",
            pair[
                "bias_shape"
            ],
        )

        function = getattr(
            library,
            pair[
                "prepack_symbol"
            ],
        )

        function.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]

        function.restype = (
            ctypes.c_int
        )

        result = function(
            ctypes.c_uint64(
                pair["cache_id"]
            ),

            ctypes.c_void_p(
                weight.data_ptr()
            ),

            ctypes.c_void_p(
                bias.data_ptr()
            ),

            ctypes.c_size_t(
                pair["n"]
            ),

            ctypes.c_size_t(
                pair["k"]
            ),
        )

        if result != 1:
            raise RuntimeError(
                "Static-FC prepack "
                f"failed for {base}"
            )

        weight_data = (
            torch_to_ait_data(
                weight
            )
        )

        bias_data = (
            torch_to_ait_data(
                bias
            )
        )

        module.set_many_constants(
            {
                pair["weight_name"]:
                    AITData(
                        0,
                        weight_data.shape,
                        weight_data.dtype,
                    ),

                pair["bias_name"]:
                    AITData(
                        0,
                        bias_data.shape,
                        bias_data.dtype,
                    ),
            }
        )

        del weight
        del bias

        if (
            index == 1
            or index % 8 == 0
            or index == total
        ):
            trim_allocator()

            rss, peak = (
                memory_mib()
            )

            print(
                f"[{index:02d}/{total}] "
                f"{base:<28} "
                f"RSS={rss:.1f} MiB  "
                f"peak={peak:.1f} MiB"
            )


def run_ait(
    common,
    metadata,
    manifest,
):
    module = Model(
        SO_PATH
    )

    persistent = (
        set_persistent_constants(
            module,
            common,
            metadata,
        )
    )

    prepack_all_fc(
        module,
        manifest,
    )

    ait_inputs = {
        "input_ids":
            torch_to_ait_data(
                common[
                    "input_ids"
                ]
            ),

        "token_type_ids":
            torch_to_ait_data(
                common[
                    "token_type_ids"
                ]
            ),

        "position_ids":
            torch_to_ait_data(
                common[
                    "position_ids"
                ]
            ),
    }

    output = torch.empty(
        BATCH,
        SEQ,
        HIDDEN,
        dtype=torch.float32,
    )

    module.run(
        ait_inputs,
        {
            "output":
                torch_to_ait_data(
                    output
                )
        },
    )

    # Keep constants alive until the
    # inference is completely finished.
    _ = persistent

    return output


@torch.no_grad()
def run_reference(
    common,
    metadata,
):
    x = (
        F.embedding(
            common["input_ids"],
            common[
                "word_embeddings"
            ],
        )
        + F.embedding(
            common[
                "token_type_ids"
            ],
            common[
                "token_type_embeddings"
            ],
        )
        + F.embedding(
            common[
                "position_ids"
            ],
            common[
                "position_embeddings"
            ],
        )
    )

    x = F.layer_norm(
        x,
        (HIDDEN,),
        common[
            "embedding_gamma"
        ],
        common[
            "embedding_beta"
        ],
        EPS,
    )

    for layer_idx in range(
        NUM_LAYERS
    ):
        prefix = (
            f"layers."
            f"{layer_idx}."
        )

        names = (
            "mha.qkv.weight",
            "mha.qkv.bias",
            "mha.proj.weight",
            "mha.proj.bias",
            "ln1.weight",
            "ln1.bias",
            "ffn1.weight",
            "ffn1.bias",
            "ffn2.weight",
            "ffn2.bias",
            "ln2.weight",
            "ln2.bias",
        )

        params = {}

        for suffix in names:
            logical_name = (
                prefix + suffix
            )

            params[
                logical_name
            ] = make_parameter(
                logical_name,
                metadata[
                    logical_name
                ]["shape"],
            )

        x = pytorch_encoder_layer(
            x,
            params,
            prefix,
            BATCH,
            SEQ,
            HIDDEN,
            HEADS,
            HEAD_DIM,
            EPS,
        )

        del params

        trim_allocator()

        print(
            "PyTorch layer",
            layer_idx + 1,
            "/",
            NUM_LAYERS,
            "complete",
        )

    return x


def main():
    if not os.path.exists(
        SO_PATH
    ):
        raise RuntimeError(
            f"{SO_PATH} not found. "
            "Run "
            "benchmark_cpu_megatron_bert_full.py "
            "first."
        )

    if not os.path.exists(
        MANIFEST_PATH
    ):
        raise RuntimeError(
            f"{MANIFEST_PATH} "
            "not found."
        )

    print(
        "===== Megatron-BERT "
        "1.3B full inference ====="
    )

    print(
        "layers       :",
        NUM_LAYERS,
    )
    print(
        "batch        :",
        BATCH,
    )
    print(
        "sequence     :",
        SEQ,
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
        HEAD_DIM,
    )
    print(
        "intermediate :",
        INTERMEDIATE,
    )

    metadata = (
        parameter_metadata()
    )

    manifest = (
        load_manifest()
    )

    common = (
        create_common_data()
    )

    print()
    print(
        "===== AITemplate ====="
    )

    ait_output = run_ait(
        common,
        metadata,
        manifest,
    )

    rss, peak = memory_mib()

    print(
        "AIT inference complete"
    )

    print(
        "RSS MiB :",
        rss,
    )

    print(
        "Peak MiB:",
        peak,
    )

    print()
    print(
        "===== PyTorch reference ====="
    )

    reference = run_reference(
        common,
        metadata,
    )

    diff = (
        ait_output
        - reference
    ).abs()

    print()
    print(
        "===== Numerical result ====="
    )

    print(
        "output shape :",
        list(
            ait_output.shape
        ),
    )

    print(
        "max abs diff :",
        diff.max().item(),
    )

    print(
        "mean abs diff:",
        diff.mean().item(),
    )

    torch.testing.assert_close(
        ait_output,
        reference,
        rtol=5e-3,
        atol=5e-3,
    )

    print(
        "MEGATRON-BERT 1.3B "
        "FULL INFERENCE PASS"
    )


if __name__ == "__main__":
    main()
