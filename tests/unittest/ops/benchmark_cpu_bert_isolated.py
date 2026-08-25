import ctypes
import gc
import importlib.util
import json
import os
import resource
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from aitemplate.compiler.model import AITData, Model, torch_to_ait_data
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


BENCH_THREADS = max(
    1,
    int(os.environ.get(
        "AIT_CPU_THREADS",
        "1",
    )),
)

torch.set_num_threads(BENCH_THREADS)
torch.set_num_interop_threads(1)


def _physical_core_key(cpu_id):
    base = (
        f"/sys/devices/system/cpu/cpu{cpu_id}/topology"
    )

    try:
        with open(
            base + "/physical_package_id",
            "r",
        ) as f:
            package_id = int(f.read().strip())

        with open(
            base + "/core_id",
            "r",
        ) as f:
            core_id = int(f.read().strip())

        return (package_id, core_id)

    except Exception:
        return (0, cpu_id)


# Give AITemplate and PyTorch the same number of distinct physical
# cores. pthreadpool worker threads inherit this affinity mask.
allowed = sorted(os.sched_getaffinity(0))
selected_cpus = []
seen_cores = set()

for candidate in allowed:
    key = _physical_core_key(candidate)

    if key in seen_cores:
        continue

    seen_cores.add(key)
    selected_cpus.append(candidate)

    if len(selected_cpus) == BENCH_THREADS:
        break

if len(selected_cpus) < BENCH_THREADS:
    for candidate in allowed:
        if candidate in selected_cpus:
            continue

        selected_cpus.append(candidate)

        if len(selected_cpus) == BENCH_THREADS:
            break

if len(selected_cpus) < BENCH_THREADS:
    raise RuntimeError(
        f"Requested {BENCH_THREADS} CPU threads, "
        f"but only {len(allowed)} logical CPUs are available"
    )

os.sched_setaffinity(
    0,
    set(selected_cpus),
)

cpu = ",".join(
    str(x)
    for x in selected_cpus
)


spec = importlib.util.spec_from_file_location(
    "cpu_bert_full_test",
    "tests/unittest/ops/test_cpu_bert_full_numerical.py",
)

full_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_test)

CPUEncoder = full_test.CPUEncoder
pytorch_encoder_layer = full_test.pytorch_encoder_layer


BATCH = 1
SEQ = 128
HIDDEN = 768
HEADS = 12
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 3072
NUM_LAYERS = 12

VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2

EPS = 1e-12

SO_PATH = "./tmp/benchmark_cpu_bert_full/test.so"
PREPACK_MANIFEST_PATH = (
    "./tmp/benchmark_cpu_bert_full/"
    "static_fc_prepack_manifest.json"
)


def benchmark(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )

    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }


def get_memory():
    usage = resource.getrusage(
        resource.RUSAGE_SELF
    )

    rss = None
    hwm = None

    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024.0

            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0

    return {
        "peak_rss": usage.ru_maxrss / 1024.0,
        "vmrss": rss,
        "vmhwm": hwm,
    }


def _trim_cpu_allocator():
    gc.collect()

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _make_common_data():
    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH, SEQ),
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
        .reshape(1, SEQ)
        .expand(BATCH, -1)
        .contiguous()
    )

    word_embeddings = (
        torch.randn(
            VOCAB_SIZE,
            HIDDEN,
        )
        * 0.02
    )

    token_type_embeddings = (
        torch.randn(
            TYPE_VOCAB_SIZE,
            HIDDEN,
        )
        * 0.02
    )

    position_embeddings = (
        torch.randn(
            MAX_POSITION_EMBEDDINGS,
            HIDDEN,
        )
        * 0.02
    )

    embedding_gamma = torch.ones(
        HIDDEN
    )

    embedding_beta = torch.zeros(
        HIDDEN
    )

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
        "word_embeddings": word_embeddings,
        "token_type_embeddings":
            token_type_embeddings,
        "position_embeddings":
            position_embeddings,
        "embedding_gamma": embedding_gamma,
        "embedding_beta": embedding_beta,
    }


def _make_encoder_metadata():
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

    metadata = []

    for logical_name, param in encoder.named_parameters():
        if logical_name.endswith("mha.cu_length"):
            continue

        tensor = param.tensor()

        shape = [
            int(dim._attrs["values"][0])
            for dim in tensor._attrs["shape"]
        ]

        metadata.append(
            {
                "logical_name": logical_name,
                "ait_name": tensor._attrs["name"],
                "shape": shape,
            }
        )

    return metadata


def _init_param(
    logical_name,
    shape,
):
    if (
        "ln1.weight" in logical_name
        or "ln2.weight" in logical_name
    ):
        return torch.ones(
            shape,
            dtype=torch.float32,
        )

    if (
        "ln1.bias" in logical_name
        or "ln2.bias" in logical_name
    ):
        return torch.zeros(
            shape,
            dtype=torch.float32,
        )

    value = torch.empty(
        shape,
        dtype=torch.float32,
    )

    value.normal_(
        0.0,
        0.02,
    )

    return value


def _load_prepack_manifest():
    if not os.path.exists(
        PREPACK_MANIFEST_PATH
    ):
        raise RuntimeError(
            f"{PREPACK_MANIFEST_PATH} not found. "
            "Re-run benchmark_cpu_bert_full.py "
            "with the updated backend first."
        )

    with open(
        PREPACK_MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    pairs = manifest.get(
        "pairs",
        [],
    )

    expected_pairs = NUM_LAYERS * 4

    if len(pairs) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} static FC "
            f"pairs in manifest, got {len(pairs)}"
        )

    expected_raw_mib = 324.31640625
    raw_mib = (
        int(manifest["total_raw_bytes"])
        / 1024.0
        / 1024.0
    )

    if abs(raw_mib - expected_raw_mib) > 0.01:
        raise RuntimeError(
            "Unexpected static-FC raw size in "
            f"manifest: {raw_mib} MiB"
        )

    return manifest


def create_ait_data():
    """Create only constants that must remain resident.

    Static dense FC weights are deliberately NOT allocated here.
    They are created one pair at a time in run_ait(), prepacked,
    detached from the runtime, and immediately released.
    """

    torch.manual_seed(0)

    data = _make_common_data()
    metadata = _make_encoder_metadata()
    manifest = _load_prepack_manifest()

    packed_names = set()

    for pair in manifest["pairs"]:
        packed_names.add(
            pair["weight_name"]
        )
        packed_names.add(
            pair["bias_name"]
        )

    ait_params = {}

    for item in metadata:
        ait_name = item["ait_name"]

        if ait_name in packed_names:
            continue

        value = _init_param(
            item["logical_name"],
            item["shape"],
        )

        ait_params[ait_name] = value

    data["ait_params"] = ait_params
    data["manifest"] = manifest

    return data


def create_torch_data():
    torch.manual_seed(0)

    data = _make_common_data()
    metadata = _make_encoder_metadata()

    logical_params = {}

    for item in metadata:
        logical_params[
            item["logical_name"]
        ] = _init_param(
            item["logical_name"],
            item["shape"],
        )

    data["logical_params"] = (
        logical_params
    )

    return data


def _prepack_static_fc_constants(
    module,
    manifest,
):
    """Incrementally materialize, prepack, and release every dense FC pair."""

    raw_dll = module.DLL.DLL

    function_cache = {}

    def get_prepack_function(
        symbol,
    ):
        fn = function_cache.get(
            symbol
        )

        if fn is not None:
            return fn

        try:
            fn = getattr(
                raw_dll,
                symbol,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Missing exported static-FC prepack "
                f"symbol: {symbol}"
            ) from exc

        fn.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]

        fn.restype = ctypes.c_int

        function_cache[
            symbol
        ] = fn

        return fn

    total_raw_bytes = 0

    print()
    print(
        "===== Incremental static-FC prepack ====="
    )

    memory_before = get_memory()

    print(
        "RSS before prepack MiB:",
        memory_before["vmrss"],
    )
    print(
        "Peak before prepack MiB:",
        memory_before["peak_rss"],
    )

    for index, pair in enumerate(
        manifest["pairs"],
        start=1,
    ):
        weight = _init_param(
            pair["logical_base"]
            + ".weight",
            pair["weight_shape"],
        )

        bias = _init_param(
            pair["logical_base"]
            + ".bias",
            pair["bias_shape"],
        )

        weight_ait = torch_to_ait_data(
            weight
        )

        bias_ait = torch_to_ait_data(
            bias
        )

        module.set_constant(
            pair["weight_name"],
            weight_ait,
        )

        module.set_constant(
            pair["bias_name"],
            bias_ait,
        )

        prepack = get_prepack_function(
            pair["prepack_symbol"]
        )

        status = prepack(
            ctypes.c_uint64(
                int(pair["cache_id"])
            ),
            ctypes.c_void_p(
                weight.data_ptr()
            ),
            ctypes.c_void_p(
                bias.data_ptr()
            ),
            ctypes.c_size_t(
                int(pair["n"])
            ),
            ctypes.c_size_t(
                int(pair["k"])
            ),
        )

        if status != 1:
            raise RuntimeError(
                "Static-FC prepack rejected "
                f"{pair['logical_base']} via "
                f"{pair['prepack_symbol']}"
            )

        # The generated kernel will now find this packed operator
        # by cache_id, not by raw weight pointer.  Clear the model's
        # raw pointers before freeing the Python tensors.
        module.set_many_constants(
            {
                pair["weight_name"]:
                    AITData(
                        0,
                        weight_ait.shape,
                        weight_ait.dtype,
                    ),
                pair["bias_name"]:
                    AITData(
                        0,
                        bias_ait.shape,
                        bias_ait.dtype,
                    ),
            }
        )

        pair_bytes = (
            weight.numel()
            * weight.element_size()
            + bias.numel()
            * bias.element_size()
        )

        total_raw_bytes += pair_bytes

        del weight
        del bias
        del weight_ait
        del bias_ait

        _trim_cpu_allocator()

        if (
            index == 1
            or index % 12 == 0
            or index == len(
                manifest["pairs"]
            )
        ):
            memory = get_memory()

            print(
                f"[{index:02d}/"
                f"{len(manifest['pairs']):02d}] "
                f"{pair['logical_base']} "
                f"RSS={memory['vmrss']:.3f} MiB "
                f"Peak={memory['peak_rss']:.3f} MiB"
            )

    expected_bytes = int(
        manifest["total_raw_bytes"]
    )

    if total_raw_bytes != expected_bytes:
        raise RuntimeError(
            "Incremental prepack raw-byte count "
            f"mismatch: got {total_raw_bytes}, "
            f"expected {expected_bytes}"
        )

    memory_after = get_memory()

    print(
        "Prepacked raw MiB:",
        total_raw_bytes
        / 1024.0
        / 1024.0,
    )
    print(
        "RSS after prepack MiB:",
        memory_after["vmrss"],
    )
    print(
        "Peak after prepack MiB:",
        memory_after["peak_rss"],
    )

    return total_raw_bytes


def run_ait(
    data,
):
    if not os.path.exists(
        SO_PATH
    ):
        raise RuntimeError(
            f"{SO_PATH} not found. "
            "Run benchmark_cpu_bert_full.py first."
        )

    module = Model(
        SO_PATH
    )

    ait_inputs = {
        "input_ids":
            torch_to_ait_data(
                data["input_ids"]
            ),
        "token_type_ids":
            torch_to_ait_data(
                data["token_type_ids"]
            ),
        "position_ids":
            torch_to_ait_data(
                data["position_ids"]
            ),
    }

    # Constants that remain live for inference:
    # embeddings and layernorm parameters.
    ait_constants = {
        "word_embeddings":
            torch_to_ait_data(
                data["word_embeddings"]
            ),
        "token_type_embeddings":
            torch_to_ait_data(
                data["token_type_embeddings"]
            ),
        "position_embeddings":
            torch_to_ait_data(
                data["position_embeddings"]
            ),
        "embedding_gamma":
            torch_to_ait_data(
                data["embedding_gamma"]
            ),
        "embedding_beta":
            torch_to_ait_data(
                data["embedding_beta"]
            ),
    }

    for name, value in (
        data["ait_params"].items()
    ):
        ait_constants[name] = (
            torch_to_ait_data(
                value
            )
        )

    module.set_many_constants(
        ait_constants
    )

    _prepack_static_fc_constants(
        module,
        data["manifest"],
    )

    output = torch.empty(
        BATCH,
        SEQ,
        HIDDEN,
        dtype=torch.float32,
    )

    ait_outputs = {
        "output":
            torch_to_ait_data(
                output
            )
    }

    def forward():
        module.run(
            ait_inputs,
            ait_outputs,
        )

    memory_before_inference = (
        get_memory()
    )

    print()
    print(
        "RSS before first inference MiB:",
        memory_before_inference["vmrss"],
    )
    print(
        "Peak before first inference MiB:",
        memory_before_inference[
            "peak_rss"
        ],
    )

    # The static FCs are already packed. Warmup now exercises
    # only the real inference path.
    for _ in range(5):
        forward()

    reference = output.clone()

    forward()

    max_repeat_diff = (
        output - reference
    ).abs().max().item()

    if max_repeat_diff != 0.0:
        raise RuntimeError(
            "Repeated inference changed output "
            f"after raw-weight release: "
            f"{max_repeat_diff}"
        )

    stats = benchmark(
        forward,
        warmup=0,
        iterations=20,
    )

    memory = get_memory()

    print(
        "Repeat max abs diff:",
        max_repeat_diff,
    )

    return stats, memory


def run_torch(
    data,
):
    @torch.no_grad()
    def forward():
        x = (
            F.embedding(
                data["input_ids"],
                data["word_embeddings"],
            )
            + F.embedding(
                data["token_type_ids"],
                data[
                    "token_type_embeddings"
                ],
            )
            + F.embedding(
                data["position_ids"],
                data[
                    "position_embeddings"
                ],
            )
        )

        x = F.layer_norm(
            x,
            (HIDDEN,),
            data["embedding_gamma"],
            data["embedding_beta"],
            EPS,
        )

        for layer_idx in range(
            NUM_LAYERS
        ):
            x = pytorch_encoder_layer(
                x,
                data["logical_params"],
                "layers."
                + str(layer_idx)
                + ".",
                BATCH,
                SEQ,
                HIDDEN,
                HEADS,
                HEAD_DIM,
                EPS,
            )

        return x

    stats = benchmark(
        forward
    )

    memory = get_memory()

    return stats, memory


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: benchmark_cpu_bert_isolated.py "
            "[ait|torch]"
        )

    mode = sys.argv[1]

    print(
        "===== Isolated benchmark ====="
    )
    print("mode       :", mode)
    print("pinned CPU :", cpu)
    print(
        "threads    :",
        torch.get_num_threads(),
    )

    if mode in {
        "ait",
        "ait-release",
    }:
        # ait-release is kept as a compatibility alias.
        data = create_ait_data()
        stats, memory = run_ait(
            data
        )

    elif mode == "torch":
        data = create_torch_data()
        stats, memory = run_torch(
            data
        )

    else:
        raise SystemExit(
            "mode must be 'ait' or 'torch'"
        )

    print()
    print("===== Latency =====")
    print(
        "median ms:",
        stats["median"],
    )
    print(
        "mean ms  :",
        stats["mean"],
    )
    print(
        "min ms   :",
        stats["min"],
    )
    print(
        "max ms   :",
        stats["max"],
    )

    print()
    print("===== Memory =====")
    print(
        "Peak RSS MiB:",
        memory["peak_rss"],
    )
    print(
        "VmHWM MiB   :",
        memory["vmhwm"],
    )
    print(
        "VmRSS MiB   :",
        memory["vmrss"],
    )


if __name__ == "__main__":
    main()
