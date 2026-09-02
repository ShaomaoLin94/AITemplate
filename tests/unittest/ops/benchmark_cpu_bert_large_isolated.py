import ctypes
import gc
import hashlib
import importlib.util
import json
import math
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
HIDDEN = 1024
HEADS = 16
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 4096
NUM_LAYERS = 24
VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2
EPS = 1e-12

SO_PATH = "./tmp/benchmark_cpu_bert_large_full/test.so"
MANIFEST_PATH = (
    "./tmp/benchmark_cpu_bert_large_full/"
    "static_fc_prepack_manifest.json"
)

PACKED_GEMM_PARAM_SUFFIXES = (
    "mha.qkv.weight",
    "mha.qkv.bias",
    "mha.proj.weight",
    "mha.proj.bias",
    "ffn1.weight",
    "ffn1.bias",
    "ffn2.weight",
    "ffn2.bias",
)


def pin_single_cpu():
    cpus = sorted(os.sched_getaffinity(0))
    if not cpus:
        raise RuntimeError("No CPU available in affinity mask")
    cpu = cpus[0]
    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return cpu


def get_memory():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = None
    hwm = None

    with open("/proc/self/status", "r", encoding="utf-8") as f:
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


def trim_cpu_allocator():
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def benchmark(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return {
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }


def _seed_for(name):
    digest = hashlib.blake2b(
        name.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _randn_for(name, shape, scale=0.02):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_for(name))
    return torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
    ) * scale


def _param_value(logical_name, shape):
    if logical_name.endswith("ln1.weight") or logical_name.endswith("ln2.weight"):
        return torch.ones(shape, dtype=torch.float32)

    if logical_name.endswith("ln1.bias") or logical_name.endswith("ln2.bias"):
        return torch.zeros(shape, dtype=torch.float32)

    return _randn_for(logical_name, shape)


def _is_packed_param(logical_name):
    return logical_name.endswith(PACKED_GEMM_PARAM_SUFFIXES)


def build_param_meta():
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

    meta = {}
    for logical_name, param in encoder.named_parameters():
        if logical_name.endswith("mha.cu_length"):
            continue

        tensor = param.tensor()
        shape = [
            int(dim._attrs["values"][0])
            for dim in tensor._attrs["shape"]
        ]
        meta[logical_name] = {
            "ait_name": tensor._attrs["name"],
            "shape": shape,
        }

    return meta


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise RuntimeError(
            f"{MANIFEST_PATH} not found. "
            "Run benchmark_cpu_bert_large_full.py first."
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected = NUM_LAYERS * 4
    if len(manifest.get("pairs", [])) != expected:
        raise RuntimeError(
            f"Expected {expected} static FC pairs, "
            f"got {len(manifest.get('pairs', []))}"
        )

    return manifest


def create_common_data():
    input_gen = torch.Generator(device="cpu")
    input_gen.manual_seed(0)

    return {
        "input_ids": torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH, SEQ),
            generator=input_gen,
            dtype=torch.int64,
        ),
        "token_type_ids": torch.zeros(
            BATCH,
            SEQ,
            dtype=torch.int64,
        ),
        "position_ids": torch.arange(
            SEQ,
            dtype=torch.int64,
        ).reshape(1, SEQ).expand(BATCH, -1).contiguous(),
        "word_embeddings": _randn_for(
            "embeddings.word",
            [VOCAB_SIZE, HIDDEN],
        ),
        "token_type_embeddings": _randn_for(
            "embeddings.token_type",
            [TYPE_VOCAB_SIZE, HIDDEN],
        ),
        "position_embeddings": _randn_for(
            "embeddings.position",
            [MAX_POSITION_EMBEDDINGS, HIDDEN],
        ),
        "embedding_gamma": torch.ones(
            HIDDEN,
            dtype=torch.float32,
        ),
        "embedding_beta": torch.zeros(
            HIDDEN,
            dtype=torch.float32,
        ),
    }


def build_all_torch_params(meta):
    return {
        logical_name: _param_value(
            logical_name,
            item["shape"],
        )
        for logical_name, item in meta.items()
    }


def torch_forward(data, logical_params):
    x = (
        F.embedding(data["input_ids"], data["word_embeddings"])
        + F.embedding(
            data["token_type_ids"],
            data["token_type_embeddings"],
        )
        + F.embedding(
            data["position_ids"],
            data["position_embeddings"],
        )
    )

    x = F.layer_norm(
        x,
        (HIDDEN,),
        data["embedding_gamma"],
        data["embedding_beta"],
        EPS,
    )

    for layer_idx in range(NUM_LAYERS):
        x = pytorch_encoder_layer(
            x,
            logical_params,
            f"layers.{layer_idx}.",
            BATCH,
            SEQ,
            HIDDEN,
            HEADS,
            HEAD_DIM,
            EPS,
        )

    return x


def _call_prepack(module, pair, weight, bias):
    fn = getattr(module.DLL.DLL, pair["prepack_symbol"])
    fn.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int

    ok = fn(
        ctypes.c_uint64(pair["cache_id"]),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_size_t(pair["n"]),
        ctypes.c_size_t(pair["k"]),
    )

    if ok != 1:
        raise RuntimeError(
            "Static FC prepack failed for "
            f"{pair['logical_base']} via {pair['prepack_symbol']}"
        )


def _null_constant(shape):
    return AITData(
        0,
        list(shape),
        "float32",
    )


def prepare_ait(data, meta, manifest, verbose=True):
    if not os.path.exists(SO_PATH):
        raise RuntimeError(
            f"{SO_PATH} not found. "
            "Run benchmark_cpu_bert_large_full.py first."
        )

    module = Model(SO_PATH)

    ait_inputs = {
        "input_ids": torch_to_ait_data(data["input_ids"]),
        "token_type_ids": torch_to_ait_data(data["token_type_ids"]),
        "position_ids": torch_to_ait_data(data["position_ids"]),
    }

    # Keep these tensors alive for the entire model lifetime because they are
    # normal constants rather than XNNPACK-packed FC constants.
    keepalive = {
        "word_embeddings": data["word_embeddings"],
        "token_type_embeddings": data["token_type_embeddings"],
        "position_embeddings": data["position_embeddings"],
        "embedding_gamma": data["embedding_gamma"],
        "embedding_beta": data["embedding_beta"],
    }

    non_static_constants = {
        name: torch_to_ait_data(value)
        for name, value in keepalive.items()
    }

    for logical_name, item in meta.items():
        if _is_packed_param(logical_name):
            continue

        value = _param_value(logical_name, item["shape"])
        keepalive[item["ait_name"]] = value
        non_static_constants[item["ait_name"]] = torch_to_ait_data(value)

    module.set_many_constants(non_static_constants)
    del non_static_constants
    trim_cpu_allocator()

    before_prepack = get_memory()

    if verbose:
        print()
        print("===== Incremental static-FC prepack =====")
        print("pairs                  :", len(manifest["pairs"]))
        print(
            "raw FC total MiB       :",
            manifest["total_raw_bytes"] / 1024.0 / 1024.0,
        )
        print("RSS before prepack MiB :", before_prepack["vmrss"])

    max_pair_raw = 0

    # True incremental prepack: at most one raw weight+bias pair is alive.
    for idx, pair in enumerate(manifest["pairs"], start=1):
        base = pair["logical_base"]
        weight_logical = base + ".weight"
        bias_logical = base + ".bias"

        weight = _param_value(
            weight_logical,
            pair["weight_shape"],
        )
        bias = _param_value(
            bias_logical,
            pair["bias_shape"],
        )

        raw_bytes = (
            weight.numel() * weight.element_size()
            + bias.numel() * bias.element_size()
        )
        max_pair_raw = max(max_pair_raw, raw_bytes)

        _call_prepack(module, pair, weight, bias)

        # Mark both runtime constants as set, but leave their pointers null.
        # The generated wrappers resolve the already packed operator by
        # compile-time cache_id and therefore no longer need raw storage.
        module.set_many_constants(
            {
                pair["weight_name"]: _null_constant(pair["weight_shape"]),
                pair["bias_name"]: _null_constant(pair["bias_shape"]),
            }
        )

        del weight
        del bias
        trim_cpu_allocator()

        if verbose and (idx == 1 or idx % 8 == 0 or idx == len(manifest["pairs"])):
            mem = get_memory()
            print(
                f"[{idx:02d}/{len(manifest['pairs'])}] "
                f"{base:<28} RSS={mem['vmrss']:.1f} MiB  "
                f"peak={mem['peak_rss']:.1f} MiB"
            )

    after_prepack = get_memory()

    output = torch.empty(
        BATCH,
        SEQ,
        HIDDEN,
        dtype=torch.float32,
    )
    ait_outputs = {
        "output": torch_to_ait_data(output),
    }

    if verbose:
        print("max raw pair MiB        :", max_pair_raw / 1024.0 / 1024.0)
        print("RSS after prepack MiB   :", after_prepack["vmrss"])
        print("Peak after prepack MiB  :", after_prepack["peak_rss"])

    return {
        "module": module,
        "inputs": ait_inputs,
        "outputs": ait_outputs,
        "output": output,
        "keepalive": keepalive,
        "memory_before_prepack": before_prepack,
        "memory_after_prepack": after_prepack,
    }


def run_ait():
    meta = build_param_meta()
    manifest = load_manifest()
    data = create_common_data()
    prepared = prepare_ait(data, meta, manifest, verbose=True)

    module = prepared["module"]
    ait_inputs = prepared["inputs"]
    ait_outputs = prepared["outputs"]

    def forward():
        module.run(ait_inputs, ait_outputs)

    for _ in range(5):
        forward()

    first_run_memory = get_memory()
    stats = benchmark(forward, warmup=0, iterations=20)
    memory = get_memory()

    print()
    print("===== AITemplate BERT-large =====")
    print("median ms:", stats["median"])
    print("mean ms  :", stats["mean"])
    print("min ms   :", stats["min"])
    print("max ms   :", stats["max"])
    print("VmRSS MiB:", memory["vmrss"])
    print("Peak MiB :", memory["peak_rss"])
    print("RSS after first inference MiB:", first_run_memory["vmrss"])

    return stats, memory


def run_torch():
    meta = build_param_meta()
    data = create_common_data()
    logical_params = build_all_torch_params(meta)

    @torch.no_grad()
    def forward():
        return torch_forward(data, logical_params)

    stats = benchmark(forward, warmup=5, iterations=20)
    memory = get_memory()

    print()
    print("===== PyTorch BERT-large reference =====")
    print("median ms:", stats["median"])
    print("mean ms  :", stats["mean"])
    print("min ms   :", stats["min"])
    print("max ms   :", stats["max"])
    print("VmRSS MiB:", memory["vmrss"])
    print("Peak MiB :", memory["peak_rss"])

    return stats, memory


def run_numerical():
    meta = build_param_meta()
    manifest = load_manifest()
    data = create_common_data()

    print("===== Numerical: PyTorch reference =====")
    logical_params = build_all_torch_params(meta)

    with torch.no_grad():
        reference = torch_forward(data, logical_params).clone()

    del logical_params
    trim_cpu_allocator()

    print("===== Numerical: AITemplate =====")
    prepared = prepare_ait(data, meta, manifest, verbose=False)
    prepared["module"].run(prepared["inputs"], prepared["outputs"])
    actual = prepared["output"]

    diff = (actual - reference).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()

    finite = bool(torch.isfinite(actual).all())

    print()
    print("===== Numerical error =====")
    print("finite    :", finite)
    print("max error :", max_error)
    print("mean error:", mean_error)
    print("rmse      :", rmse)

    torch.testing.assert_close(
        actual,
        reference,
        rtol=1e-4,
        atol=1e-4,
    )

    print("PASS: AITemplate BERT-large matches PyTorch")


def _result_json(mode, stats, memory):
    print(
        "RESULT_JSON:",
        json.dumps(
            {
                "mode": mode,
                "median_ms": stats["median"],
                "mean_ms": stats["mean"],
                "min_ms": stats["min"],
                "max_ms": stats["max"],
                "vmrss_mib": memory["vmrss"],
                "peak_rss_mib": memory["peak_rss"],
            },
            sort_keys=True,
        ),
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: benchmark_cpu_bert_large_isolated.py "
            "[numerical|ait|torch]"
        )

    mode = sys.argv[1]
    cpu = pin_single_cpu()

    print("===== BERT-large isolated benchmark =====")
    print("mode         :", mode)
    print("pinned CPU   :", cpu)
    print("threads      :", torch.get_num_threads())
    print("layers       :", NUM_LAYERS)
    print("batch        :", BATCH)
    print("sequence     :", SEQ)
    print("hidden       :", HIDDEN)
    print("heads        :", HEADS)
    print("head dim     :", HEAD_DIM)
    print("intermediate :", INTERMEDIATE)

    if mode == "numerical":
        run_numerical()
        return

    if mode == "ait":
        stats, memory = run_ait()
        _result_json("ait", stats, memory)
        return

    if mode == "torch":
        stats, memory = run_torch()
        _result_json("torch", stats, memory)
        return

    raise SystemExit("mode must be numerical, ait, or torch")


if __name__ == "__main__":
    main()
