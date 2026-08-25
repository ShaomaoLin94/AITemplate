import ctypes
import gc
import importlib.util
import mmap
import os
import re
import resource
import statistics
import subprocess
import time

import torch

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import torch_to_ait_data
from aitemplate.frontend import Tensor
from aitemplate.frontend.nn.attention import MultiheadAttention
from aitemplate.frontend.nn.linear import Linear


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

allowed = sorted(os.sched_getaffinity(0))
cpu = allowed[0]
os.sched_setaffinity(0, {cpu})


spec = importlib.util.spec_from_file_location(
    "cpu_bert_full_test",
    "tests/unittest/ops/test_cpu_bert_full_numerical.py",
)
full_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_test)

CPUEncoder = full_test.CPUEncoder


BATCH = 1
SEQ = 128
HIDDEN = 768
HEADS = 12
INTERMEDIATE = 3072
NUM_LAYERS = 12

VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2

EPS = 1e-12

WORKDIR = "./tmp"
MODEL_NAME = "probe_cpu_bert_incremental_prepack"


def current_rss_mib():
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return None


def peak_rss_mib():
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0
    )


def benchmark(fn, warmup=3, iterations=10):
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


def build_graph():
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

    output = encoder(x)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True

    return output, encoder


def is_static_fc_param(logical_name):
    return (
        ".mha.proj." in logical_name
        or ".ffn1." in logical_name
        or ".ffn2." in logical_name
    )


def allocate_mmap_tensor(shape):
    numel = 1

    for dim in shape:
        numel *= int(dim)

    nbytes = numel * 4

    mm = mmap.mmap(
        -1,
        nbytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )

    tensor = torch.frombuffer(
        mm,
        dtype=torch.float32,
        count=numel,
    ).reshape(shape)

    return mm, tensor


def tensor_shape(param):
    return [
        int(dim._attrs["values"][0])
        for dim in param.tensor()._attrs["shape"]
    ]


def init_param(logical_name, value):
    if (
        "ln1.weight" in logical_name
        or "ln2.weight" in logical_name
    ):
        value.fill_(1.0)

    elif (
        "ln1.bias" in logical_name
        or "ln2.bias" in logical_name
    ):
        value.zero_()

    else:
        value.normal_(0.0, 0.02)


def discover_prepack_functions(module):
    output = subprocess.check_output(
        ["nm", "-D", module.lib_path],
        text=True,
    )

    names = sorted(
        set(
            re.findall(
                r"\b([A-Za-z0-9_]+_prepack)$",
                output,
                flags=re.MULTILINE,
            )
        )
    )

    if not names:
        raise RuntimeError(
            "No *_prepack symbols were exported from the model .so"
        )

    raw_dll = module.DLL.DLL
    funcs = {}

    for name in names:
        fn = getattr(raw_dll, name)

        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]

        fn.restype = ctypes.c_int

        funcs[name] = fn

    return funcs


def prepack_pair(
    funcs,
    symbol,
    weight,
    bias,
    n,
    k,
):
    if symbol not in funcs:
        raise RuntimeError(
            "Expected prepack symbol was not exported: "
            f"{symbol}; available={sorted(funcs)}"
        )

    fn = funcs[symbol]

    result = fn(
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_size_t(n),
        ctypes.c_size_t(k),
    )

    if result != 1:
        raise RuntimeError(
            f"{symbol} rejected its mapped parameter "
            f"shape N={n}, K={k}"
        )

    return symbol


def make_runtime_inputs():
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

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def set_nonstatic_constants(module, encoder):
    keepalive = {}

    compiled_tensors = {
        tensor._attrs["name"]: tensor
        for tensor in module.debug_sorted_graph
    }

    embeddings = {
        "word_embeddings":
            torch.randn(
                VOCAB_SIZE,
                HIDDEN,
                dtype=torch.float32,
            ) * 0.02,
        "token_type_embeddings":
            torch.randn(
                TYPE_VOCAB_SIZE,
                HIDDEN,
                dtype=torch.float32,
            ) * 0.02,
        "position_embeddings":
            torch.randn(
                MAX_POSITION_EMBEDDINGS,
                HIDDEN,
                dtype=torch.float32,
            ) * 0.02,
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

    keepalive.update(embeddings)

    for name, value in embeddings.items():
        module.set_constant(
            name,
            torch_to_ait_data(value),
        )

    static_groups = {}

    for logical_name, param in encoder.named_parameters():
        if logical_name.endswith("mha.cu_length"):
            continue

        ait_name = param.tensor()._attrs["name"]
        shape = tensor_shape(param)

        if is_static_fc_param(logical_name):
            if ait_name not in compiled_tensors:
                raise RuntimeError(
                    f"Compiled graph is missing constant tensor {ait_name}"
                )

            tensor_desc = compiled_tensors[ait_name]

            fc_dst_ops = [
                op
                for op in tensor_desc.dst_ops()
                if op._attrs.get("op")
                in {
                    "gemm_rcr_bias_add",
                    "gemm_rcr_bias_fast_gelu",
                }
            ]

            if len(fc_dst_ops) != 1:
                raise RuntimeError(
                    "Expected exactly one static-FC destination op "
                    f"for {logical_name}, got "
                    f"{[(op._attrs.get('op'), op._attrs.get('name')) for op in fc_dst_ops]}"
                )

            prepack_symbol = (
                fc_dst_ops[0]._attrs["name"]
                + "_prepack"
            )

            if logical_name.endswith(".weight"):
                base = logical_name[:-len(".weight")]
                kind = "weight"
            elif logical_name.endswith(".bias"):
                base = logical_name[:-len(".bias")]
                kind = "bias"
            else:
                raise RuntimeError(
                    f"Unexpected static FC parameter: {logical_name}"
                )

            static_groups.setdefault(base, {})[kind] = {
                "logical_name": logical_name,
                "ait_name": ait_name,
                "shape": shape,
                "prepack_symbol": prepack_symbol,
            }

            continue

        value = torch.empty(
            shape,
            dtype=torch.float32,
        )

        init_param(
            logical_name,
            value,
        )

        module.set_constant(
            ait_name,
            torch_to_ait_data(value),
        )

        keepalive[ait_name] = value

    return keepalive, static_groups


def incremental_prepack(
    module,
    funcs,
    static_groups,
):
    mappings = []
    total_raw_bytes = 0

    print()
    print("===== Incremental static-FC prepack =====")
    print(
        "RSS before static FC MiB :",
        current_rss_mib(),
    )
    print(
        "Peak before static FC MiB:",
        peak_rss_mib(),
    )

    for index, base in enumerate(
        sorted(static_groups),
        start=1,
    ):
        group = static_groups[base]

        if set(group) != {"weight", "bias"}:
            raise RuntimeError(
                f"Incomplete static FC pair for {base}: {group.keys()}"
            )

        weight_info = group["weight"]
        bias_info = group["bias"]

        if (
            weight_info["prepack_symbol"]
            != bias_info["prepack_symbol"]
        ):
            raise RuntimeError(
                f"Weight/bias mapped to different functions for {base}: "
                f"{weight_info['prepack_symbol']} vs "
                f"{bias_info['prepack_symbol']}"
            )

        mapped_symbol = weight_info["prepack_symbol"]

        weight_mm, weight = allocate_mmap_tensor(
            weight_info["shape"]
        )

        bias_mm, bias = allocate_mmap_tensor(
            bias_info["shape"]
        )

        init_param(
            weight_info["logical_name"],
            weight,
        )

        init_param(
            bias_info["logical_name"],
            bias,
        )

        module.set_constant(
            weight_info["ait_name"],
            torch_to_ait_data(weight),
        )

        module.set_constant(
            bias_info["ait_name"],
            torch_to_ait_data(bias),
        )

        n = int(weight_info["shape"][0])
        k = int(weight_info["shape"][1])

        raw_bytes = (
            weight.numel() * weight.element_size()
            + bias.numel() * bias.element_size()
        )

        total_raw_bytes += raw_bytes

        symbol = prepack_pair(
            funcs,
            mapped_symbol,
            weight,
            bias,
            n,
            k,
        )

        rss_after_pack = current_rss_mib()
        peak_after_pack = peak_rss_mib()

        # Keep the virtual mapping and pointer value alive, but
        # drop the resident raw pages after XNNPACK has packed them.
        del weight
        del bias
        gc.collect()

        weight_mm.madvise(
            mmap.MADV_DONTNEED
        )

        bias_mm.madvise(
            mmap.MADV_DONTNEED
        )

        mappings.append(weight_mm)
        mappings.append(bias_mm)

        rss_after_release = current_rss_mib()

        print(
            f"[{index:02d}/{len(static_groups):02d}] "
            f"{base}"
        )
        print(
            f"  shape N,K          : {n}, {k}"
        )
        print(
            f"  prepack symbol     : {symbol}"
        )
        print(
            f"  raw pair MiB       : "
            f"{raw_bytes / 1024.0 / 1024.0:.3f}"
        )
        print(
            f"  RSS after pack MiB : {rss_after_pack}"
        )
        print(
            f"  RSS after drop MiB : {rss_after_release}"
        )
        print(
            f"  Peak so far MiB    : {peak_after_pack}"
        )

    return mappings, total_raw_bytes


def main():
    if not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError(
            "mmap.MADV_DONTNEED is unavailable"
        )

    torch.manual_seed(0)

    print("===== Compile incremental-prepack probe =====")

    output_tensor, encoder = build_graph()

    module = compile_model(
        output_tensor,
        CPU(),
        WORKDIR,
        MODEL_NAME,
    )

    input_map = module.get_input_name_to_index_map()

    print()
    print("runtime input map:")
    print(input_map)

    expected_inputs = {
        "input_ids",
        "token_type_ids",
        "position_ids",
    }

    if set(input_map) != expected_inputs:
        raise RuntimeError(
            f"Unexpected runtime inputs: {input_map}"
        )

    funcs = discover_prepack_functions(
        module
    )

    print()
    print("exported prepack symbols:")
    for name in sorted(funcs):
        print(" ", name)

    keepalive, static_groups = set_nonstatic_constants(
        module,
        encoder,
    )

    print()
    print(
        "static FC pairs:",
        len(static_groups),
    )

    print()
    print("static FC compiler mapping:")
    for base in sorted(static_groups):
        group = static_groups[base]
        weight_symbol = group["weight"]["prepack_symbol"]
        bias_symbol = group["bias"]["prepack_symbol"]

        if weight_symbol != bias_symbol:
            raise RuntimeError(
                f"Inconsistent compiler mapping for {base}: "
                f"{weight_symbol} vs {bias_symbol}"
            )

        print(
            f"  {base} -> {weight_symbol}"
        )

    print(
        "RSS after non-static constants MiB:",
        current_rss_mib(),
    )
    print(
        "Peak after non-static constants MiB:",
        peak_rss_mib(),
    )

    mappings, total_raw_bytes = incremental_prepack(
        module,
        funcs,
        static_groups,
    )

    print()
    print("===== After incremental prepack =====")
    print(
        "total static raw MiB:",
        total_raw_bytes / 1024.0 / 1024.0,
    )
    print(
        "RSS before first inference MiB:",
        current_rss_mib(),
    )
    print(
        "Peak before first inference MiB:",
        peak_rss_mib(),
    )

    runtime_inputs = make_runtime_inputs()

    ait_inputs = {
        name: torch_to_ait_data(value)
        for name, value in runtime_inputs.items()
    }

    output = torch.empty(
        BATCH,
        SEQ,
        HIDDEN,
        dtype=torch.float32,
    )

    ait_outputs = {
        "output": torch_to_ait_data(output),
    }

    # Raw static FC pages are already gone here.
    module.run(
        ait_inputs,
        ait_outputs,
    )

    first_output = output.clone()

    module.run(
        ait_inputs,
        ait_outputs,
    )

    repeat_max_abs_diff = (
        output - first_output
    ).abs().max().item()

    print()
    print("===== First inference after prepack =====")
    print(
        "output finite:",
        bool(torch.isfinite(output).all().item()),
    )
    print(
        "repeat max abs diff:",
        repeat_max_abs_diff,
    )
    print(
        "RSS after inference MiB:",
        current_rss_mib(),
    )
    print(
        "Peak RSS MiB:",
        peak_rss_mib(),
    )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Output contains non-finite values"
        )

    if repeat_max_abs_diff != 0.0:
        raise RuntimeError(
            "Repeated inference changed output unexpectedly"
        )

    stats = benchmark(
        lambda: module.run(
            ait_inputs,
            ait_outputs,
        )
    )

    print()
    print("===== Post-prepack latency =====")
    print("median ms:", stats["median"])
    print("mean ms  :", stats["mean"])
    print("min ms   :", stats["min"])
    print("max ms   :", stats["max"])

    print()
    print("===== Final memory =====")
    print(
        "Current RSS MiB:",
        current_rss_mib(),
    )
    print(
        "Peak RSS MiB   :",
        peak_rss_mib(),
    )

    # Keep mappings and ordinary constants alive until all runs finish.
    _ = mappings
    _ = keepalive


if __name__ == "__main__":
    main()

