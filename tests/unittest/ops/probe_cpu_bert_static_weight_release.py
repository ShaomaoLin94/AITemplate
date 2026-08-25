import gc
import importlib.util
import mmap
import os
import resource
import statistics
import time

import torch

from aitemplate.backend.cpu import CPU
from aitemplate.compiler import compile_model, ops
from aitemplate.compiler.model import Model, torch_to_ait_data
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
MODEL_NAME = "probe_cpu_bert_static_weight_release"


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

    # These are deliberately constants: no is_input=True.
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

    # Do NOT mark encoder params is_input=True.
    # AITemplate will treat them as unbound constants.
    output = encoder(x)
    output._attrs["name"] = "output"
    output._attrs["is_output"] = True

    return output, encoder


def create_data(encoder):
    torch.manual_seed(0)

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

    constants = {
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

    mmap_regions = []
    static_raw_bytes = 0

    for logical_name, param in encoder.named_parameters():
        if logical_name.endswith("mha.cu_length"):
            continue

        tensor_desc = param.tensor()
        ait_name = tensor_desc._attrs["name"]

        shape = [
            dim._attrs["values"][0]
            for dim in tensor_desc._attrs["shape"]
        ]

        if is_static_fc_param(logical_name):
            mm, value = allocate_mmap_tensor(shape)

            mmap_regions.append(
                {
                    "logical_name": logical_name,
                    "ait_name": ait_name,
                    "mmap": mm,
                    "tensor": value,
                }
            )

            static_raw_bytes += (
                value.numel() * value.element_size()
            )
        else:
            value = torch.empty(
                shape,
                dtype=torch.float32,
            )

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

        constants[ait_name] = value

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
        "constants": constants,
        "mmap_regions": mmap_regions,
        "static_raw_bytes": static_raw_bytes,
    }


def release_static_fc_pages(data):
    if not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError(
            "mmap.MADV_DONTNEED is unavailable"
        )

    released = 0

    # Remove references from constants dict first.
    for region in data["mmap_regions"]:
        ait_name = region["ait_name"]
        data["constants"].pop(ait_name, None)

    gc.collect()

    for region in data["mmap_regions"]:
        mm = region["mmap"]

        mm.madvise(
            mmap.MADV_DONTNEED,
        )

        released += len(mm)

    return released


def main():
    print("===== Compile dedicated constant-model probe =====")

    output_tensor, encoder = build_graph()

    module = compile_model(
        output_tensor,
        CPU(),
        WORKDIR,
        MODEL_NAME,
    )

    print()
    print("runtime input map:")
    print(module.get_input_name_to_index_map())

    expected_inputs = {
        "input_ids",
        "token_type_ids",
        "position_ids",
    }

    actual_inputs = set(
        module.get_input_name_to_index_map().keys()
    )

    if actual_inputs != expected_inputs:
        raise RuntimeError(
            "Probe graph does not have exactly 3 runtime inputs. "
            f"Got: {sorted(actual_inputs)}"
        )

    data = create_data(encoder)

    ait_constants = {
        name: torch_to_ait_data(value)
        for name, value in data["constants"].items()
    }

    module.set_many_constants(
        ait_constants,
    )

    ait_inputs = {
        "input_ids":
            torch_to_ait_data(data["input_ids"]),
        "token_type_ids":
            torch_to_ait_data(data["token_type_ids"]),
        "position_ids":
            torch_to_ait_data(data["position_ids"]),
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

    raw_static_mib = (
        data["static_raw_bytes"]
        / 1024.0
        / 1024.0
    )

    print()
    print("===== Static FC raw-weight release probe =====")
    print("pinned CPU :", cpu)
    print("threads    :", torch.get_num_threads())
    print("static FC raw weights+bias MiB:", raw_static_mib)
    print("RSS before first run MiB      :", current_rss_mib())
    print("Peak before first run MiB     :", peak_rss_mib())

    # First run creates/packs all static XNNPACK FC operators.
    module.run(
        ait_inputs,
        ait_outputs,
    )

    baseline = output.clone()

    rss_after_pack = current_rss_mib()
    peak_after_pack = peak_rss_mib()

    print()
    print("RSS after packing MiB         :", rss_after_pack)
    print("Peak after packing MiB        :", peak_after_pack)

    released = release_static_fc_pages(
        data
    )

    gc.collect()

    rss_after_release = current_rss_mib()

    print()
    print(
        "MADV_DONTNEED requested MiB   :",
        released / 1024.0 / 1024.0,
    )
    print(
        "RSS after raw release MiB     :",
        rss_after_release,
    )
    print(
        "Peak after raw release MiB    :",
        peak_rss_mib(),
    )

    # If static FC does not need its raw weights after packing,
    # this must still match the first output.
    module.run(
        ait_inputs,
        ait_outputs,
    )

    max_abs_diff = (
        output - baseline
    ).abs().max().item()

    mean_abs_diff = (
        output - baseline
    ).abs().mean().item()

    rss_after_second = current_rss_mib()

    print()
    print("===== Correctness after raw release =====")
    print("max abs diff :", max_abs_diff)
    print("mean abs diff:", mean_abs_diff)
    print("RSS after second run MiB:", rss_after_second)

    tolerance = 1e-6

    if max_abs_diff > tolerance:
        print()
        print("FAIL")
        print(
            "At least one released raw static-FC tensor "
            "is still read after operator creation."
        )
        raise SystemExit(1)

    print()
    print("PASS")
    print(
        "Projection/FFN raw weights are not required "
        "after XNNPACK static-FC packing."
    )

    stats = benchmark(
        lambda: module.run(
            ait_inputs,
            ait_outputs,
        )
    )

    print()
    print("===== Post-release latency =====")
    print("median ms:", stats["median"])
    print("mean ms  :", stats["mean"])
    print("min ms   :", stats["min"])
    print("max ms   :", stats["max"])

    print()
    print("===== Memory summary =====")
    print("raw static MiB        :", raw_static_mib)
    print("RSS after pack MiB    :", rss_after_pack)
    print("RSS after release MiB :", rss_after_second)

    if (
        rss_after_pack is not None
        and rss_after_second is not None
    ):
        print(
            "RSS delta MiB         :",
            rss_after_pack - rss_after_second,
        )

    print("Peak RSS MiB          :", peak_rss_mib())


if __name__ == "__main__":
    main()
