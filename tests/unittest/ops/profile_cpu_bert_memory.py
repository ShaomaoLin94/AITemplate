import gc
import importlib.util
import os

import torch

from aitemplate.compiler.model import Model, torch_to_ait_data


spec = importlib.util.spec_from_file_location(
    "bench",
    "tests/unittest/ops/benchmark_cpu_bert_isolated.py",
)

bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0

    return 0.0


def show(name):
    print(f"{name:<28}: {rss():9.2f} MiB")


show("process start")

data = bench.create_data()
show("after create_data")

module = Model(
    "./tmp/benchmark_cpu_bert_full/test.so"
)
show("after Model()")

ait_inputs = {
    "input_ids": torch_to_ait_data(
        data["input_ids"]
    ),
    "token_type_ids": torch_to_ait_data(
        data["token_type_ids"]
    ),
    "position_ids": torch_to_ait_data(
        data["position_ids"]
    ),
    "word_embeddings": torch_to_ait_data(
        data["word_embeddings"]
    ),
    "token_type_embeddings": torch_to_ait_data(
        data["token_type_embeddings"]
    ),
    "position_embeddings": torch_to_ait_data(
        data["position_embeddings"]
    ),
    "embedding_gamma": torch_to_ait_data(
        data["embedding_gamma"]
    ),
    "embedding_beta": torch_to_ait_data(
        data["embedding_beta"]
    ),
}

for name, value in data["ait_params"].items():
    ait_inputs[name] = torch_to_ait_data(value)

output = torch.empty(
    bench.BATCH,
    bench.SEQ,
    bench.HIDDEN,
    dtype=torch.float32,
)

ait_outputs = {
    "output": torch_to_ait_data(output)
}

show("after AITData setup")

for i in range(1, 6):
    module.run(
        ait_inputs,
        ait_outputs,
    )

    show(f"after inference {i}")

gc.collect()
show("after gc")
