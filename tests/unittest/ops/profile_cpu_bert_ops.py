import importlib.util
import json
import os

import torch

from aitemplate.compiler.model import Model, torch_to_ait_data


spec = importlib.util.spec_from_file_location(
    "cpu_bert_benchmark",
    "tests/unittest/ops/benchmark_cpu_bert_isolated.py",
)

bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main():
    so_path = "./tmp/benchmark_cpu_bert_full/test.so"

    if not os.path.exists(so_path):
        raise RuntimeError(
            f"{so_path} not found. "
            "Run benchmark_cpu_bert_full.py first."
        )

    print("===== AITemplate CPU BERT operator profiling =====")
    print("pinned CPU :", bench.cpu)
    print("threads    :", torch.get_num_threads())

    data = bench.create_data()
    module = Model(so_path)

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

    # Warm up the complete model first.
    for _ in range(5):
        module.run(
            ait_inputs,
            ait_outputs,
        )

    report_path = "/tmp/ait_bert_ops.json"

    print()
    print("Profiling operators...")

    module.profile(
        ait_inputs,
        ait_outputs,
        num_iters=20,
        filename=report_path,
    )

    with open(report_path, "r") as f:
        report = json.load(f)

    results = []

    for name, info in report.items():
        results.append(
            (
                name,
                float(info["ms_per_iter"]),
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    total = sum(ms for _, ms in results)

    print()
    print("===== Operator ranking =====")

    for rank, (name, ms) in enumerate(
        results,
        start=1,
    ):
        percent = (
            100.0 * ms / total
            if total > 0
            else 0.0
        )

        print(
            f"{rank:2d}. "
            f"{name:<45} "
            f"{ms:9.4f} ms  "
            f"{percent:6.2f}%"
        )

    print()
    print("===== Sum of isolated op times =====")
    print(f"{total:.4f} ms")

    print()
    print("raw report:", report_path)


if __name__ == "__main__":
    main()
