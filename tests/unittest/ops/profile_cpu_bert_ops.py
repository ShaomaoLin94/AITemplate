import importlib.util
import json
import os
import tempfile

import torch

from aitemplate.compiler.model import Model, torch_to_ait_data


# Reuse the exact current isolated benchmark lifecycle instead of
# duplicating its constant/prepack logic.
spec = importlib.util.spec_from_file_location(
    "cpu_bert_isolated_current",
    "tests/unittest/ops/benchmark_cpu_bert_isolated.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def _op_group(name):
    if name.startswith("gemm_rcr_bias_fast_gelu"):
        return "FFN1 GEMM + FastGELU"

    if name.startswith("gemm_rcr_bias_permute"):
        return "QKV GEMM + physical permute"

    if name.startswith("gemm_rcr_bias_add"):
        # Two shapes share this op family:
        #   projection: K=768
        #   FFN2:       K=3072
        # The profiler JSON does not retain dimensions, so keep the
        # per-op ranking as the authoritative split and aggregate the
        # family only as a secondary summary.
        return "Projection / FFN2 GEMM + residual"

    if name.startswith("bmm_softmax_bmm_permute"):
        return "Attention"

    if name.startswith("layernorm"):
        return "LayerNorm"

    if name.startswith("bert_embeddings"):
        return "Embeddings"

    if name.startswith("split"):
        return "QKV split/view"

    return "Other"


def main():
    print("===== Current AITemplate CPU BERT operator profiling =====")
    print("pinned CPU :", bench.cpu)
    print("threads    :", torch.get_num_threads())

    if not os.path.exists(bench.SO_PATH):
        raise RuntimeError(
            f"{bench.SO_PATH} not found. "
            "Rebuild benchmark_cpu_bert_full.py first."
        )

    data = bench.create_ait_data()
    module = Model(bench.SO_PATH)

    ait_inputs = {
        "input_ids":
            torch_to_ait_data(data["input_ids"]),
        "token_type_ids":
            torch_to_ait_data(data["token_type_ids"]),
        "position_ids":
            torch_to_ait_data(data["position_ids"]),
    }

    ait_constants = {
        "word_embeddings":
            torch_to_ait_data(data["word_embeddings"]),
        "token_type_embeddings":
            torch_to_ait_data(data["token_type_embeddings"]),
        "position_embeddings":
            torch_to_ait_data(data["position_embeddings"]),
        "embedding_gamma":
            torch_to_ait_data(data["embedding_gamma"]),
        "embedding_beta":
            torch_to_ait_data(data["embedding_beta"]),
    }

    for name, value in data["ait_params"].items():
        ait_constants[name] = torch_to_ait_data(value)

    module.set_many_constants(ait_constants)

    # Exact same 48-pair incremental prepack used by the formal
    # isolated benchmark. Dense raw constants become nullptr here.
    bench._prepack_static_fc_constants(
        module,
        data["manifest"],
    )

    output = torch.empty(
        bench.BATCH,
        bench.SEQ,
        bench.HIDDEN,
        dtype=torch.float32,
    )

    ait_outputs = {
        "output":
            torch_to_ait_data(output)
    }

    # Warm the real inference path. Static FCs were already packed,
    # so this does not include first-time packing cost.
    for _ in range(5):
        module.run(
            ait_inputs,
            ait_outputs,
        )

    with tempfile.NamedTemporaryFile(
        mode="r+",
        suffix=".json",
        delete=False,
    ) as f:
        profile_path = f.name

    try:
        print()
        print("Profiling operators (30 iterations/op)...")

        module.profile(
            ait_inputs,
            ait_outputs,
            num_iters=30,
            filename=profile_path,
        )

        with open(
            profile_path,
            "r",
            encoding="utf-8",
        ) as f:
            records = json.load(f)

    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    ranking = []

    for name, record in records.items():
        runtime_ms = float(
            record["ms_per_iter"]
        )

        ranking.append(
            (
                runtime_ms,
                name,
                _op_group(name),
            )
        )

    ranking.sort(reverse=True)

    standalone_layernorm = [
        name
        for _, name, _ in ranking
        if name.startswith("layernorm")
    ]

    if standalone_layernorm:
        print()
        print(
            "WARNING: standalone LayerNorm operators "
            "still exist in the profiled graph:"
        )

        for name in standalone_layernorm:
            print("  ", name)

    else:
        print()
        print(
            "Fusion check: no standalone encoder "
            "LayerNorm operators detected"
        )

    total_ms = sum(
        item[0]
        for item in ranking
    )

    print()
    print("===== Operator ranking =====")

    for rank, (
        runtime_ms,
        name,
        group,
    ) in enumerate(
        ranking,
        start=1,
    ):
        pct = (
            runtime_ms / total_ms * 100.0
            if total_ms > 0
            else 0.0
        )

        print(
            f"{rank:3d}. "
            f"{name:<52} "
            f"{runtime_ms:8.4f} ms  "
            f"{pct:6.2f}%  "
            f"{group}"
        )

    grouped = {}

    for runtime_ms, _, group in ranking:
        grouped[group] = (
            grouped.get(group, 0.0)
            + runtime_ms
        )

    print()
    print("===== Group totals =====")

    for group, runtime_ms in sorted(
        grouped.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        pct = (
            runtime_ms / total_ms * 100.0
            if total_ms > 0
            else 0.0
        )

        print(
            f"{group:<36} "
            f"{runtime_ms:8.3f} ms  "
            f"{pct:6.2f}%"
        )

    print()
    print("Profile sequential op total ms:", total_ms)
    print()
    print(
        "NOTE: Model::Profile measures each op separately, so the "
        "sum is for bottleneck attribution, not an end-to-end latency."
    )


if __name__ == "__main__":
    main()
