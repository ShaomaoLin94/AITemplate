import importlib.util
import json
import os
import tempfile


spec = importlib.util.spec_from_file_location(
    "bert_large_bench",
    "tests/unittest/ops/benchmark_cpu_bert_large_isolated.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def build_fc_symbol_map(manifest):
    mapping = {}
    for pair in manifest["pairs"]:
        symbol = pair["prepack_symbol"]
        if not symbol.endswith("_prepack"):
            continue
        mapping[symbol[: -len("_prepack")]] = pair["logical_base"]
    return mapping


def op_group(name, fc_symbol_map):
    logical = fc_symbol_map.get(name)

    if logical is not None:
        if logical.endswith(".mha.qkv"):
            return "QKV FC + physical permute"
        if logical.endswith(".mha.proj"):
            return "Projection FC + residual + LayerNorm"
        if logical.endswith(".ffn1"):
            return "FFN1 FC + FastGELU"
        if logical.endswith(".ffn2"):
            return "FFN2 FC + residual + LayerNorm"

    if name.startswith("bmm_softmax_bmm_permute"):
        return "Attention"
    if name.startswith("bert_embeddings"):
        return "Embeddings + LayerNorm"
    if name.startswith("split"):
        return "QKV split/view"
    if name.startswith("layernorm"):
        return "Standalone LayerNorm"
    return "Other"


def main():
    cpu = bench.pin_single_cpu()
    print("===== AITemplate CPU BERT-large precise operator profiling =====")
    print("pinned CPU:", cpu)

    meta = bench.build_param_meta()
    manifest = bench.load_manifest()
    fc_symbol_map = build_fc_symbol_map(manifest)
    data = bench.create_common_data()
    prepared = bench.prepare_ait(
        data,
        meta,
        manifest,
        verbose=False,
    )

    module = prepared["module"]
    inputs = prepared["inputs"]
    outputs = prepared["outputs"]

    for _ in range(5):
        module.run(inputs, outputs)

    with tempfile.NamedTemporaryFile(
        mode="r+",
        suffix=".json",
        delete=False,
    ) as f:
        profile_path = f.name

    try:
        module.profile(
            inputs,
            outputs,
            num_iters=30,
            filename=profile_path,
        )

        with open(profile_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    finally:
        try:
            os.remove(profile_path)
        except OSError:
            pass

    ranking = []
    for name, record in records.items():
        runtime_ms = float(record["ms_per_iter"])
        logical = fc_symbol_map.get(name, "")
        group = op_group(name, fc_symbol_map)
        ranking.append((runtime_ms, name, group, logical))

    ranking.sort(reverse=True)
    total_ms = sum(item[0] for item in ranking)

    print()
    print("===== Operator ranking =====")
    for rank, (runtime_ms, name, group, logical) in enumerate(ranking, start=1):
        pct = runtime_ms / total_ms * 100.0 if total_ms else 0.0
        logical_text = f" [{logical}]" if logical else ""
        print(
            f"{rank:3d}. {name:<54} "
            f"{runtime_ms:9.4f} ms  {pct:6.2f}%  {group}{logical_text}"
        )

    grouped = {}
    group_counts = {}
    for runtime_ms, _, group, _ in ranking:
        grouped[group] = grouped.get(group, 0.0) + runtime_ms
        group_counts[group] = group_counts.get(group, 0) + 1

    print()
    print("===== Group totals =====")
    ordered = sorted(
        grouped.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    group_json = {}
    for group, runtime_ms in ordered:
        pct = runtime_ms / total_ms * 100.0 if total_ms else 0.0
        count = group_counts[group]
        avg = runtime_ms / count
        print(
            f"{group:<42} {runtime_ms:9.4f} ms  "
            f"{pct:6.2f}%  count={count:3d}  avg={avg:.4f} ms"
        )
        group_json[group] = {
            "total_ms": runtime_ms,
            "pct": pct,
            "count": count,
            "avg_ms": avg,
        }

    print()
    print("profile total ms:", total_ms)
    print("GROUP_JSON:", json.dumps(group_json, sort_keys=True))


if __name__ == "__main__":
    main()
