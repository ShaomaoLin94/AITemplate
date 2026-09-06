import argparse
import datetime
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time


SELF = os.path.abspath(__file__)

INFERENCE_SCRIPT = (
    "tests/unittest/ops/"
    "benchmark_cpu_megatron_bert_inference.py"
)

THREAD_COUNTS = [1, 2, 4, 8]
ROUNDS = 5

WARMUP = 2
ITERATIONS = 5

LOG_DIR = "benchmark_logs"


def _read_int(path):
    try:
        with open(path, "r") as f:
            return int(
                f.read().strip()
            )
    except (
        OSError,
        ValueError,
    ):
        return None


def physical_core_cpus():
    allowed = sorted(
        os.sched_getaffinity(0)
    )

    representatives = {}

    for cpu in allowed:
        base = (
            "/sys/devices/system/cpu/"
            f"cpu{cpu}/topology"
        )

        package_id = _read_int(
            f"{base}/physical_package_id"
        )

        core_id = _read_int(
            f"{base}/core_id"
        )

        if (
            package_id is None
            or core_id is None
        ):
            key = (
                "logical",
                cpu,
            )
        else:
            key = (
                package_id,
                core_id,
            )

        representatives.setdefault(
            key,
            cpu,
        )

    cpus = list(
        representatives.values()
    )

    if (
        0 in cpus
        and len(cpus) > 1
    ):
        cpus.remove(0)
        cpus.append(0)

    return cpus


def load_inference_module():
    spec = (
        importlib.util.spec_from_file_location(
            "megatron_inference",
            INFERENCE_SCRIPT,
        )
    )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def benchmark_function(function):
    for _ in range(WARMUP):
        function()

    samples = []

    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()

        function()

        end = time.perf_counter_ns()

        samples.append(
            (
                end - start
            )
            / 1_000_000.0
        )

    return {
        "median_ms":
            statistics.median(
                samples
            ),

        "mean_ms":
            statistics.mean(
                samples
            ),

        "min_ms":
            min(samples),

        "max_ms":
            max(samples),

        "samples_ms":
            samples,
    }


def build_pytorch_parameters(
    inference,
    metadata,
):
    params = {}

    total = (
        inference.NUM_LAYERS
    )

    print(
        "===== Building PyTorch "
        "Megatron parameters ====="
    )

    for layer_idx in range(
        total
    ):
        prefix = (
            f"layers."
            f"{layer_idx}."
        )

        suffixes = (
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

        for suffix in suffixes:
            logical_name = (
                prefix
                + suffix
            )

            params[
                logical_name
            ] = (
                inference.make_parameter(
                    logical_name,
                    metadata[
                        logical_name
                    ]["shape"],
                )
            )

        print(
            f"[{layer_idx + 1:02d}/"
            f"{total}] "
            "parameters ready"
        )

    return params


def build_pytorch_runner(
    inference,
    common,
    metadata,
):
    import torch
    import torch.nn.functional as F

    params = (
        build_pytorch_parameters(
            inference,
            metadata,
        )
    )

    @torch.no_grad()
    def run():
        x = (
            F.embedding(
                common[
                    "input_ids"
                ],
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
            (
                inference.HIDDEN,
            ),
            common[
                "embedding_gamma"
            ],
            common[
                "embedding_beta"
            ],
            inference.EPS,
        )

        for layer_idx in range(
            inference.NUM_LAYERS
        ):
            prefix = (
                f"layers."
                f"{layer_idx}."
            )

            x = (
                inference.pytorch_encoder_layer(
                    x,
                    params,
                    prefix,
                    inference.BATCH,
                    inference.SEQ,
                    inference.HIDDEN,
                    inference.HEADS,
                    inference.HEAD_DIM,
                    inference.EPS,
                )
            )

        return x

    return run, params


def build_ait_runner(
    inference,
    common,
    metadata,
    manifest,
):
    import torch

    from aitemplate.compiler.model import (
        Model,
        torch_to_ait_data,
    )

    module = Model(
        inference.SO_PATH
    )

    persistent = (
        inference.set_persistent_constants(
            module,
            common,
            metadata,
        )
    )

    inference.prepack_all_fc(
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
        inference.BATCH,
        inference.SEQ,
        inference.HIDDEN,
        dtype=torch.float32,
    )

    output_data = (
        torch_to_ait_data(
            output
        )
    )

    def run():
        module.run(
            ait_inputs,
            {
                "output":
                    output_data
            },
        )

        return output

    return (
        run,
        module,
        persistent,
        output,
    )


def worker(
    mode,
    threads,
    cpus,
):
    if mode not in (
        "ait",
        "torch",
    ):
        raise RuntimeError(
            "mode must be ait or torch"
        )

    selected_cpus = set(
        cpus[:threads]
    )

    os.sched_setaffinity(
        0,
        selected_cpus,
    )

    os.environ[
        "AIT_CPU_NUM_THREADS"
    ] = str(threads)

    os.environ[
        "OMP_NUM_THREADS"
    ] = str(threads)

    os.environ[
        "MKL_NUM_THREADS"
    ] = str(threads)

    os.environ[
        "OPENBLAS_NUM_THREADS"
    ] = str(threads)

    import torch

    torch.set_num_threads(
        threads
    )

    torch.set_num_interop_threads(
        1
    )

    inference = (
        load_inference_module()
    )

    # inference.py sets torch threads=1
    # during import. Restore the requested
    # benchmark setting afterwards.
    torch.set_num_threads(
        threads
    )

    metadata = (
        inference.parameter_metadata()
    )

    common = (
        inference.create_common_data()
    )

    print(
        "===== Megatron-BERT "
        "1.3B worker ====="
    )

    print(
        "mode       :",
        mode,
    )

    print(
        "threads    :",
        threads,
    )

    print(
        "CPUs       :",
        sorted(
            selected_cpus
        ),
    )

    print(
        "torch      :",
        torch.get_num_threads(),
    )

    print(
        "layers     :",
        inference.NUM_LAYERS,
    )

    print(
        "hidden     :",
        inference.HIDDEN,
    )

    print(
        "heads      :",
        inference.HEADS,
    )

    print(
        "FFN        :",
        inference.INTERMEDIATE,
    )

    if mode == "ait":
        manifest = (
            inference.load_manifest()
        )

        (
            run,
            module,
            persistent,
            output,
        ) = build_ait_runner(
            inference,
            common,
            metadata,
            manifest,
        )

        gc.collect()

        result = (
            benchmark_function(
                run
            )
        )

        # Keep model/constants alive through
        # the complete benchmark.
        _ = (
            module,
            persistent,
            output,
        )

    else:
        run, params = (
            build_pytorch_runner(
                inference,
                common,
                metadata,
            )
        )

        gc.collect()

        result = (
            benchmark_function(
                run
            )
        )

        _ = params

    rss, peak = (
        inference.memory_mib()
    )

    result.update(
        {
            "mode":
                mode,

            "threads":
                threads,

            "cpus":
                sorted(
                    selected_cpus
                ),

            "vmrss_mib":
                rss,

            "peak_rss_mib":
                peak,
        }
    )

    print()
    print(
        "===== Result ====="
    )

    print(
        "median ms:",
        result[
            "median_ms"
        ],
    )

    print(
        "mean ms  :",
        result[
            "mean_ms"
        ],
    )

    print(
        "min ms   :",
        result[
            "min_ms"
        ],
    )

    print(
        "max ms   :",
        result[
            "max_ms"
        ],
    )

    print(
        "RSS MiB  :",
        rss,
    )

    print(
        "Peak MiB :",
        peak,
    )

    print(
        "RESULT_JSON:",
        json.dumps(
            result,
            sort_keys=True,
        ),
    )


def run_worker(
    mode,
    threads,
    cpus,
):
    cpu_text = ",".join(
        str(cpu)
        for cpu in cpus[
            :threads
        ]
    )

    proc = subprocess.run(
        [
            "python3",
            SELF,
            "__worker__",
            mode,
            str(threads),
            cpu_text,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.returncode != 0:
        print(
            proc.stdout,
            end="",
        )

        raise RuntimeError(
            f"{mode} "
            f"{threads}-thread "
            "worker failed"
        )

    for line in reversed(
        proc.stdout.splitlines()
    ):
        if line.startswith(
            "RESULT_JSON:"
        ):
            return json.loads(
                line.split(
                    ":",
                    1,
                )[1].strip()
            )

    print(
        proc.stdout,
        end="",
    )

    raise RuntimeError(
        "No RESULT_JSON from worker"
    )


def median(values):
    return statistics.median(
        values
    )


def summarize(records):
    medians = [
        x["median_ms"]
        for x in records
    ]

    return {
        "median_of_medians_ms":
            median(medians),

        "mean_of_medians_ms":
            statistics.mean(
                medians
            ),

        "min_median_ms":
            min(medians),

        "max_median_ms":
            max(medians),

        "stdev_median_ms":
            (
                statistics.stdev(
                    medians
                )
                if len(medians) > 1
                else 0.0
            ),

        "median_vmrss_mib":
            median(
                [
                    x[
                        "vmrss_mib"
                    ]
                    for x in records
                ]
            ),

        "median_peak_rss_mib":
            median(
                [
                    x[
                        "peak_rss_mib"
                    ]
                    for x in records
                ]
            ),
    }


def save_json(
    path,
    data,
):
    temp = (
        path + ".tmp"
    )

    with open(
        temp,
        "w",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    os.replace(
        temp,
        path,
    )


def controller():
    cpus = (
        physical_core_cpus()
    )

    if len(cpus) < 8:
        raise RuntimeError(
            "Need at least 8 "
            "physical cores"
        )

    os.makedirs(
        LOG_DIR,
        exist_ok=True,
    )

    timestamp = (
        datetime.datetime.now()
        .astimezone()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    txt_path = os.path.join(
        LOG_DIR,
        (
            "megatron_bert_1_3b_"
            "thread_stability_"
            f"{timestamp}.txt"
        ),
    )

    json_path = os.path.join(
        LOG_DIR,
        (
            "megatron_bert_1_3b_"
            "thread_stability_"
            f"{timestamp}.json"
        ),
    )

    data = {
        "model":
            "Megatron-BERT 1.3B",

        "layers":
            24,

        "batch":
            1,

        "sequence":
            128,

        "hidden":
            2048,

        "heads":
            32,

        "intermediate":
            8192,

        "dtype":
            "float32",

        "rounds":
            ROUNDS,

        "warmup":
            WARMUP,

        "iterations_per_round":
            ITERATIONS,

        "thread_counts":
            THREAD_COUNTS,

        "physical_core_cpus":
            cpus,

        "results":
            {},
    }

    with open(
        txt_path,
        "w",
        buffering=1,
    ) as log_file:

        def log(*values):
            text = " ".join(
                str(v)
                for v in values
            )

            print(
                text,
                flush=True,
            )

            print(
                text,
                file=log_file,
                flush=True,
            )

        log(
            "===== Megatron-BERT "
            "1.3B thread stability ====="
        )

        log(
            "rounds/thread:",
            ROUNDS,
        )

        log(
            "threads:",
            THREAD_COUNTS,
        )

        log(
            "physical CPUs:",
            cpus,
        )

        for thread_index, threads in enumerate(
            THREAD_COUNTS
        ):
            log()
            log(
                "================================"
            )
            log(
                "THREADS =",
                threads,
            )
            log(
                "================================"
            )

            records = {
                "ait": [],
                "torch": [],
                "paired_speedups": [],
            }

            for round_idx in range(
                ROUNDS
            ):
                if (
                    round_idx
                    + thread_index
                ) % 2 == 0:
                    order = (
                        "ait",
                        "torch",
                    )
                else:
                    order = (
                        "torch",
                        "ait",
                    )

                log()
                log(
                    f"Round "
                    f"{round_idx + 1}/"
                    f"{ROUNDS}:",
                    f"{order[0]}"
                    " -> "
                    f"{order[1]}",
                )

                current = {}

                for mode in order:
                    result = (
                        run_worker(
                            mode,
                            threads,
                            cpus,
                        )
                    )

                    current[
                        mode
                    ] = result

                    records[
                        mode
                    ].append(
                        result
                    )

                    log(
                        f"{mode:<5}",
                        f"median="
                        f"{result['median_ms']:.3f} ms",
                        f"mean="
                        f"{result['mean_ms']:.3f} ms",
                        f"RSS="
                        f"{result['vmrss_mib']:.1f} MiB",
                        f"peak="
                        f"{result['peak_rss_mib']:.1f} MiB",
                    )

                speedup = (
                    current[
                        "torch"
                    ][
                        "median_ms"
                    ]
                    /
                    current[
                        "ait"
                    ][
                        "median_ms"
                    ]
                )

                records[
                    "paired_speedups"
                ].append(
                    speedup
                )

                log(
                    "paired AIT speedup:",
                    f"{speedup:.5f}x",
                )

                data[
                    "results"
                ][
                    str(threads)
                ] = records

                save_json(
                    json_path,
                    data,
                )

            ait_summary = summarize(
                records["ait"]
            )

            torch_summary = summarize(
                records["torch"]
            )

            paired_median = median(
                records[
                    "paired_speedups"
                ]
            )

            records[
                "summary"
            ] = {
                "ait":
                    ait_summary,

                "torch":
                    torch_summary,

                "paired_speedup_median":
                    paired_median,

                "aggregate_speedup":
                    (
                        torch_summary[
                            "median_of_medians_ms"
                        ]
                        /
                        ait_summary[
                            "median_of_medians_ms"
                        ]
                    ),
            }

            log()
            log(
                f"===== "
                f"{threads}-thread "
                "summary ====="
            )

            log(
                "AIT median-of-medians:",
                f"{ait_summary['median_of_medians_ms']:.3f} ms",
            )

            log(
                "PT median-of-medians :",
                f"{torch_summary['median_of_medians_ms']:.3f} ms",
            )

            log(
                "paired speedup median:",
                f"{paired_median:.5f}x",
            )

            save_json(
                json_path,
                data,
            )

        baseline_ait = (
            data["results"]["1"]
            ["summary"]["ait"]
            ["median_of_medians_ms"]
        )

        baseline_torch = (
            data["results"]["1"]
            ["summary"]["torch"]
            ["median_of_medians_ms"]
        )

        log()
        log(
            "===== FINAL SUMMARY ====="
        )

        log(
            "threads | AIT ms | "
            "AIT scale | PT ms | "
            "PT scale | AIT/PT"
        )

        scaling = {}

        for threads in (
            THREAD_COUNTS
        ):
            result = (
                data[
                    "results"
                ][str(threads)]
                ["summary"]
            )

            ait_ms = (
                result["ait"]
                [
                    "median_of_medians_ms"
                ]
            )

            torch_ms = (
                result["torch"]
                [
                    "median_of_medians_ms"
                ]
            )

            ait_scale = (
                baseline_ait
                / ait_ms
            )

            torch_scale = (
                baseline_torch
                / torch_ms
            )

            speedup = (
                torch_ms
                / ait_ms
            )

            scaling[
                str(threads)
            ] = {
                "ait_ms":
                    ait_ms,

                "torch_ms":
                    torch_ms,

                "ait_scaling":
                    ait_scale,

                "torch_scaling":
                    torch_scale,

                "ait_vs_torch":
                    speedup,
            }

            log(
                f"{threads:>7} | "
                f"{ait_ms:>8.2f} | "
                f"{ait_scale:>8.3f}x | "
                f"{torch_ms:>8.2f} | "
                f"{torch_scale:>8.3f}x | "
                f"{speedup:>7.3f}x"
            )

        data[
            "scaling_summary"
        ] = scaling

        save_json(
            json_path,
            data,
        )

        log()
        log(
            "TXT :",
            txt_path,
        )

        log(
            "JSON:",
            json_path,
        )


if __name__ == "__main__":
    if (
        len(sys.argv) >= 2
        and sys.argv[1]
        == "__worker__"
    ):
        if len(sys.argv) != 5:
            raise RuntimeError(
                "__worker__ "
                "mode threads cpus"
            )

        worker(
            sys.argv[2],
            int(
                sys.argv[3]
            ),
            [
                int(cpu)
                for cpu
                in sys.argv[
                    4
                ].split(",")
            ],
        )

    else:
        controller()
