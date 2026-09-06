import datetime
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


SELF = os.path.abspath(__file__)

ISOLATED = (
    "tests/unittest/ops/"
    "benchmark_cpu_bert_isolated.py"
)

THREAD_COUNTS = [1, 2, 4]
ROUNDS = 10
LOG_DIR = "benchmark_logs"


def _read_int(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def representative_cpus():
    allowed = sorted(
        os.sched_getaffinity(0)
    )

    groups = {}

    for cpu in allowed:
        base = (
            f"/sys/devices/system/cpu/"
            f"cpu{cpu}/topology"
        )

        package = _read_int(
            f"{base}/physical_package_id"
        )

        core = _read_int(
            f"{base}/core_id"
        )

        if (
            package is None
            or core is None
        ):
            key = ("logical", cpu)
        else:
            key = (
                package,
                core,
            )

        groups.setdefault(
            key,
            cpu,
        )

    cpus = list(
        groups.values()
    )

    if 0 in cpus and len(cpus) > 1:
        cpus.remove(0)
        cpus.append(0)

    return cpus


def load_isolated():
    spec = (
        importlib.util.spec_from_file_location(
            "bert_base_final_worker",
            ISOLATED,
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


def worker(
    mode,
    threads,
    cpus,
):
    import contextlib
    import io

    if mode not in (
        "ait",
        "torch",
    ):
        raise RuntimeError(
            "mode must be ait or torch"
        )

    selected = set(
        cpus[:threads]
    )

    real_affinity = (
        os.sched_setaffinity
    )

    real_affinity(
        0,
        selected,
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

    real_set_threads = (
        torch.set_num_threads
    )

    real_set_interop = (
        torch.set_num_interop_threads
    )

    real_set_threads(
        threads
    )

    real_set_interop(
        1
    )

    # Prevent benchmark_cpu_bert_isolated.py
    # from forcing itself back to one CPU/thread
    # while it is imported.
    torch.set_num_threads = (
        lambda ignored: None
    )

    torch.set_num_interop_threads = (
        lambda ignored: None
    )

    os.sched_setaffinity = (
        lambda pid, mask: None
    )

    bench = load_isolated()

    torch.set_num_threads = (
        real_set_threads
    )

    torch.set_num_interop_threads = (
        real_set_interop
    )

    os.sched_setaffinity = (
        real_affinity
    )

    real_set_threads(
        threads
    )

    print(
        "===== BERT-base worker ====="
    )

    print(
        "mode    :",
        mode,
    )

    print(
        "threads :",
        threads,
    )

    print(
        "CPUs    :",
        sorted(selected),
    )

    print(
        "torch   :",
        torch.get_num_threads(),
    )

    # Let the isolated benchmark use its own
    # current data-creation/runtime API.
    #
    # Prefer ait-release when that mode exists,
    # so packed raw FC weights are released.
    source = Path(
        ISOLATED
    ).read_text()

    if mode == "ait":
        isolated_mode = (
            "ait-release"
            if "ait-release" in source
            else "ait"
        )
    else:
        isolated_mode = "torch"

    old_argv = sys.argv

    sys.argv = [
        ISOLATED,
        isolated_mode,
    ]

    capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(
            capture
        ):
            bench.main()
    finally:
        sys.argv = old_argv

    output_text = (
        capture.getvalue()
    )

    print(
        output_text,
        end="",
    )

    # If the isolated benchmark already emits
    # RESULT_JSON, reuse it directly.
    for line in reversed(
        output_text.splitlines()
    ):
        if line.startswith(
            "RESULT_JSON:"
        ):
            result = json.loads(
                line.split(
                    ":",
                    1,
                )[1].strip()
            )

            result[
                "threads"
            ] = threads

            result[
                "cpus"
            ] = sorted(
                selected
            )

            print(
                "RESULT_JSON:",
                json.dumps(
                    result,
                    sort_keys=True,
                ),
            )

            return

    values = {}

    for line in (
        output_text.splitlines()
    ):
        stripped = line.strip()

        if stripped.startswith(
            "median ms:"
        ):
            values[
                "median_ms"
            ] = float(
                stripped.split(
                    ":",
                    1,
                )[1]
            )

        elif stripped.startswith(
            "mean ms"
        ):
            values[
                "mean_ms"
            ] = float(
                stripped.split(
                    ":",
                    1,
                )[1]
            )

        elif stripped.startswith(
            "min ms"
        ):
            values[
                "min_ms"
            ] = float(
                stripped.split(
                    ":",
                    1,
                )[1]
            )

        elif stripped.startswith(
            "max ms"
        ):
            values[
                "max_ms"
            ] = float(
                stripped.split(
                    ":",
                    1,
                )[1]
            )

        elif stripped.startswith(
            "Peak RSS MiB:"
        ):
            values[
                "peak_rss_mib"
            ] = float(
                stripped.split(
                    ":",
                    1,
                )[1]
            )

        elif stripped.startswith(
            "VmRSS MiB"
        ):
            values[
                "vmrss_mib"
            ] = float(
                stripped.split(
                    ":",
                    1,
                )[1]
            )

    required = (
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "vmrss_mib",
        "peak_rss_mib",
    )

    missing = [
        key
        for key in required
        if key not in values
    ]

    if missing:
        raise RuntimeError(
            "Could not parse isolated "
            "benchmark output; missing: "
            + ", ".join(missing)
        )

    result = {
        "mode":
            mode,

        "threads":
            threads,

        "cpus":
            sorted(selected),

        **values,
    }

    print(
        "RESULT_JSON:",
        json.dumps(
            result,
            sort_keys=True,
        ),
    )


def run_once(
    mode,
    threads,
    cpus,
):
    cpu_text = ",".join(
        str(cpu)
        for cpu in cpus[:threads]
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
            f"{mode} {threads}T failed"
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
        "worker produced no RESULT_JSON"
    )


def summarize(records):
    medians = [
        record["median_ms"]
        for record in records
    ]

    mean_value = (
        statistics.mean(
            medians
        )
    )

    stdev = (
        statistics.stdev(
            medians
        )
        if len(medians) > 1
        else 0.0
    )

    return {
        "median_of_medians_ms":
            statistics.median(
                medians
            ),

        "mean_of_medians_ms":
            mean_value,

        "stdev_of_medians_ms":
            stdev,

        "cv_pct":
            (
                stdev
                / mean_value
                * 100.0
            ),

        "min_median_ms":
            min(medians),

        "max_median_ms":
            max(medians),

        "median_vmrss_mib":
            statistics.median(
                [
                    record[
                        "vmrss_mib"
                    ]
                    for record
                    in records
                ]
            ),

        "median_peak_rss_mib":
            statistics.median(
                [
                    record[
                        "peak_rss_mib"
                    ]
                    for record
                    in records
                ]
            ),
    }


def save_json(path, data):
    temp = path + ".tmp"

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
    cpus = representative_cpus()

    if len(cpus) < 4:
        raise RuntimeError(
            "Need at least 4 "
            "guest-reported cores"
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
            "bert_base_thread_stability_"
            f"{timestamp}.txt"
        ),
    )

    json_path = os.path.join(
        LOG_DIR,
        (
            "bert_base_thread_stability_"
            f"{timestamp}.json"
        ),
    )

    data = {
        "model": {
            "name":
                "BERT-base",

            "layers":
                12,

            "batch":
                1,

            "sequence":
                128,

            "hidden":
                768,

            "heads":
                12,

            "intermediate":
                3072,

            "dtype":
                "float32",
        },

        "rounds":
            ROUNDS,

        "thread_counts":
            THREAD_COUNTS,

        "representative_cpus":
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
                str(value)
                for value in values
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
            "===== BERT-base "
            "thread stability ====="
        )

        log(
            "threads:",
            THREAD_COUNTS,
        )

        log(
            "rounds/thread:",
            ROUNDS,
        )

        log(
            "representative CPUs:",
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

            current = {
                "ait": [],
                "torch": [],
                "paired_speedups": [],
            }

            data[
                "results"
            ][str(threads)] = current

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
                    f"{ROUNDS}: "
                    f"{order[0]} -> "
                    f"{order[1]}"
                )

                pair = {}

                for mode in order:
                    result = run_once(
                        mode,
                        threads,
                        cpus,
                    )

                    pair[mode] = result

                    current[
                        mode
                    ].append(
                        result
                    )

                    log(
                        f"{mode:<5} "
                        f"median="
                        f"{result['median_ms']:.3f} ms  "
                        f"mean="
                        f"{result['mean_ms']:.3f} ms  "
                        f"RSS="
                        f"{result['vmrss_mib']:.1f} MiB"
                    )

                speedup = (
                    pair["torch"][
                        "median_ms"
                    ]
                    /
                    pair["ait"][
                        "median_ms"
                    ]
                )

                current[
                    "paired_speedups"
                ].append(
                    speedup
                )

                log(
                    "paired AIT speedup:",
                    f"{speedup:.5f}x",
                )

                save_json(
                    json_path,
                    data,
                )

            ait_summary = summarize(
                current["ait"]
            )

            torch_summary = summarize(
                current["torch"]
            )

            current[
                "summary"
            ] = {
                "ait":
                    ait_summary,

                "torch":
                    torch_summary,

                "paired_speedup_median":
                    statistics.median(
                        current[
                            "paired_speedups"
                        ]
                    ),

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
                f"===== {threads}T summary ====="
            )

            log(
                "AIT:",
                f"{ait_summary['median_of_medians_ms']:.3f} ms",
            )

            log(
                "PT :",
                f"{torch_summary['median_of_medians_ms']:.3f} ms",
            )

            log(
                "AIT CV:",
                f"{ait_summary['cv_pct']:.3f}%",
            )

            log(
                "PT CV :",
                f"{torch_summary['cv_pct']:.3f}%",
            )

            log(
                "AIT/PT:",
                f"{current['summary']['aggregate_speedup']:.5f}x",
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

        baseline_pt = (
            data["results"]["1"]
            ["summary"]["torch"]
            ["median_of_medians_ms"]
        )

        scaling = {}

        log()
        log(
            "===== FINAL SUMMARY ====="
        )

        log(
            "threads | AIT ms | "
            "AIT scale | PT ms | "
            "PT scale | AIT/PT"
        )

        for threads in THREAD_COUNTS:
            summary = (
                data[
                    "results"
                ][str(threads)][
                    "summary"
                ]
            )

            ait_ms = (
                summary["ait"][
                    "median_of_medians_ms"
                ]
            )

            pt_ms = (
                summary["torch"][
                    "median_of_medians_ms"
                ]
            )

            ait_scale = (
                baseline_ait
                / ait_ms
            )

            pt_scale = (
                baseline_pt
                / pt_ms
            )

            speedup = (
                pt_ms
                / ait_ms
            )

            scaling[
                str(threads)
            ] = {
                "ait_ms":
                    ait_ms,

                "torch_ms":
                    pt_ms,

                "ait_scaling":
                    ait_scale,

                "torch_scaling":
                    pt_scale,

                "ait_vs_torch":
                    speedup,
            }

            log(
                f"{threads:>7} | "
                f"{ait_ms:>7.2f} | "
                f"{ait_scale:>8.3f}x | "
                f"{pt_ms:>7.2f} | "
                f"{pt_scale:>8.3f}x | "
                f"{speedup:>6.3f}x"
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
            int(sys.argv[3]),
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
