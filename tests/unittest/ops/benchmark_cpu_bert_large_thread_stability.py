import datetime
import importlib.util
import json
import os
import random
import statistics
import subprocess


SCALING_SCRIPT = (
    "tests/unittest/ops/"
    "benchmark_cpu_bert_large_thread_scaling.py"
)

THREAD_COUNTS = [1, 2, 4, 8]
ROUNDS = 10

LOG_DIR = "benchmark_logs"


def load_scaling_module():
    spec = importlib.util.spec_from_file_location(
        "bert_large_thread_scaling",
        SCALING_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def median(values):
    return statistics.median(values)


def mean(values):
    return statistics.mean(values)


def stdev(values):
    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def summarize(records):
    medians = [
        record["median_ms"]
        for record in records
    ]

    means = [
        record["mean_ms"]
        for record in records
    ]

    rss = [
        record["vmrss_mib"]
        for record in records
    ]

    peaks = [
        record["peak_rss_mib"]
        for record in records
    ]

    mean_median = mean(medians)

    return {
        "median_of_medians_ms":
            median(medians),

        "mean_of_medians_ms":
            mean_median,

        "stdev_of_medians_ms":
            stdev(medians),

        "cv_of_medians_pct":
            (
                stdev(medians)
                / mean_median
                * 100.0
                if mean_median > 0.0
                else 0.0
            ),

        "min_median_ms":
            min(medians),

        "max_median_ms":
            max(medians),

        "median_of_means_ms":
            median(means),

        "median_vmrss_mib":
            median(rss),

        "median_peak_rss_mib":
            median(peaks),
    }


def bootstrap_median_ci(
    values,
    seed,
    samples=20000,
):
    rng = random.Random(seed)

    estimates = []

    n = len(values)

    for _ in range(samples):
        sample = [
            values[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        estimates.append(
            statistics.median(
                sample
            )
        )

    estimates.sort()

    low = int(
        0.025
        * (samples - 1)
    )

    high = int(
        0.975
        * (samples - 1)
    )

    return (
        estimates[low],
        estimates[high],
    )


def git_commit():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:
        return None


def git_dirty():
    try:
        result = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            text=True,
        )

        return bool(
            result.strip()
        )

    except Exception:
        return None


def cpu_model():
    try:
        with open(
            "/proc/cpuinfo",
            "r",
        ) as f:
            for line in f:
                if line.startswith(
                    "model name"
                ):
                    return (
                        line.split(
                            ":",
                            1,
                        )[1]
                        .strip()
                    )
    except OSError:
        pass

    return None


def save_json(
    path,
    data,
):
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


def main():
    scaling = (
        load_scaling_module()
    )

    cpus = (
        scaling.physical_core_cpus()
    )

    if len(cpus) < max(
        THREAD_COUNTS
    ):
        raise RuntimeError(
            "Need at least 8 physical cores; "
            f"found {len(cpus)}"
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
            "bert_large_thread_stability_"
            f"{timestamp}.txt"
        ),
    )

    json_path = os.path.join(
        LOG_DIR,
        (
            "bert_large_thread_stability_"
            f"{timestamp}.json"
        ),
    )

    result_data = {
        "timestamp":
            datetime.datetime.now()
            .astimezone()
            .isoformat(),

        "model": {
            "name":
                "BERT-large",

            "layers":
                24,

            "batch":
                1,

            "sequence":
                128,

            "hidden":
                1024,

            "heads":
                16,

            "intermediate":
                4096,
        },

        "rounds":
            ROUNDS,

        "thread_counts":
            THREAD_COUNTS,

        "physical_core_cpus":
            cpus,

        "cpu_model":
            cpu_model(),

        "git_commit":
            git_commit(),

        "git_dirty":
            git_dirty(),

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
            "===== BERT-large "
            "10-round thread stability ====="
        )

        log(
            "timestamp       :",
            result_data[
                "timestamp"
            ],
        )

        log(
            "CPU             :",
            result_data[
                "cpu_model"
            ],
        )

        log(
            "physical cores  :",
            len(cpus),
        )

        log(
            "representatives :",
            cpus,
        )

        log(
            "threads         :",
            THREAD_COUNTS,
        )

        log(
            "rounds/thread   :",
            ROUNDS,
        )

        log(
            "git commit      :",
            result_data[
                "git_commit"
            ],
        )

        log(
            "working tree "
            "dirty:",
            result_data[
                "git_dirty"
            ],
        )

        log()

        for thread_index, threads in enumerate(
            THREAD_COUNTS
        ):
            selected_cpus = (
                cpus[:threads]
            )

            log(
                "========================================"
            )

            log(
                f"THREADS = {threads}"
            )

            log(
                "CPUs:",
                selected_cpus,
            )

            log(
                "========================================"
            )

            thread_result = {
                "threads":
                    threads,

                "cpus":
                    selected_cpus,

                "round_records":
                    [],

                "ait":
                    [],

                "torch":
                    [],

                "paired_speedups":
                    [],
            }

            result_data[
                "results"
            ][str(threads)] = (
                thread_result
            )

            for round_idx in range(
                ROUNDS
            ):
                # Alternate both by round and thread
                # count so one mode does not always
                # run first after a thread-count change.
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

                round_result = {}

                for mode in order:
                    record = (
                        scaling.run_once(
                            mode,
                            threads,
                            selected_cpus,
                        )
                    )

                    round_result[
                        mode
                    ] = record

                    thread_result[
                        mode
                    ].append(
                        record
                    )

                    spread = (
                        record[
                            "max_ms"
                        ]
                        - record[
                            "min_ms"
                        ]
                    )

                    log(
                        f"{mode:<5} "
                        f"median="
                        f"{record['median_ms']:.3f} ms  "
                        f"mean="
                        f"{record['mean_ms']:.3f} ms  "
                        f"min="
                        f"{record['min_ms']:.3f}  "
                        f"max="
                        f"{record['max_ms']:.3f}  "
                        f"spread="
                        f"{spread:.3f} ms  "
                        f"RSS="
                        f"{record['vmrss_mib']:.1f} MiB"
                    )

                pair_speedup = (
                    round_result[
                        "torch"
                    ][
                        "median_ms"
                    ]
                    /
                    round_result[
                        "ait"
                    ][
                        "median_ms"
                    ]
                )

                thread_result[
                    "paired_speedups"
                ].append(
                    pair_speedup
                )

                thread_result[
                    "round_records"
                ].append(
                    {
                        "round":
                            round_idx + 1,

                        "order":
                            list(order),

                        "ait":
                            round_result[
                                "ait"
                            ],

                        "torch":
                            round_result[
                                "torch"
                            ],

                        "paired_speedup":
                            pair_speedup,
                    }
                )

                log(
                    "paired AIT speedup:",
                    f"{pair_speedup:.5f}x",
                )

                # Persist after every round so an
                # interrupted run still leaves data.
                save_json(
                    json_path,
                    result_data,
                )

            ait_summary = summarize(
                thread_result[
                    "ait"
                ]
            )

            torch_summary = summarize(
                thread_result[
                    "torch"
                ]
            )

            pair_median = median(
                thread_result[
                    "paired_speedups"
                ]
            )

            ci_low, ci_high = (
                bootstrap_median_ci(
                    thread_result[
                        "paired_speedups"
                    ],
                    seed=(
                        20260904
                        + threads
                    ),
                )
            )

            aggregate_speedup = (
                torch_summary[
                    "median_of_medians_ms"
                ]
                /
                ait_summary[
                    "median_of_medians_ms"
                ]
            )

            jitter_high = (
                ait_summary[
                    "cv_of_medians_pct"
                ]
                > 2.5
                or
                torch_summary[
                    "cv_of_medians_pct"
                ]
                > 2.5
            )

            thread_result[
                "summary"
            ] = {
                "ait":
                    ait_summary,

                "torch":
                    torch_summary,

                "aggregate_speedup":
                    aggregate_speedup,

                "paired_speedup_median":
                    pair_median,

                "paired_ci95_low":
                    ci_low,

                "paired_ci95_high":
                    ci_high,

                "jitter_high":
                    jitter_high,
            }

            log()
            log(
                f"===== {threads}-thread summary ====="
            )

            log(
                "AIT median-of-medians :",
                f"{ait_summary['median_of_medians_ms']:.3f} ms",
            )

            log(
                "PT median-of-medians  :",
                f"{torch_summary['median_of_medians_ms']:.3f} ms",
            )

            log(
                "AIT CV                :",
                f"{ait_summary['cv_of_medians_pct']:.3f}%",
            )

            log(
                "PT CV                 :",
                f"{torch_summary['cv_of_medians_pct']:.3f}%",
            )

            log(
                "aggregate AIT/PT       :",
                f"{aggregate_speedup:.5f}x",
            )

            log(
                "paired median          :",
                f"{pair_median:.5f}x",
            )

            log(
                "paired median 95% CI   :",
                f"{ci_low:.5f}x .. "
                f"{ci_high:.5f}x",
            )

            log(
                "jitter                 :",
                (
                    "HIGH"
                    if jitter_high
                    else "controlled"
                ),
            )

            save_json(
                json_path,
                result_data,
            )

        baseline = (
            result_data[
                "results"
            ]["1"]["summary"]
        )

        baseline_ait = (
            baseline[
                "ait"
            ][
                "median_of_medians_ms"
            ]
        )

        baseline_pt = (
            baseline[
                "torch"
            ][
                "median_of_medians_ms"
            ]
        )

        log()
        log(
            "========================================"
        )
        log(
            "FINAL SCALING SUMMARY"
        )
        log(
            "========================================"
        )

        log(
            "threads | "
            "AIT ms | "
            "AIT scale | "
            "PT ms | "
            "PT scale | "
            "AIT/PT"
        )

        scaling_summary = {}

        for threads in THREAD_COUNTS:
            summary = (
                result_data[
                    "results"
                ][str(threads)][
                    "summary"
                ]
            )

            ait_ms = (
                summary[
                    "ait"
                ][
                    "median_of_medians_ms"
                ]
            )

            pt_ms = (
                summary[
                    "torch"
                ][
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

            same_thread_speedup = (
                pt_ms
                / ait_ms
            )

            scaling_summary[
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
                    same_thread_speedup,
            }

            log(
                f"{threads:>7} | "
                f"{ait_ms:>7.2f} | "
                f"{ait_scale:>9.3f}x | "
                f"{pt_ms:>7.2f} | "
                f"{pt_scale:>8.3f}x | "
                f"{same_thread_speedup:>6.3f}x"
            )

        result_data[
            "scaling_summary"
        ] = scaling_summary

        save_json(
            json_path,
            result_data,
        )

        log()
        log(
            "TXT log :",
            txt_path,
        )

        log(
            "JSON log:",
            json_path,
        )


if __name__ == "__main__":
    main()
