import argparse
import importlib.util
import json
import os
import subprocess
import sys


ISOLATED = (
    "tests/unittest/ops/"
    "benchmark_cpu_bert_large_isolated.py"
)

SELF = os.path.abspath(__file__)


def _read_int(path):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def physical_core_cpus():
    """
    Pick one logical CPU from each physical core.

    This avoids accidentally counting SMT siblings as separate
    physical cores during the initial scaling sweep.
    """
    allowed = sorted(
        os.sched_getaffinity(0)
    )

    representatives = {}

    for cpu in allowed:
        base = (
            f"/sys/devices/system/cpu/"
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
            key = ("logical", cpu)
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

    # Avoid CPU0 for the first benchmark core if alternatives exist.
    if 0 in cpus and len(cpus) > 1:
        cpus.remove(0)
        cpus.append(0)

    return cpus


def worker(
    mode,
    threads,
    cpus,
):
    if mode not in (
        "ait",
        "torch",
    ):
        raise SystemExit(
            "worker mode must be ait or torch"
        )

    if threads < 1:
        raise SystemExit(
            "threads must be positive"
        )

    if len(cpus) < threads:
        raise SystemExit(
            "not enough CPUs for worker"
        )

    cpu_set = set(
        cpus[:threads]
    )

    # Set the actual process affinity before importing
    # the existing isolated benchmark.
    real_sched_setaffinity = (
        os.sched_setaffinity
    )

    real_sched_setaffinity(
        0,
        cpu_set,
    )

    # AITemplate CPU backend.
    os.environ[
        "AIT_CPU_NUM_THREADS"
    ] = str(threads)

    # Keep PyTorch/BLAS thread configuration consistent.
    os.environ[
        "OMP_NUM_THREADS"
    ] = str(threads)

    os.environ[
        "MKL_NUM_THREADS"
    ] = str(threads)

    # Import torch only after environment variables are set.
    import torch

    real_set_num_threads = (
        torch.set_num_threads
    )

    real_set_num_threads(
        threads
    )

    # The existing isolated benchmark contains:
    #
    #   torch.set_num_threads(1)
    #
    # Ignore only that later override. The thread count above
    # remains active.
    def keep_requested_threads(
        ignored_value,
    ):
        return None

    torch.set_num_threads = (
        keep_requested_threads
    )

    # The isolated benchmark also narrows affinity to:
    #
    #   {allowed[0]}
    #
    # Prevent only that Python-level narrowing. The real process
    # affinity has already been set above.
    os.sched_setaffinity = (
        lambda pid, mask: None
    )

    sys.argv = [
        ISOLATED,
        mode,
    ]

    spec = (
        importlib.util.spec_from_file_location(
            "bert_large_thread_worker",
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

    print()
    print(
        "THREAD_SCALING mode   :",
        mode,
    )

    print(
        "THREAD_SCALING threads:",
        threads,
    )

    print(
        "THREAD_SCALING CPUs   :",
        sorted(cpu_set),
    )

    print(
        "THREAD_SCALING torch  :",
        torch.get_num_threads(),
    )

    module.main()


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
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if proc.returncode != 0:
        print(
            proc.stdout,
            end="",
        )

        raise RuntimeError(
            f"{mode} {threads}-thread "
            f"benchmark failed"
        )

    for line in reversed(
        proc.stdout.splitlines()
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

            return result

    print(
        proc.stdout,
        end="",
    )

    raise RuntimeError(
        "worker produced no RESULT_JSON"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "threads",
        nargs="*",
        type=int,
        help=(
            "thread counts to test; "
            "default: 1 2 4 8"
        ),
    )

    args = parser.parse_args()

    requested = (
        args.threads
        if args.threads
        else [1, 2, 4, 8]
    )

    if any(
        value < 1
        for value in requested
    ):
        raise SystemExit(
            "thread counts must be positive"
        )

    # Remove duplicates while keeping ascending order.
    requested = sorted(
        set(requested)
    )

    cpus = physical_core_cpus()

    if not cpus:
        raise SystemExit(
            "no CPUs available"
        )

    thread_counts = [
        value
        for value in requested
        if value <= len(cpus)
    ]

    if not thread_counts:
        raise SystemExit(
            "requested thread counts exceed "
            "available physical cores"
        )

    print(
        "===== BERT-large thread scaling ====="
    )

    print(
        "physical cores available:",
        len(cpus),
    )

    print(
        "representative CPUs     :",
        cpus,
    )

    print(
        "thread counts           :",
        thread_counts,
    )

    results = {}

    for index, threads in enumerate(
        thread_counts
    ):
        selected_cpus = (
            cpus[:threads]
        )

        print()
        print(
            "===== Threads:",
            threads,
            "CPUs:",
            selected_cpus,
            "====="
        )

        # Alternate order to reduce simple thermal/order bias.
        order = (
            ("ait", "torch")
            if index % 2 == 0
            else ("torch", "ait")
        )

        current = {}

        for mode in order:
            result = run_once(
                mode,
                threads,
                selected_cpus,
            )

            current[mode] = (
                result
            )

            print(
                f"{mode:<5} "
                f"median="
                f"{result['median_ms']:.3f} ms  "
                f"mean="
                f"{result['mean_ms']:.3f} ms  "
                f"min="
                f"{result['min_ms']:.3f}  "
                f"max="
                f"{result['max_ms']:.3f}"
            )

        results[threads] = (
            current
        )

        speedup = (
            current["torch"][
                "median_ms"
            ]
            / current["ait"][
                "median_ms"
            ]
        )

        print(
            "AIT/PT speedup:",
            f"{speedup:.5f}x",
        )

    baseline_threads = (
        thread_counts[0]
    )

    baseline_ait = (
        results[
            baseline_threads
        ]["ait"]["median_ms"]
    )

    baseline_torch = (
        results[
            baseline_threads
        ]["torch"]["median_ms"]
    )

    print()
    print(
        "===== Scaling summary ====="
    )

    print(
        "threads | AIT ms | AIT scale | "
        "PT ms | PT scale | AIT/PT"
    )

    best_ait_threads = None
    best_ait_ms = None

    for threads in thread_counts:
        ait_ms = (
            results[
                threads
            ]["ait"]["median_ms"]
        )

        torch_ms = (
            results[
                threads
            ]["torch"]["median_ms"]
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

        if (
            best_ait_ms is None
            or ait_ms < best_ait_ms
        ):
            best_ait_ms = ait_ms
            best_ait_threads = (
                threads
            )

        print(
            f"{threads:>7} | "
            f"{ait_ms:>7.2f} | "
            f"{ait_scale:>9.3f}x | "
            f"{torch_ms:>7.2f} | "
            f"{torch_scale:>8.3f}x | "
            f"{speedup:>6.3f}x"
        )

    best_result = (
        results[
            best_ait_threads
        ]
    )

    best_same_thread_speedup = (
        best_result["torch"][
            "median_ms"
        ]
        / best_result["ait"][
            "median_ms"
        ]
    )

    print()
    print(
        "best AIT thread count :",
        best_ait_threads,
    )

    print(
        "best AIT median ms    :",
        best_ait_ms,
    )

    print(
        "same-thread AIT/PT    :",
        best_same_thread_speedup,
    )

    print()
    print(
        "RESULT_JSON:",
        json.dumps(
            {
                "physical_core_cpus":
                    cpus,
                "thread_counts":
                    thread_counts,
                "baseline_threads":
                    baseline_threads,
                "best_ait_threads":
                    best_ait_threads,
                "best_ait_ms":
                    best_ait_ms,
                "best_same_thread_speedup":
                    best_same_thread_speedup,
                "results":
                    results,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    if (
        len(sys.argv) >= 2
        and sys.argv[1]
        == "__worker__"
    ):
        if len(sys.argv) != 5:
            raise SystemExit(
                "worker usage: "
                "__worker__ "
                "[ait|torch] "
                "threads cpu0,cpu1,..."
            )

        worker(
            sys.argv[2],
            int(sys.argv[3]),
            [
                int(value)
                for value
                in sys.argv[4].split(",")
                if value
            ],
        )

    else:
        main()
