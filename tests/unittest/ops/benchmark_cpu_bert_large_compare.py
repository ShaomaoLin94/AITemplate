import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time


SCRIPT = "tests/unittest/ops/benchmark_cpu_bert_large_isolated.py"
DEFAULT_ROUNDS = 7


def _read_cpu_stat():
    result = {}

    with open("/proc/stat", "r") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue

            parts = line.split()
            name = parts[0]

            if not name[3:].isdigit():
                continue

            cpu = int(name[3:])
            values = [int(x) for x in parts[1:]]

            while len(values) < 8:
                values.append(0)

            user, nice, system, idle, iowait, irq, softirq, steal = values[:8]

            idle_all = idle + iowait
            total = sum(values)

            result[cpu] = (total, idle_all)

    return result


def _parse_cpu_list(text):
    cpus = set()

    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))

    return cpus


def _siblings(cpu):
    path = (
        f"/sys/devices/system/cpu/cpu{cpu}/"
        "topology/thread_siblings_list"
    )

    try:
        with open(path, "r") as f:
            return _parse_cpu_list(f.read())
    except OSError:
        return {cpu}


def choose_quiet_cpu(sample_seconds=0.50):
    allowed = sorted(os.sched_getaffinity(0))

    if len(allowed) == 1:
        return allowed[0], {allowed[0]: 0.0}

    before = _read_cpu_stat()
    time.sleep(sample_seconds)
    after = _read_cpu_stat()

    loads = {}

    for cpu in allowed:
        if cpu not in before or cpu not in after:
            loads[cpu] = 1.0
            continue

        total0, idle0 = before[cpu]
        total1, idle1 = after[cpu]

        dt = total1 - total0
        didle = idle1 - idle0

        if dt <= 0:
            loads[cpu] = 1.0
        else:
            loads[cpu] = max(
                0.0,
                min(1.0, 1.0 - didle / dt),
            )

    allowed_set = set(allowed)

    def score(cpu):
        sibs = _siblings(cpu) & allowed_set
        if not sibs:
            sibs = {cpu}

        sibling_load = sum(
            loads.get(sib, 1.0)
            for sib in sibs
        )

        # Avoid CPU0 when another allowed logical CPU exists because
        # CPU0 commonly handles more kernel/IRQ housekeeping.
        cpu0_penalty = 0.25 if cpu == 0 else 0.0

        return sibling_load + cpu0_penalty

    selected = min(
        allowed,
        key=score,
    )

    return selected, loads


def read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


def cpu_frequency_khz(cpu):
    return read_text(
        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
    )


def cpu_governor(cpu):
    return read_text(
        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    )


def cpu_pressure():
    text = read_text("/proc/pressure/cpu")
    if text is None:
        return None

    first = text.splitlines()[0]
    return first


def run_once(mode, cpu):
    freq_before = cpu_frequency_khz(cpu)
    pressure_before = cpu_pressure()

    proc = subprocess.run(
        ["python3", SCRIPT, mode],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    freq_after = cpu_frequency_khz(cpu)
    pressure_after = cpu_pressure()

    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise RuntimeError(
            f"{mode} benchmark failed with code {proc.returncode}"
        )

    result = None

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            result = json.loads(
                line.split(":", 1)[1].strip()
            )
            break

    if result is None:
        print(proc.stdout, end="")
        raise RuntimeError(
            f"No RESULT_JSON from {mode} benchmark"
        )

    result["freq_before_khz"] = freq_before
    result["freq_after_khz"] = freq_after
    result["pressure_before"] = pressure_before
    result["pressure_after"] = pressure_after

    return result


def median(values):
    return statistics.median(values)


def mean(values):
    return statistics.mean(values)


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(name, records):
    medians = [r["median_ms"] for r in records]
    means = [r["mean_ms"] for r in records]
    rss = [r["vmrss_mib"] for r in records]
    peaks = [r["peak_rss_mib"] for r in records]

    med_mean = mean(medians)

    result = {
        "median_of_medians_ms": median(medians),
        "mean_of_medians_ms": med_mean,
        "stdev_of_medians_ms": stdev(medians),
        "cv_of_medians_pct": (
            stdev(medians) / med_mean * 100.0
            if med_mean > 0.0
            else 0.0
        ),
        "min_median_ms": min(medians),
        "max_median_ms": max(medians),
        "median_mean_ms": median(means),
        "median_vmrss_mib": median(rss),
        "min_vmrss_mib": min(rss),
        "max_vmrss_mib": max(rss),
        "median_peak_rss_mib": median(peaks),
    }

    print()
    print(f"===== {name} aggregate =====")
    print(
        "median of medians ms :",
        result["median_of_medians_ms"],
    )
    print(
        "mean of medians ms   :",
        result["mean_of_medians_ms"],
    )
    print(
        "stdev medians ms     :",
        result["stdev_of_medians_ms"],
    )
    print(
        "CV medians %         :",
        result["cv_of_medians_pct"],
    )
    print(
        "median range ms       :",
        result["min_median_ms"],
        "..",
        result["max_median_ms"],
    )
    print(
        "median of means ms   :",
        result["median_mean_ms"],
    )
    print(
        "VmRSS range MiB      :",
        result["min_vmrss_mib"],
        "..",
        result["max_vmrss_mib"],
    )
    print(
        "median VmRSS MiB     :",
        result["median_vmrss_mib"],
    )
    print(
        "median Peak MiB      :",
        result["median_peak_rss_mib"],
    )

    return result


def bootstrap_median_ci(values, samples=20000):
    rng = random.Random(20260902)
    n = len(values)

    if n == 1:
        return values[0], values[0]

    estimates = []

    for _ in range(samples):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(
            statistics.median(sample)
        )

    estimates.sort()

    lo_idx = int(0.025 * (samples - 1))
    hi_idx = int(0.975 * (samples - 1))

    return estimates[lo_idx], estimates[hi_idx]


def _fmt_freq(value):
    if value is None:
        return "n/a"

    try:
        return f"{int(value) / 1000.0:.0f}MHz"
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "rounds",
        nargs="?",
        type=int,
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=None,
        help=(
            "Pin all benchmark children to this logical CPU. "
            "Default: auto-select a quiet allowed CPU."
        ),
    )

    args = parser.parse_args()

    if args.rounds < 5:
        raise SystemExit(
            "rounds must be >= 5 for a stability comparison"
        )

    allowed = sorted(os.sched_getaffinity(0))

    if args.cpu is None:
        cpu, sampled_loads = choose_quiet_cpu()
    else:
        if args.cpu not in allowed:
            raise SystemExit(
                f"CPU {args.cpu} is not in allowed affinity {allowed}"
            )
        cpu = args.cpu
        sampled_loads = {}

    # The isolated child inherits this one-CPU affinity. Its existing
    # allowed[0] pinning therefore resolves to the same chosen CPU.
    os.sched_setaffinity(0, {cpu})

    results = {
        "ait": [],
        "torch": [],
    }

    paired_speedups = []

    print("===== BERT-large stability-aware comparison =====")
    print("rounds              :", args.rounds)
    print("selected CPU        :", cpu)
    print("CPU governor        :", cpu_governor(cpu))
    print(
        "initial CPU freq    :",
        _fmt_freq(cpu_frequency_khz(cpu)),
    )

    if sampled_loads:
        print(
            "sampled CPU load    :",
            f"{sampled_loads.get(cpu, 0.0) * 100.0:.1f}%",
        )

    print("order               : alternating AIT/PT and PT/AIT")
    print(
        "decision basis      : paired speedup + bootstrap CI + jitter"
    )

    for round_idx in range(args.rounds):
        order = (
            ("ait", "torch")
            if round_idx % 2 == 0
            else ("torch", "ait")
        )

        print()
        print(
            f"===== Round {round_idx + 1}/{args.rounds}: "
            f"{order[0]} -> {order[1]} ====="
        )

        round_result = {}

        for mode in order:
            result = run_once(mode, cpu)

            results[mode].append(result)
            round_result[mode] = result

            inner_spread = (
                result["max_ms"]
                - result["min_ms"]
            )

            print(
                f"{mode:<5} "
                f"median={result['median_ms']:.3f} ms  "
                f"mean={result['mean_ms']:.3f} ms  "
                f"min={result['min_ms']:.3f}  "
                f"max={result['max_ms']:.3f}  "
                f"spread={inner_spread:.1f} ms  "
                f"RSS={result['vmrss_mib']:.1f} MiB  "
                f"freq={_fmt_freq(result['freq_after_khz'])}"
            )

        pair_speedup = (
            round_result["torch"]["median_ms"]
            / round_result["ait"]["median_ms"]
        )

        paired_speedups.append(pair_speedup)

        print(
            f"paired AIT speedup: {pair_speedup:.5f}x"
        )

    ait = summarize(
        "AITemplate",
        results["ait"],
    )

    pt = summarize(
        "PyTorch",
        results["torch"],
    )

    aggregate_speedup = (
        pt["median_of_medians_ms"]
        / ait["median_of_medians_ms"]
    )

    paired_median = median(
        paired_speedups
    )

    ci_lo, ci_hi = bootstrap_median_ci(
        paired_speedups
    )

    latency_gap_ms = (
        ait["median_of_medians_ms"]
        - pt["median_of_medians_ms"]
    )

    latency_gap_pct = (
        latency_gap_ms
        / pt["median_of_medians_ms"]
        * 100.0
    )

    rss_ratio = (
        ait["median_vmrss_mib"]
        / pt["median_vmrss_mib"]
    )

    peak_ratio = (
        ait["median_peak_rss_mib"]
        / pt["median_peak_rss_mib"]
    )

    jitter_high = (
        ait["cv_of_medians_pct"] > 2.5
        or pt["cv_of_medians_pct"] > 2.5
    )

    rss_anomaly = (
        pt["min_vmrss_mib"]
        < 0.90 * pt["median_vmrss_mib"]
        or ait["min_vmrss_mib"]
        < 0.90 * ait["median_vmrss_mib"]
    )

    print()
    print("===== Stability-aware result =====")
    print(
        "AIT median-of-medians ms:",
        ait["median_of_medians_ms"],
    )
    print(
        "PT median-of-medians ms :",
        pt["median_of_medians_ms"],
    )
    print(
        "aggregate AIT speedup   :",
        aggregate_speedup,
    )
    print(
        "paired speedup median   :",
        paired_median,
    )
    print(
        "paired speedup range    :",
        min(paired_speedups),
        "..",
        max(paired_speedups),
    )
    print(
        "paired median 95% CI    :",
        ci_lo,
        "..",
        ci_hi,
    )
    print(
        "latency gap %           :",
        latency_gap_pct,
    )
    print(
        "VmRSS ratio AIT/PT      :",
        rss_ratio,
    )
    print(
        "Peak ratio AIT/PT       :",
        peak_ratio,
    )

    if jitter_high:
        print(
            "JITTER: high process-to-process variance; "
            "do not call the result stable yet"
        )
    else:
        print(
            "JITTER: process-to-process variance is controlled"
        )

    if rss_anomaly:
        print(
            "RSS: anomalous low-RSS run detected; "
            "memory result should use the median, not that round"
        )

    if not jitter_high and ci_lo > 1.0:
        print(
            "PERF: AITemplate is measurably faster in this run set"
        )
    elif not jitter_high and ci_hi < 1.0:
        print(
            "PERF: PyTorch is measurably faster in this run set"
        )
    else:
        print(
            "PERF: current noise/CI overlaps parity; "
            "treat AIT and PyTorch as tied"
        )

    if peak_ratio > 1.25:
        print(
            "MEMORY: investigate AIT peak RSS"
        )
    else:
        print(
            "MEMORY: AIT peak remains controlled"
        )

    print()
    print(
        "CPU pressure at end:",
        cpu_pressure(),
    )

    print(
        "RESULT_JSON:",
        json.dumps(
            {
                "rounds": args.rounds,
                "cpu": cpu,
                "ait": ait,
                "torch": pt,
                "aggregate_speedup": aggregate_speedup,
                "paired_speedup_median": paired_median,
                "paired_speedup_min": min(
                    paired_speedups
                ),
                "paired_speedup_max": max(
                    paired_speedups
                ),
                "paired_speedup_ci95_low": ci_lo,
                "paired_speedup_ci95_high": ci_hi,
                "latency_gap_ms": latency_gap_ms,
                "latency_gap_pct": latency_gap_pct,
                "vmrss_ratio": rss_ratio,
                "peak_ratio": peak_ratio,
                "jitter_high": jitter_high,
                "rss_anomaly": rss_anomaly,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
