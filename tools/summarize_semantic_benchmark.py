#!/usr/bin/env python3
"""Summarize native solver measurements; timings never serve as test assertions."""
import argparse
import csv
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty measurements: {path}")
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["mode"]].append(row)
    return rows, groups


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def status(rows):
    return "/".join(sorted(set(row["status"] for row in rows)))


def fallback(rows):
    return f"{100 * sum(row['fallback'] == '1' for row in rows) / len(rows):.0f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    old_rows, before = load(args.before)
    new_rows, after = load(args.after)
    settings = ("timeout_ms", "budget", "max_candidates", "seed", "z3_version", "compiler", "build_mode")
    for key in before.keys() & after.keys():
        for field in settings + ("shapes", "known", "pins"):
            if {r[field] for r in before[key]} != {r[field] for r in after[key]}:
                raise ValueError(f"incomparable {key}: {field}")
    print("# Semantic constraint benchmark\n")
    print(f"Recorded {datetime.now(timezone.utc).isoformat()}. "
          f"Host: {platform.system()} {platform.machine()}.\n")
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                print(f"CPU: {line.split(':', 1)[1].strip()}.\n")
                break
    row = new_rows[0]
    print(f"Z3: {row['z3_version']}; compiler: {row['compiler']} ({row['build_mode']}). "
          f"Timeout: {row['timeout_ms']} ms; work budget: {row['budget']}; "
          f"controlled candidate limit: {row['max_candidates']}; seed: {row['seed']}.\n")
    print("Times are medians of repeated runs, in milliseconds. Rules are parsed "
          "and analyzed once per distinct rule per invocation; compilation is excluded "
          "from matching time. Both controlled modes disable ordered inference and "
          "candidate preferences. Production uses normal dispatch and has no separate "
          "candidate-count cap. These measure the native solver, excluding Rust graph "
          "matching and ABI serialization.\n")
    print("Metadata time includes source-legality Z3 calls; phase times overlap. "
          "Unknown outcomes include resource limits and are retained in the tables. "
          "Timing ratios are shown only when both modes consistently return the "
          "same conclusive status. This is a local benchmark, not a universal "
          "performance guarantee.\n")
    print("## Constraint-generation extensions\n")
    print("| Case | Before result | After result | Before ms | After ms | "
          "Before/after solver calls | Before/after rejected candidates | "
          "Before/after fallback | After method |")
    print("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for (case, mode), rows in after.items():
        old = before.get((case, mode))
        if mode != "compiled" or not old:
            continue
        methods = "/".join(sorted(set(r["method"] for r in rows)))
        print(f"| {case} | {status(old)} | {status(rows)} | "
              f"{median(old, 'total_ms'):.3f} | {median(rows, 'total_ms'):.3f} | "
              f"{median(old, 'solver_calls'):g}/{median(rows, 'solver_calls'):g} | "
              f"{median(old, 'rejected_candidates'):g}/{median(rows, 'rejected_candidates'):g} | "
              f"{fallback(old)}/{fallback(rows)} | {methods} |")
    print("\n## Current metadata versus semantic constraints\n")
    print("| Case | Metadata result | Compiled result | Metadata ms | Compiled ms | "
          "Metadata/compiled time ratio | Construction ms | Z3 ms | Verification ms |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (case, mode), rows in after.items():
        base = after.get((case, "metadata"))
        if mode != "compiled" or not base:
            continue
        a, b = status(base), status(rows)
        ratio = f"{median(base, 'total_ms') / max(median(rows, 'total_ms'), 1e-9):.2f}"
        if a != b or a not in ("found", "invalid"):
            ratio = "—"
        print(f"| {case} | {a} | {b} | {median(base, 'total_ms'):.3f} | "
              f"{median(rows, 'total_ms'):.3f} | {ratio} | "
              f"{median(rows, 'constraints_ms'):.3f} | {median(rows, 'z3_ms'):.3f} | "
              f"{median(rows, 'verification_ms'):.3f} |")
    print("\n## Production dispatch\n")
    print("| Case | Result | Method | Match ms | Ordered admission/construction ms | Fallback |")
    print("| --- | --- | --- | ---: | ---: | --- |")
    for (case, mode), rows in after.items():
        if mode == "production":
            methods = "/".join(sorted(set(r["method"] for r in rows)))
            print(f"| {case} | {status(rows)} | {methods} | "
                  f"{median(rows, 'total_ms'):.3f} | {median(rows, 'ordered_ms'):.3f} | {fallback(rows)} |")


if __name__ == "__main__":
    main()
