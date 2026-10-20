#!/usr/bin/env python3
"""Recheck explicit/full-auto coverage after reasoning fixes; preserve the baseline."""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pilot
import run_corpus

ROOT, HERE = pilot.ROOT, pilot.HERE
BASELINE = HERE / "results/corpus.json"
RESULT = HERE / "results/reasoning_fixes.json"
REPORT = HERE / "reports/reasoning_fixes.md"


def coverage(data):
    primary = [c for c in data["cases"] if c["attribute_scope"] == "axis_extent"]
    lookup = {(r["case"], r["mode"]): r for r in data["results"] if not r["control"]}
    explicit = {c["name"] for c in primary if lookup[c["name"], "explicit"]["status"] == "proved"}
    full = {c["name"] for c in primary if lookup[c["name"], "full_auto"]["status"] == "proved"}
    identity = {c["name"] for c in primary if run_corpus.identity_witness(c, lookup[c["name"], "full_auto"])}
    return primary, lookup, explicit, full, identity


def report(data):
    primary, lookup, explicit, full, identity = coverage(data)
    _, old_lookup, old_explicit, old_full, old_identity = coverage(data["baseline"])
    stage = data.get("stage", "reasoning_fixes")
    if stage == "slice_frame_inference":
        title = "Shared-frame slice inference: XLA attribute-inference recheck"
        change = ("A single non-reduced output domain now retains its coordinate frame when "
                  "that frame overlaps an input. This lets existing affine index normalization "
                  "match a composed zero-start, unit-stride slice to its inferred replacement. "
                  "Completed rules are verified in a fresh context.")
        baseline_name = "reverse_inference"
        baseline_label = "The previous"
    elif stage == "reverse_inference":
        title = "Same-frame reverse inference: XLA attribute-inference recheck"
        change = ("The compiler now binds an unknown reverse selector to the normalized "
                  "source selector after matching coordinate identities. Nested reversals in one "
                  "frame normalize to parity; a cancelled reversal supplies the empty selector. "
                  "Completed rules are verified in a fresh context.")
        baseline_name = "reasoning_fixes"
        baseline_label = "The previous"
    else:
        title = "General reasoning fixes: XLA attribute-inference recheck"
        change = ("The compiler now propagates uniform-vector values through arithmetic and selections, "
                  "eliminates length-compatible unit scaling/zero offsets, and normalizes zero-cardinality "
                  "coordinate groups before preserving registered views. No operation names or benchmark "
                  "names participate in these algorithms.")
        baseline_name = "corpus"
        baseline_label = "The original"
    lines = ["# " + title, "", data["run_at_utc"], "", change, "",
             "Uses the original frozen 42-case manifest and identical generated explicit/full-auto inputs: "
             "38 primary axis/extent cases and four supplementary scalar-setting cases. Original rules, "
             "guards and dialect semantics are unchanged. Assistance is not rerun in this comparison. "
             f"{baseline_label} results/{baseline_name}.json is preserved, with its hash recorded in the new evidence.", "",
             "| Measurement | Before | After |", "|---|---:|---:|",
             f"| Primary explicit proofs | {len(old_explicit)}/{len(primary)} | {len(explicit)}/{len(primary)} |",
             f"| Primary full-auto proofs | {len(old_full)}/{len(primary)} | {len(full)}/{len(primary)} |",
             f"| Full auto within the original explicit-proved cohort | {len(old_full & old_explicit)}/{len(old_explicit)} | {len(full & old_explicit)}/{len(old_explicit)} |",
             f"| Full-auto proofs excluding detected identity witnesses | {len(old_full - old_identity)}/{len(primary)} | {len(full - identity)}/{len(primary)} |", "",
             "## Changed outcomes", "", "| Case | Variant | Before | After |", "|---|---|---|---|"]
    for case in data["cases"]:
        name = case["name"]
        for mode in ("explicit", "full_auto"):
            before, after = old_lookup[name, mode]["status"], lookup[name, mode]["status"]
            if before != after:
                lines.append(f"| {name} | {mode} | {before} | {after} |")
    regressions = [(n, m) for (n, m), row in lookup.items()
                   if old_lookup[n, m]["status"] == "proved" and row["status"] != "proved"]
    control = next(r for r in data["results"] if r["control"])
    lines += ["", f"Previously proved explicit/full-auto variants lost: **{len(regressions)}**. "
              f"Incorrect permutation-order control: **{control['status']}**.", "",
              "## All explicit/full-auto outcomes", "",
              "| Case | Cohort | Explicit | Full auto |", "|---|---|---|---|"]
    for case in data["cases"]:
        name = case["name"]
        lines.append(f"| {name} | {case['attribute_scope']} | {lookup[name, 'explicit']['status']} | {lookup[name, 'full_auto']['status']} |")
    lines += ["", "Raw stdout, diagnostics, constructors, source/input hashes and compiler hash are in "
              f"[{stage}.json](../results/{stage}.json). "
              "An unproved case is not an impossibility result. The detected singleton-stride identity "
              "witness is still reported separately from optimization coverage. "
              "These are proof-coverage measurements, not timing or complete XLA/IEEE equivalence results.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stage", choices=("reasoning_fixes", "reverse_inference",
                                            "slice_frame_inference"),
                        default="reasoning_fixes")
    args = parser.parse_args()
    pilot.require(args.timeout > 0, "timeout must be positive")
    compiler = args.compiler.resolve(strict=True)
    baseline_path = {"reasoning_fixes": BASELINE,
                     "reverse_inference": RESULT,
                     "slice_frame_inference": HERE / "results/reverse_inference.json"}[args.stage]
    result_path = HERE / f"results/{args.stage}.json"
    report_path = HERE / f"reports/{args.stage}.md"
    if args.check:
        data = json.loads(result_path.read_text())
        pilot.require(pilot.digest(compiler) == data["compiler_sha256"], "compiler changed")
        for name, expected in data["dependencies"].items():
            pilot.require(pilot.digest(ROOT / name) == expected, "changed dependency: " + name)
        pilot.require(report_path.read_text() == report(data), "recheck report is stale")
        print("Recheck evidence hashes and report are current.")
        return
    baseline = json.loads(baseline_path.read_text())
    manifest = json.loads(run_corpus.MANIFEST.read_text())
    pilot.require(manifest == run_corpus.build_manifest(), "frozen manifest changed")
    pilot.require(manifest["cases"] == baseline["cases"], "baseline cohort differs")
    jobs = [j for j in run_corpus.prepare(manifest)
            if j["mode"] in ("explicit", "full_auto", "incorrect_control")]
    for job in jobs:
        pilot.require(pilot.digest(ROOT / job["file"]) == baseline["dependencies"][job["file"]],
                      "input differs from baseline: " + job["file"])
    paths = pilot.dependency_paths(jobs, manifest) + [baseline_path, run_corpus.MANIFEST,
            run_corpus.CLASSIFICATION, Path(run_corpus.__file__).resolve(), Path(__file__).resolve()]
    dependencies = {str(p.relative_to(ROOT)): pilot.digest(p) for p in paths}
    compiler_hash = pilot.digest(compiler)
    rows = []
    for job in jobs:
        row = pilot.run_job(compiler, job, args.timeout)
        rows.append(row)
        print(f"{row['case']} / {row['mode']}: {row['status']}", flush=True)
    pilot.require(pilot.digest(compiler) == compiler_hash and
                  all(pilot.digest(ROOT / p) == h for p, h in dependencies.items()),
                  "compiler or dependencies changed during recheck")
    data = {"schema_version": 1, "stage": args.stage,
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "tepl_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines(),
            "compiler_sha256": compiler_hash, "environment": platform.platform(),
            "timeout_seconds_per_variant": args.timeout, "dependencies": dependencies,
            "cases": manifest["cases"], "results": rows,
            "baseline": {"cases": baseline["cases"], "results": [
                {k: r[k] for k in ("case", "mode", "control", "status", "constructors")}
                for r in baseline["results"]]}}
    result_path.write_text(json.dumps(data, indent=2) + "\n")
    report_path.write_text(report(data))
    pilot.require(all(r["status"] not in ("error", "timeout") for r in rows), "execution error; inspect evidence")
    pilot.require(next(r for r in rows if r["control"])["status"] == "not_proved", "control was not rejected")
    print(f"Saved reports/{args.stage}.md; baseline results are unchanged.")


if __name__ == "__main__":
    main()
