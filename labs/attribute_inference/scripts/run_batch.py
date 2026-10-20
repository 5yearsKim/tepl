#!/usr/bin/env python3
"""Check only frozen batch_01 (five new cases), preserving the original pilot."""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pilot

HERE, ROOT = pilot.HERE, pilot.ROOT
MANIFEST = HERE / "data/batch_01.json"
RESULT = HERE / "results/batch_01.json"
REPORT = HERE / "reports/batch_01.md"


def prepare(manifest):
    jobs = []
    pilot.require(len(manifest["cases"]) == 5, "runner is limited to this five-case batch")
    for case in manifest["cases"]:
        source = ROOT / case["source"]
        pilot.require(pilot.digest(source) == case["source_sha256"], "source changed since selection")
        original = source.read_text()
        variants = [{"mode": "explicit"}, {"mode": "full_auto", "infer": list(case["rhs_descriptors"])}] + case["assisted"]
        for spec in variants:
            text, requested = (original, 0) if spec["mode"] == "explicit" else pilot.make_variant(original, case, spec)
            path = HERE / "tepl/batch_01" / case["name"] / (spec["mode"] + ".tepl")
            path.parent.mkdir(parents=True, exist_ok=True)
            relative = Path(os.path.relpath(pilot.DIALECT, path.parent)).as_posix()
            pilot.require(text.count('"dialects/hlo.tepl"') == 1, "unexpected import")
            text = text.replace('"dialects/hlo.tepl"', '"' + relative + '"')
            text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"
            path.write_text(text)
            jobs.append({"case": case["name"], "mode": spec["mode"],
                         "file": str(path.relative_to(ROOT)), "policy": case["policy"],
                         "control": False, "catalog_id": case["catalog_id"], "scope": case["scope"],
                         "requested_fields": requested, "metrics": pilot.code_metrics(text),
                         "expected_constructor_fields": [f for n in spec.get("infer", []) for f in case["rhs_descriptors"][n]]})
    return jobs


def report(data):
    rows = data["results"]
    by_case = {(r["case"], r["mode"]): r for r in rows}
    names = list(dict.fromkeys(r["case"] for r in rows))
    proved_baseline = {n for n in names if by_case[n, "explicit"]["status"] == "proved"}
    full_proved = {n for n in names if by_case[n, "full_auto"]["status"] == "proved"}
    assisted = [r for r in rows if r["mode"].startswith("assisted")]
    recovered = {r["case"] for r in assisted if r["status"] == "proved"} - full_proved
    lines = ["# Attribute inference: batch 01", "", data["selection"], "", data["assistance"], "",
             "These are five new cases, disjoint from the first pilot. Fractions below are within-batch outcomes, not corpus-wide coverage.", "",
             f"- Explicit versions proved: **{len(proved_baseline)}/{len(names)}**.",
             f"- Full-auto versions proved, all batch cases: **{len(full_proved)}/{len(names)}**.",
             f"- Full-auto versions proved, conditioned on current explicit proof: **{len(full_proved & proved_baseline)}/{len(proved_baseline)}**.",
             f"- Full-auto successes with an unproved explicit baseline: **{len(full_proved - proved_baseline)}**.",
             f"- Assisted variants proved: **{sum(r['status'] == 'proved' for r in assisted)}/{len(assisted)}**; additional recovered cases: **{len(recovered)}**.", "",
             "| Case | Explicit | Full auto | Assisted operation |",
             "|---|---|---|---|"]
    for name in names:
        extra = by_case.get((name, "assisted_operation"))
        lines.append(f"| {name} | {by_case[name, 'explicit']['status']} | {by_case[name, 'full_auto']['status']} | {extra['status'] if extra else 'not attempted'} |")
    lines += ["", "## Attribute code", "",
              "Same token proxy as the pilot: RHS annotations plus derivations. Conditions, restrictions and semantics are unchanged.",
              "Inferred-field counts exclude pins. Only proved variants receive a savings estimate.", "",
              "| Case | Variant | Requested fields | Attribute tokens | Reduction vs explicit |",
              "|---|---|---:|---:|---:|"]
    for r in rows:
        count = r["metrics"]["attribute_tokens"]
        baseline = by_case[r["case"], "explicit"]["metrics"]["attribute_tokens"]
        reduction = f"{baseline-count} ({100*(baseline-count)/baseline:.1f}%)" if r["status"] == "proved" else "unproved"
        lines.append(f"| {r['case']} | {r['mode']} | {r['requested_fields']} | {count} | {reduction} |")
    lines += ["", "## Successful constructors", ""]
    for r in rows:
        if r["status"] == "proved" and r["constructors"]:
            lines += [f"### {r['case']}: {r['mode']}", "", "```text"]
            lines += [f"{v['field']} = {v['expression']}" for v in r["constructors"]]
            lines += ["```", ""]
    lines += ["## Unproved cases", "", "A failed explicit baseline is permitted and reported; it is not grounds to remove the case.", ""]
    for r in rows:
        if r["status"] != "proved":
            lines += [f"### {r['case']}: {r['mode']}", "", "```text", r["stderr"].strip() or r["status"], "```", ""]
    lines += ["## Scope and next checkpoint", "",
              "Broadcast-add is a descriptor-reuse opportunity. Reduction composition, scalar-broadcast reversal and convolution scaling also test cases whose explicit baselines were previously unproved. Dot reassociation in this corpus is restricted to rank-two matrices; its constant attributes are not general-rank reconstruction evidence.",
              "Full-auto and assisted success require checked constructors and `additional-guard: true`. Unproved outcomes are not impossibility results.",
              "Original explicit source files, guards and operation semantics were not edited. No compiler changes were made for inference.",
              "The classifier is a static inventory, not an execution of the remaining corpus. CLI wall times are not a performance study.",
              "Independent numerical validation, general symbolic additions, and scaling are still pending.",
              "**Stop for user review before selecting or running the next batch.**", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    pilot.require(args.timeout > 0, "timeout must be positive")
    compiler = args.compiler.resolve(strict=True)
    if args.check:
        data = json.loads(RESULT.read_text())
        pilot.require(pilot.digest(compiler) == data["compiler_sha256"], "compiler changed")
        for p, expected in data["dependencies"].items():
            pilot.require(pilot.digest(ROOT / p) == expected, "stale batch input: " + p)
        pilot.require(REPORT.read_text() == report(data), "batch report is stale")
        print("Batch hashes and report are current; no TEPL checks executed.")
        return
    manifest = json.loads(MANIFEST.read_text())
    pilot.require(manifest["batch"] == "batch_01", "only batch_01 is authorized by this runner")
    jobs = prepare(manifest)
    paths = pilot.dependency_paths(jobs, manifest) + [MANIFEST, Path(__file__).resolve(),
                                                    HERE / "scripts/classify_corpus.py",
                                                    HERE / "results/corpus_classification.json"]
    dependencies = {str(p.relative_to(ROOT)): pilot.digest(p) for p in paths}
    compiler_hash = pilot.digest(compiler)
    rows = []
    for job in jobs:
        row = pilot.run_job(compiler, job, args.timeout)
        rows.append(row)
        print(f"{row['case']} / {row['mode']}: {row['status']}", flush=True)
    pilot.require(pilot.digest(compiler) == compiler_hash and all(pilot.digest(ROOT/p) == h for p, h in dependencies.items()),
                  "compiler or inputs changed during batch")
    data = {"schema_version": 1, "batch": manifest["batch"],
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "tepl_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines(),
            "compiler_sha256": compiler_hash, "environment": platform.platform(),
            "timeout_seconds_per_variant": args.timeout, "selection": manifest["selection"],
            "assistance": manifest["assistance"], "dependencies": dependencies, "results": rows}
    RESULT.write_text(json.dumps(data, indent=2) + "\n")
    REPORT.write_text(report(data))
    pilot.require(all(r["status"] not in ("error", "timeout") for r in rows), "infrastructure error or timeout; inspect raw evidence")
    print("Saved reports/batch_01.md; stop for user review.")


if __name__ == "__main__":
    main()
