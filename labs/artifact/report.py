#!/usr/bin/env python3
"""Collect validated run summaries without changing measured outcomes."""
import argparse
import json
from pathlib import Path
import shutil

import run as artifact
import source_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a new or empty report directory")
    measured_coverage = artifact.validate_run(run / "coverage")
    names = ["coverage/summary.json", "coverage/provenance.json",
             "einsum-report/summary.json", "einsum-report/provenance.json",
             "source-size/results.json", "source-size/verification.json"]
    names += [n for n in ("scaling/performance_summary.json", "scaling/performance.json",
                           "algebraic-tests/results.json") if (run / n).exists()]
    data = {name: json.loads((run / name).read_text()) for name in names}
    coverage = data["coverage/summary.json"]
    einsum = data["einsum-report/summary.json"]["total"]
    size = data["source-size/results.json"]
    accepted = [r for r in measured_coverage["records"] if not r.get("control")]
    for key, value in dict(cases=len(accepted), accepted=sum(r["accepted"] for r in accepted),
                           axis_tasks=sum(bool(r["axis_fields"]) for r in accepted),
                           axis_accepted=sum(r["accepted"] and bool(r["axis_fields"]) for r in accepted)).items():
        if coverage[key] != value:
            raise ValueError("coverage summary differs from measured records: " + key)
    recounted, _ = source_size.measure()
    if any(size[key] != recounted[key] for key in ("matched", "all_tepl", "matched_cases", "all_tepl_cases")):
        raise ValueError("source-size summary differs from source recount")
    summary = dict(
        schema_version=1,
        evidence_hashes={name: artifact.sha(run / name) for name in names},
        compiler_sha256=data["coverage/provenance.json"]["compiler_sha256"],
        coverage={key: coverage[key] for key in ("cases", "accepted", "axis_tasks", "axis_accepted", "changed")},
        einsum={key: einsum[key] for key in ("cases", "represented", "verified", "nonidentical", "axis_positions")},
        source_size=dict(matched_cases=size["matched_cases"], all_tepl_cases=size["all_tepl_cases"],
                         matched_tokens={key: sum(value.values()) for key, value in size["matched"].items()},
                         all_tepl_tokens={key: sum(value.values()) for key, value in size["all_tepl"].items()}),
        whole_pass_xla_integration_measured=False,
    )
    if "algebraic-tests/results.json" in data:
        value = data["algebraic-tests/results.json"]
        summary["xla_transpose_slice"] = {key: value[key] for key in ("scope", "cases", "numerical_input_sets", "binary_sha256")}
    if "scaling/performance_summary.json" in data:
        summary["scaling"] = data["scaling/performance_summary.json"]
        summary["scaling_provenance"] = {key: data["scaling/performance.json"][key] for key in
                                          ("cpu", "platform", "run_at_utc", "warmups", "repetitions", "fingerprint")}
    args.out.mkdir(parents=True, exist_ok=True)
    artifact.save(args.out / "results.json", summary)
    lines = ["# Recorded artifact results", "",
             "| Measurement | Result |", "|---|---|",
             f"| Modeled XLA coverage | {coverage['accepted']}/{coverage['cases']} accepted |",
             f"| XLA axis inference | {coverage['axis_accepted']}/{coverage['axis_tasks']} accepted |",
             f"| Einsum | {einsum['verified']} verified rewrites; {einsum['axis_positions']:,} inferred axis entries |",
             f"| Source size, {size['matched_cases']} matched rules | " +
             " / ".join(f"{sum(size['matched'][v].values()):,}" for v in ("xla", "explicit", "inferred")) + " tokens |", "",
             "Source-size totals for all 31 TEPL cases are recorded separately in the JSON. "
             "Current validation outcomes for explicit and inferred source forms remain in the raw run.", "",
             "The JSON records evidence hashes and compiler provenance. "
             "Whole-pass XLA integration and the four-workload experiment are not measured by these compiler results.", ""]
    if "xla_transpose_slice" in summary:
        value = summary["xla_transpose_slice"]
        lines += [f"The two-rule transpose integration passed {value['cases']} pass-level graph comparisons "
                  f"and {value['numerical_input_sets']} numerical input sets.", ""]
    if "scaling" in summary:
        largest = {panel: max((r for r in summary["scaling"] if r["panel"] == panel and r["successful"]),
                              key=lambda r: r["x"], default=None) for panel in ("a", "b", "c")}
        for panel, label, unit in (("a", "dots per side", "ms"), ("b", "additive terms", "ms"), ("c", "tensor rank", "µs")):
            row = largest[panel]
            if row:
                lines += [f"Largest successful {label}: {row['x']}; mean total {row['mean_total']:.1f} {unit}.", ""]
        for row in summary["scaling"]:
            if not row["successful"]:
                lines += [f"Panel {row['panel']}, size {row['x']} excluded: {row['reason']}.", ""]
        for suffix in ("png", "pdf", "svg"):
            source = run / "scaling/figures" / ("scalability." + suffix)
            if source.exists():
                shutil.copyfile(source, args.out / source.name)
        lines += ["![Scalability measurement](scalability.png)", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
