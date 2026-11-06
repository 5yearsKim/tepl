#!/usr/bin/env python3
"""Validate and render the axis-only evaluation; no legacy baseline comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import re

if __package__:
    from .paths import HERE, ROOT, repo_path, implementation_dependencies
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.paths import HERE, ROOT, repo_path, implementation_dependencies


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(n, d):
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "—"


def validate(data, classification):
    if (
        data.get("protocol") != "axis_only_v1"
        or classification.get("protocol") != "axis_only_v1"
    ):
        raise ValueError("axis-only protocol required")
    for path, value in implementation_dependencies(classification["dependencies"]).items():
        if sha(repo_path(path)) != value:
            raise ValueError("stale classification dependency: " + path)
    expected = {r["case"]: r for r in classification["cases"]}
    rows = data["results"]
    if len(rows) != len(expected) or {r["case"] for r in rows} != set(expected):
        raise ValueError("results must cover each classified case once")
    for row in rows:
        source = expected[row["case"]]
        if any(row.get(k) != v for k, v in source.items()):
            raise ValueError("classification changed after measurement")
        if row["control"] and row["outcome"] in {"inferred", "verified"}:
            raise ValueError("negative control accepted")
        if not row["in_scope"]:
            if row["outcome"] != "outside_scope" or "check" in row:
                raise ValueError("outside-scope case was measured")
            continue
        if sha(repo_path(row["file"])) != row["file_sha256"]:
            raise ValueError("stale axis task")
        if row["outcome"] not in {"inferred", "verified"}:
            continue
        check = row.get("check", {})
        proof = check.get("stdout", "")
        expected_descriptors = len({f["node"] for f in row["axis_fields"]})
        if (
            check.get("returncode") != 0
            or proof.count("status=proved") != 1
            or "additional-guard: true" not in proof
            or not row.get("validated_json")
            or row.get("compile", {}).get("returncode") != 0
            or row.get("validation", {}).get("returncode") != 0
            or row.get("inferred_descriptors") != expected_descriptors
            or row.get("requested_axis_fields") != len(row["axis_fields"])
            or not re.fullmatch("[0-9a-f]{64}", row.get("artifact_sha256", ""))
            or (row["outcome"] == "inferred") != bool(row["axis_fields"])
        ):
            raise ValueError(
                "success requires proof, all axis holes, compilation, and validation"
            )
    for path, value in implementation_dependencies(data["dependencies"]).items():
        if sha(repo_path(path)) != value:
            raise ValueError("stale synthesis dependency: " + path)


def render(data, classification):
    rows = [r for r in data["results"] if not r["control"]]
    scoped = [r for r in rows if r["in_scope"]]
    axis = [r for r in scoped if r["task"] == "axis_inference"]
    inferred = [r for r in axis if r["outcome"] == "inferred"]
    free = [r for r in scoped if r["task"] == "attribute_free"]
    supplied = [r for r in scoped if r["task"] == "supplied_non_axis"]
    success = [r for r in scoped if r["outcome"] in {"inferred", "verified"}]
    fields = sum(len(r["axis_fields"]) for r in axis)
    resolved = sum(len(r["axis_fields"]) for r in inferred)
    out = [
        "# XLA Axis-Only Attribute Inference Evaluation",
        "",
        f"Axis-field inference succeeds on **{fraction(len(inferred),len(axis))} cases**. "
        f"The run resolves all requested axis fields in those cases: **{resolved}/{fields} fields**. "
        "The field total counts only fully verified cases; unresolved cases may contain unaccepted partial candidates.",
        "",
        "One canonical migrated rule is evaluated per in-scope case. Every eligible original RHS axis field is `auto`; "
        "extent, coordinate, and other fields stay supplied. There is no explicit-baseline comparison, descriptor erasure, "
        "partial-auto search, or legacy non-axis inference. Original source conditions are retained.",
        "",
        "## Measured grouping",
        "",
        "```text",
        f"{len(rows)} XLA rewrite cases",
        f"├─ {len(scoped)} within the supported evaluation scope",
        f"│  ├─ {len(free)} with no RHS attributes",
        f"│  ├─ {len(supplied)} with only supplied non-axis attributes",
        f"│  └─ {len(axis)} axis-inference cases",
        f"│     ├─ {len(inferred)} inferred and verified",
        f"│     └─ {len(axis)-len(inferred)} unresolved",
        f"└─ {len(rows)-len(scoped)} outside the supported evaluation scope",
        "```",
        "",
        "## Outcomes",
        "",
        "| Task | Cases | Verified success |",
        "|---|---:|---:|",
        f"| Axis inference | {len(axis)} | {fraction(len(inferred),len(axis))} |",
        f'| No RHS attributes | {len(free)} | {fraction(sum(r["outcome"]=="verified" for r in free),len(free))} |',
        f'| Supplied non-axis attributes only | {len(supplied)} | {fraction(sum(r["outcome"]=="verified" for r in supplied),len(supplied))} |',
        f"| All in-scope cases | {len(scoped)} | {fraction(len(success),len(scoped))} |",
        "",
        "Success requires an explicit semantic proof with `additional-guard: true`, all requested original holes resolved, "
        "successful executable compilation, and validated JSON without unresolved holes. Any correct inferred assignment is accepted. "
        "An unproved rule is unresolved, not evidence of infeasibility. The negative control is checked separately and must fail.",
        "",
        "## Normalization and implementation scope",
        "",
        "Unit-only reshape is normalized by a shape-directed adapter to insertion broadcast and singleton-axis removal. "
        "Adapter-generated axis positions are fixed; they are not counted as synthesis targets. The removal adapter preserves "
        "element types without adding HLO reduce_sum's numeric-dtype guard. Every accepted reshape has a known ordered input "
        "shape and target shape with identical non-unit coordinates, independently of automatic axis choices.",
        "",
        "The audit found three additional axis cases and one verification-only case through unit normalization. "
        "`catalog_dot_drop_unit_dims` is excluded: once the dot attributes are automatic, some legal intermediate shapes "
        "would require non-unit splitting/merging in the final reshape. Its original handwritten attributes alone do not justify "
        "normalizing the complete inference domain. This revises the earlier estimate of 33 axis cases to 32.",
        "",
        "`catalog_broadcast_divide_broadcast` has no nonzero guard in the measured rule. "
        "Its rank-zero broadcast has a forced empty mapping and is reported separately from nontrivial axis choices. "
        "The reciprocal normalization requires exact-real arithmetic and a nonzero divisor.",
        "",
        "## Unresolved axis cases",
        "",
        "| Case | Reason |",
        "|---|---|",
    ]
    for r in axis:
        if r["outcome"] == "inferred":
            continue
        text = r.get("check", {}).get("stdout", "")
        reason = next(
            (
                line.strip().removeprefix("reason: ")
                for line in text.splitlines()
                if line.strip().startswith("reason:")
            ),
            r["outcome"],
        )
        out.append("| " + r["case"] + " | " + reason.replace("|", "\\|") + " |")
    forced = [r for r in axis if all(f["forced"] for f in r["axis_fields"])]
    out += [
        "",
        f'Forced-only tasks: **{len(forced)}**; verified: **{sum(r["outcome"]=="inferred" for r in forced)}**. '
        f'Nontrivial tasks: **{len(axis)-len(forced)}**; verified: **{sum(r["outcome"]=="inferred" for r in axis if r not in forced)}**.',
        "",
        "## Reproduction and evidence",
        "",
        "```sh",
        "bazel build //:tepl",
        "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/synthesis_cases.py",
        "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/run_synthesis.py --jobs 4 --timeout 30",
        "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/synthesis_report.py",
        "```",
        "",
        "[Classification](../data/synthesis_classification.json), [measurements](../results/synthesis.json), "
        "[per-case outcomes](synthesis_cases.md), and [canonical axis tasks](../../tepl/rules/axis/) "
        "record source/task hashes, normalization metadata, requested fields, diagnostics, compile/validation evidence, "
        "search-path labels, and timings.",
        "",
        f'- Run UTC: `{data["run_at_utc"]}`.',
        f'- Compiler SHA-256: `{data["compiler_sha256"]}`.',
        f'- Budget: {data["timeout_seconds_per_command"]:g} seconds per compiler command; {data["jobs"]} workers.',
        "- Arithmetic policy: ideal `algebraic`; no strict IEEE/backend equivalence claim.",
        "",
    ]
    return "\n".join(out)


def case_table(data):
    out = [
        "# Axis-Only Case Outcomes",
        "",
        "| Case | Task | Axis fields | Normalized | Outcome |",
        "|---|---|---:|---|---|",
    ]
    for r in data["results"]:
        name = r["case"] + (" (negative control)" if r["control"] else "")
        out.append(
            f'| {name} | {r["task"]} | {len(r["axis_fields"])} | {"yes" if r["normalizations"] else "no"} | {r["outcome"]} |'
        )
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = read(HERE / "results/synthesis.json")
    classification = read(HERE / "data/synthesis_classification.json")
    validate(data, classification)
    for name, text in [
        ("attribute_synthesis.md", render(data, classification)),
        ("synthesis_cases.md", case_table(data)),
    ]:
        path = HERE / "reports" / name
        if args.check:
            if path.read_text() != text:
                raise ValueError("stale synthesis report: " + name)
        else:
            path.write_text(text)
    print("Validated", len(data["results"]), "axis-only records.")


if __name__ == "__main__":
    main()
