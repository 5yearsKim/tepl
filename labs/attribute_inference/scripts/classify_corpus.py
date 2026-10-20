#!/usr/bin/env python3
"""Inventory the existing XLA corpus and freeze one five-case batch; no TEPL runs."""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pilot

HERE, ROOT = pilot.HERE, pilot.ROOT
COVERAGE = ROOT / "labs/xla_rewrite/coverage"
sys.path.insert(0, str(COVERAGE / "scripts"))
import hlo_support

RESULT = HERE / "results/corpus_classification.json"
REPORT = HERE / "reports/corpus_classification.md"
BATCH = HERE / "data/batch_01.json"
FAMILIES = ("broadcast", "reduction", "dot", "indexing", "convolution")
CHALLENGE_FAMILIES = {"reduction", "indexing"}


def classify():
    hlo_support.validate_support()
    operations = hlo_support.inventory()
    aliases = {alias: name for name, op in operations.items() for alias in [name, *op["aliases"]]}
    schemas = {}
    for name, op in operations.items():
        attrs = pilot.block(op["text"], "attrs")
        schemas[name] = dict(re.findall(r"\b(\w+)\s*:\s*([\w\[\]]+)\s*;", attrs[2])) if attrs else {}
        if attrs:
            pilot.require(len(schemas[name]) == attrs[2].count(";"), "unsupported schema syntax: " + name)
    manifest = json.loads((COVERAGE / "data/verification_cases.json").read_text())
    baseline = json.loads((COVERAGE / "results/verification.json").read_text())
    recorded = {r["case"]: r for r in baseline["results"]}
    catalog_path = COVERAGE.parent / "catalog/rules.json"
    catalog = {r["id"]: r for r in json.loads(catalog_path.read_text())["rules"]}
    rows = []
    for case in manifest["cases"]:
        path = (COVERAGE / case["file"]).resolve()
        text = path.read_text()
        start, end = pilot.rhs_span(text)
        rhs = pilot.uncomment(text[start:end])
        descriptors, occurrences = {}, []
        for match in re.finditer(r"\((?:hlo\.)?(\w+)(?:\s*\[@(\w+)\])?", rhs):
            op_name, descriptor = match.groups()
            pilot.require(op_name in aliases, "unknown RHS operation: " + op_name)
            name = aliases[op_name]
            fields = schemas[name]
            pilot.require(bool(fields) == bool(descriptor), "missing/unexpected explicit attributes: " + case["name"])
            if descriptor:
                pilot.require(descriptor not in descriptors or descriptors[descriptor] == fields,
                              "descriptor reused across different schemas")
                descriptors[descriptor] = fields
                occurrences.append({"operation": name, "descriptor": descriptor})
        required = hlo_support.required_operations(path)
        missing = [name for name in required if not operations[name]["semantic"]]
        field_types = [kind for schema in descriptors.values() for kind in schema.values()]
        scope = ("no_rhs_attributes" if not field_types else "axis_extent" if all(t == "index[]" for t in field_types)
                 else "other_attributes" if all(t != "index[]" for t in field_types) else "mixed_attributes")
        saved = recorded[case["name"]]
        rows.append({"name": case["name"], "catalog_id": case["catalog_id"],
                     "category": "control" if case["control"] else catalog[case["catalog_id"]]["category"],
                     "source": str(path.relative_to(ROOT)), "scope": case["scope"],
                     "policy": case["policy"], "control": case["control"],
                     "rhs_descriptors": descriptors, "rhs_occurrences": occurrences,
                     "required_operations": required, "missing_semantics": missing,
                     "attribute_scope": scope, "recorded_explicit_status": saved["status"],
                     "recorded_source_matches": pilot.digest(path) == saved["file_sha256"],
                     "source_sha256": pilot.digest(path)})
    dependencies = [pilot.MANIFEST, pilot.DIALECT, pilot.__file__, __file__,
                    catalog_path, COVERAGE / "data/verification_cases.json",
                    COVERAGE / "results/verification.json", COVERAGE / "data/hlo_support.json",
                    COVERAGE / "scripts/hlo_support.py", COVERAGE / "scripts/paths.py",
                    COVERAGE.parent / "tepl/dialects/hlo_structural.tepl"]
    dependencies = {str(Path(p).resolve().relative_to(ROOT)): pilot.digest(Path(p)) for p in dependencies}
    dependencies.update({r["source"]: r["source_sha256"] for r in rows})
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Static classification only. Verification statuses come from the saved explicit sweep, not a new check.",
            "baseline_revision": baseline["tepl_revision"], "dependencies": dependencies, "cases": rows}


def select_batch(data):
    previous = {c["name"] for c in json.loads(pilot.MANIFEST.read_text())["cases"]}
    selected = []
    for category in FAMILIES:
        candidates = [c for c in data["cases"] if c["category"] == category
                      and not c["control"] and c["name"] not in previous
                      and c["attribute_scope"] == "axis_extent" and not c["missing_semantics"]]
        if category in CHALLENGE_FAMILIES:
            candidates = [c for c in candidates if c["recorded_explicit_status"] == "not_proved"]
        pilot.require(candidates, "no candidate for " + category)
        row = candidates[0]
        assisted = []
        if row["name"] == "reverse_add_scalar":
            assisted = [{"mode": "assisted_operation", "infer": ["b"], "keep_derived": []}]
        if row["name"] == "expr_convolution_scale_filter_bias":
            assisted = [{"mode": "assisted_operation", "infer": ["kernel"], "keep_derived": []}]
        selected.append({"name": row["name"], "source": row["source"], "catalog_id": row["catalog_id"],
                         "category": category, "scope": row["scope"], "policy": row["policy"],
                         "source_sha256": row["source_sha256"],
                         "recorded_explicit_status": row["recorded_explicit_status"],
                         "rhs_descriptors": {d: list(fields) for d, fields in row["rhs_descriptors"].items()},
                         "assisted": assisted})
    return {"schema_version": 1, "batch": "batch_01",
            "selection": "Five predeclared family strata: first remaining axis/extent case with all operation semantics in broadcast, reduction, dot, indexing, convolution, in original manifest order. Reduction and indexing strata require a recorded unproved explicit baseline to probe synthesis beyond baseline-proved rules. Exclude pilot cases. Not representative sampling.",
            "assistance": "For the two selected mixed-operation cases, infer only broadcast descriptors and retain the other operations' captured descriptors. No guard changes. No post-result tuning.",
            "cases": selected}


def report(data):
    rows = [r for r in data["cases"] if not r["control"]]
    attrs = [r for r in rows if r["rhs_descriptors"]]
    supported = [r for r in attrs if not r["missing_semantics"]]
    eligible = [r for r in supported if r["attribute_scope"] == "axis_extent"]
    lines = ["# XLA corpus classification for attribute inference", "", data["note"], "",
             "Cases, not catalog schemas, are the counting unit. The negative control is listed in JSON and excluded below.", "",
             "| RHS inference work | Recorded proved | Recorded unproved | Missing semantics | Total |",
             "|---|---:|---:|---:|---:|"]
    for has_attrs, label in [(False, "No RHS attributes"), (True, "Has RHS attributes")]:
        subset = [r for r in rows if bool(r["rhs_descriptors"]) == has_attrs]
        counts = Counter(r["recorded_explicit_status"] for r in subset)
        lines.append(f"| {label} | {counts['proved']} | {counts['not_proved']} | {counts['missing_semantics']} | {len(subset)} |")
    lines += ["", f"Of **{len(attrs)}** attribute-bearing cases, **{len(supported)}** have semantics for all required operations.",
              f"Of these, **{len(eligible)}** have only axis/extent (`index[]`) RHS fields; **{len(supported)-len(eligible)}** request other or mixed fields.",
              "This type-based eligibility criterion identifies experiments to attempt; it does not promise inference support.",
              "Comparison direction is a scalar setting, reported separately from axis/extent inference.", "",
              "| Attribute scope with semantics | Recorded proved | Recorded unproved | Total |",
              "|---|---:|---:|---:|"]
    for scope in ("axis_extent", "other_attributes", "mixed_attributes"):
        subset = [r for r in supported if r["attribute_scope"] == scope]
        count = Counter(r["recorded_explicit_status"] for r in subset)
        lines.append(f"| {scope} | {count['proved']} | {count['not_proved']} | {len(subset)} |")
    lines += ["", "## Next batch", "", select_batch(data)["selection"], "",
              "| Case | Family | Recorded explicit result |",
              "|---|---|---|"]
    for c in select_batch(data)["cases"]:
        lines.append(f"| {c['name']} | {c['category']} | {c['recorded_explicit_status']} |")
    mismatches = [r["name"] for r in rows if not r["recorded_source_matches"]]
    lines += ["", "Source files differing from their recorded explicit sweep hashes: " + (", ".join(mismatches) or "none") + ".", "",
              "The next batch is a five-case mechanism sample, not an estimate of full-corpus coverage.",
              "Only manifest-listed cases will execute. The rest remain unmeasured for automatic inference.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate saved inventory without running TEPL")
    args = parser.parse_args()
    if args.check:
        saved = json.loads(RESULT.read_text())
        for name, expected in saved["dependencies"].items():
            pilot.require(pilot.digest(ROOT / name) == expected, "stale classification: " + name)
        pilot.require(REPORT.read_text() == report(saved), "classification report is stale")
        pilot.require(json.loads(BATCH.read_text()) == select_batch(saved), "batch selection differs from classification")
        print("Classification and batch selection are current; no TEPL checks executed.")
        return
    data = classify()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(data, indent=2) + "\n")
    REPORT.write_text(report(data))
    BATCH.write_text(json.dumps(select_batch(data), indent=2) + "\n")
    print("Classified the XLA corpus and froze five cases in data/batch_01.json; no TEPL checks executed.")


if __name__ == "__main__":
    main()
