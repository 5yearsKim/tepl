#!/usr/bin/env python3
"""Classify original rewrite graphs by overlapping operation-family membership."""

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from . import hlo_support
    from .paths import HERE, ROOT
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import hlo_support
    from scripts.paths import HERE, ROOT


MAPPING = HERE / "data/operation_family_mapping.json"
CLASSIFICATION = HERE / "data/synthesis_classification.json"
OUTPUT = HERE / "data/case_families.json"
REPORT = HERE / "reports/operation_families.md"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operation_mapping(mapping, declared):
    result = {}
    for family in mapping["families"]:
        for operation in family["operations"]:
            if operation in result:
                raise ValueError("operation belongs to multiple families: " + operation)
            result[operation] = family["id"]
    if set(result) != set(declared):
        raise ValueError("family mapping must cover every declared HLO operation exactly once")
    if len({f["id"] for f in mapping["families"]}) != len(mapping["families"]):
        raise ValueError("duplicate family ID")
    return result


def classify(cases, mapping):
    by_operation = operation_mapping(mapping, hlo_support.inventory())
    order = [f["id"] for f in mapping["families"]]
    rows = []
    seen = set()
    for case in cases:
        name = case["case"]
        if name in seen:
            raise ValueError("duplicate case: " + name)
        seen.add(name)
        source = (HERE / case["source_file"]).resolve()
        if sha(source) != case["source_sha256"]:
            raise ValueError("source differs from frozen classification: " + name)
        operations = hlo_support.required_operations(source)
        if operations != case["required_operations"]:
            raise ValueError("operation inventory differs from frozen classification: " + name)
        families = {by_operation[op] for op in operations}
        if not families:
            raise ValueError("case has no operation family: " + name)
        rows.append(dict(
            case=name,
            catalog_id=case["catalog_id"],
            control=case["control"],
            in_scope=case["in_scope"],
            source_file=str(source.relative_to(ROOT)),
            source_sha256=sha(source),
            operations=operations,
            families=[f for f in order if f in families],
        ))
    return rows


def summarize(rows, mapping):
    corpus = [r for r in rows if not r["control"]]
    total = len(corpus)
    summaries = []
    for family in mapping["families"]:
        members = [r for r in corpus if family["id"] in r["families"]]
        count = len(members)
        in_scope_count = sum(r["in_scope"] for r in members)
        summaries.append(dict(
            family=family["id"], count=count,
            percentage=round(100 * count / total, 1) if total else 0.0,
            in_scope_count=in_scope_count,
            in_scope_percentage=round(100 * in_scope_count / count, 1) if count else None,
        ))
    return summaries


def render(data, mapping):
    rows = data["cases"]
    corpus = [r for r in rows if not r["control"]]
    schemas = len({r["catalog_id"] for r in corpus})
    controls = [r["case"] for r in rows if r["control"]]
    lines = [
        "# Operation-family coverage", "",
        f"The corpus contains **{len(corpus)} rewrite cases** representing **{schemas} catalog schemas**. "
        f"The {len(controls)} negative control is excluded from family counts and percentages.", "",
        "| Operation family | Representative operations | Cases involving family, n (%) | Within evaluation scope, n (%) |",
        "|---|---|---:|---:|",
    ]
    for family, summary in zip(mapping["families"], data["summary"]):
        scoped = (f'{summary["in_scope_count"]} ({summary["in_scope_percentage"]:.1f}%)'
                  if summary["count"] else "0 (—)")
        lines.append(f'| {family["label"]} | {family["examples"]} | '
                     f'{summary["count"]} ({summary["percentage"]:.1f}%) | {scoped} |')
    lines += [
        "",
        f"Each case contributes once to every family appearing on either side of its original rewrite. "
        f"The cases-involving-family percentages use {len(corpus)} distinct rewrite cases as the denominator. Memberships overlap; "
        "counts may sum to more than the corpus size and percentages may sum to more than 100%.",
        "",
        "Within-evaluation-scope percentages use each row's family case count as the denominator. "
        "Scope membership is the frozen axis-only classification's `in_scope` flag, including axis-inference "
        "and verification-only tasks. It describes eligibility in that evaluation, not successful verification "
        "or a general limit on TEPL support. The negative control is excluded from both columns.",
        "", "## Classification method", "",
        "Classification parses the original semantic rule graphs, before axis-task migration or adapter "
        "normalization, and resolves operation aliases to their canonical HLO names. It includes all cases, "
        "regardless of evaluation scope or verification outcome. Repeated operations and operations present "
        "on both sides contribute only one membership per family. Captured subgraphs are opaque; operations "
        "inside captures, imported declarations, and metadata functions in attributes or conditions are not counted.", "",
        "The complete mapping below assigns each local HLO operation to exactly one family. "
        "Elementwise includes scalar/complex applications, conversion, map, and the constant constructors "
        "zero_like, fill, and scalar. Indexing includes coordinate generation (iota), rotation, padding, "
        "concatenation, and indexed reads/updates. Generalized dot remains in Dot even though its semantics "
        "use summation. Reduce-window operations belong to Reduction; restricted and general convolution "
        "belong to Convolution. Transpose and Reshape are separate families.", "",
        "These are case memberships based on explicit operations, rather than the catalog's single schema "
        "category. They characterize the written TEPL rules, not every possible instantiation of each XLA schema.",
        "", "## Complete operation-to-family mapping", "",
        "| Family | Canonical local HLO operations |", "|---|---|",
    ]
    for family in mapping["families"]:
        lines.append(f'| {family["label"]} | ' + ", ".join(f'`{op}`' for op in family["operations"]) + " |")
    lines += [
        "", "## Evidence and reproduction", "",
        "[Mapping](../data/operation_family_mapping.json) and "
        "[per-case memberships](../data/case_families.json) provide the full classification. "
        "Generated memberships record input and source hashes. The generator checks source hashes and "
        "operation inventories against the frozen synthesis classification; no compiler run is needed.", "",
        "Excluded negative control: " + ", ".join(f"`{c}`" for c in controls) + ".", "",
        "```sh",
        "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/operation_families.py",
        "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/operation_families.py --check",
        "```", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check generated files without writing")
    args = parser.parse_args()
    mapping = read(MAPPING)
    rows = classify(read(CLASSIFICATION)["cases"], mapping)
    data = dict(
        schema_version=1,
        policy="Union of explicit original LHS/RHS operation families; one membership per family per case; controls excluded from summary.",
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in (
            MAPPING, CLASSIFICATION, Path(__file__).resolve(),
            HERE / "scripts/hlo_support.py", HERE / "scripts/synthesis_cases.py",
            HERE.parent / "tepl/dialects/hlo.tepl",
        )},
        rewrite_case_count=sum(not r["control"] for r in rows),
        control_count=sum(r["control"] for r in rows),
        summary=summarize(rows, mapping), cases=rows,
    )
    outputs = {OUTPUT: json.dumps(data, indent=2) + "\n", REPORT: render(data, mapping)}
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise ValueError("stale generated file: " + str(path))
        else:
            path.write_text(content)
    print(f'{data["rewrite_case_count"]} rewrite cases; {data["control_count"]} control; family report ' +
          ("checked" if args.check else "written"))


if __name__ == "__main__":
    main()
