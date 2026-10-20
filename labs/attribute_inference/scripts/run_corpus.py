#!/usr/bin/env python3
"""Complete the classified attribute-bearing corpus without changing the compiler."""

import argparse
import itertools
import json
import os
import platform
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pilot

HERE, ROOT = pilot.HERE, pilot.ROOT
CLASSIFICATION = HERE / "results/corpus_classification.json"
MANIFEST = HERE / "data/corpus_cases.json"
RESULT = HERE / "results/corpus.json"
REPORT = HERE / "reports/corpus.md"
LITERAL = re.compile(r"\[\s*(?:\d+\s*(?:,\s*\d+\s*)*,?\s*)?\]")
POLICY = (
    "All non-control attribute-bearing cases with definitions for every required operation. "
    "Primary cohort: only index[] RHS fields (axis/extent); other fields are supplementary. "
    "Run explicit and full-auto variants for every case. On full-auto failure, test every nonempty "
    "proper subset of RHS descriptors for operation-level assistance. For single-descriptor cases, "
    "also test every nonempty proper subset of available literal-list field values from the original "
    "constructor, its spreads, or direct conjunctive guard equalities. At least one field must remain "
    "inferred. Do not invent literals, change guards, or combine the two assistance modes. "
    "Assisted variants of full-auto successes are prepared but not executed."
)


def split_top(text, separator):
    """Split statements/record fields without splitting lists or comprehensions."""
    spans, start, depth = [], 0, 0
    clean = pilot.uncomment(text)
    for i, char in enumerate(clean):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        pilot.require(depth >= 0, "unbalanced derivation")
        if char == separator and depth == 0:
            spans.append(text[start:i].strip())
            start = i + 1
    pilot.require(depth == 0, "unbalanced derivation")
    if text[start:].strip():
        spans.append(text[start:].strip())
    return [part for part in spans if part]


def definitions(source):
    derive = pilot.block(source, "derive")
    if not derive:
        return []
    found = []
    for statement in split_top(derive[2], ";"):
        match = re.fullmatch(r"\s*(@\w+|let\s+\w+)\s*=\s*(.*)", pilot.uncomment(statement), re.DOTALL)
        pilot.require(match is not None, "unsupported derive statement")
        key, value = match.groups()
        key = key if key.startswith("@") else key.split()[1]
        found.append((key, value.strip(), statement + ";"))
    return found


def retained_derive(source, roots):
    items = definitions(source)
    by_name = {name: value for name, value, _ in items}
    needed = set(roots) & set(by_name)
    pending = list(needed)
    while pending:
        name = pending.pop()
        for token in pilot.tokens(by_name[name]):
            if token in by_name and token not in needed:
                needed.add(token)
                pending.append(token)
    return [statement for name, _, statement in items if name in needed]


def literal_fields(source, descriptor):
    """Read only pre-existing literal lists; symbolic expressions are never pins."""
    guard = pilot.block(source, "where")
    known = {}
    if guard:
        # Restrict extraction to whole direct equality conjuncts, not nested
        # disjunctions, negations, comprehensions, or local computations.
        for clause in pilot.uncomment(guard[2]).split("&&"):
            match = re.fullmatch(r"\s*(@\w+)\.(\w+)\s*==\s*(\[[\d,\s]*\])\s*", clause)
            if match and LITERAL.fullmatch(match[3]):
                known.setdefault(match[1], {})[match[2]] = match[3]
    items = definitions(source)
    for name, value, _ in items:
        if not name.startswith("@"):
            continue
        if re.fullmatch(r"@\w+", value):
            known[name] = dict(known.get(value, {}))
        elif value.startswith("{") and value.endswith("}"):
            fields = {}
            for part in split_top(value[1:-1], ","):
                spread = re.fullmatch(r"\.\.\s*(@\w+)", part)
                if spread:
                    fields.update(known.get(spread[1], {}))
                    continue
                match = re.fullmatch(r"(\w+)\s*:\s*(.*)", part, re.DOTALL)
                pilot.require(match is not None, "unsupported descriptor field")
                field, expression = match.groups()
                fields.pop(field, None)
                if LITERAL.fullmatch(expression.strip()):
                    fields[field] = expression.strip()
            known[name] = fields
    return known.get("@" + descriptor, {})


def proper_subsets(items):
    for count in range(1, len(items)):
        yield from itertools.combinations(items, count)


def assistance(source, case):
    descriptors = list(case["rhs_descriptors"])
    variants = []
    for infer in proper_subsets(descriptors):
        variants.append({"mode": "assisted_operation__" + "__".join(infer),
                         "infer": list(infer), "pins": {}})
    if len(descriptors) == 1:
        name = descriptors[0]
        fields = case["rhs_descriptors"][name]
        candidates = literal_fields(source, name)
        candidates = {f: v for f, v in candidates.items() if f in fields}
        for count in range(1, min(len(candidates), len(fields) - 1) + 1):
            for pinned in itertools.combinations(candidates, count):
                variants.append({"mode": "assisted_field__" + "__".join(pinned),
                                 "infer": [name], "pins": {name: {f: candidates[f] for f in pinned}}})
    return variants


def build_manifest():
    data = json.loads(CLASSIFICATION.read_text())
    for name, expected in data["dependencies"].items():
        pilot.require(pilot.digest(ROOT / name) == expected, "stale classification: " + name)
    cases = []
    for row in data["cases"]:
        if row["control"] or not row["rhs_descriptors"] or row["missing_semantics"]:
            continue
        case = {key: row[key] for key in ("name", "catalog_id", "category", "source", "scope", "policy",
                                          "source_sha256", "attribute_scope", "recorded_explicit_status")}
        case["rhs_descriptors"] = {n: list(fields) for n, fields in row["rhs_descriptors"].items()}
        case["assisted"] = assistance((ROOT / row["source"]).read_text(), case)
        cases.append(case)
    pilot.require(len(cases) == 42 and sum(c["attribute_scope"] == "axis_extent" for c in cases) == 38,
                  "classified corpus changed; review inclusion criteria")
    return {"schema_version": 1, "selection": POLICY, "cases": cases,
            "control": json.loads(pilot.MANIFEST.read_text())["control"]}


def make_variant(source, case, spec):
    names, infer = set(case["rhs_descriptors"]), set(spec["infer"])
    pins = spec.get("pins", {})
    pilot.require(infer <= names and set(pins) <= infer, "unknown inference target")
    start, end = pilot.rhs_span(source)
    rhs = source[start:end]
    found = re.findall(r"\[@(\w+)\]", pilot.uncomment(rhs))
    pilot.require(set(found) == names and len(found) == len(names), "unsupported RHS descriptor occurrences")
    for name in infer:
        pilot.require(set(pins.get(name, {})) <= set(case["rhs_descriptors"][name]), "unknown field pin")
        annotation = "[auto" + "".join(f", {f} = {v}" for f, v in pins.get(name, {}).items()) + "]"
        rhs = rhs.replace("[@" + name + "]", annotation)
    result = source[:start] + rhs + source[end:]
    derive = pilot.block(result, "derive")
    if derive:
        retained = retained_derive(source, {"@" + n for n in names - infer})
        replacement = "derive {\n        " + "\n        ".join(retained) + "\n    }" if retained else ""
        result = result[:derive[0]] + replacement + result[derive[1]:]
    pilot.require(pilot.invariant(result) == pilot.invariant(source), "variant changes structure or guards")
    requested = sum(len(case["rhs_descriptors"][n]) - len(pins.get(n, {})) for n in infer)
    pilot.require(requested > 0, "no fields left to infer")
    return result, requested


def prepare(manifest):
    jobs = []
    for case in manifest["cases"]:
        source = ROOT / case["source"]
        pilot.require(pilot.digest(source) == case["source_sha256"], "case changed since selection")
        original = source.read_text()
        specs = [{"mode": "explicit"}, {"mode": "full_auto", "infer": list(case["rhs_descriptors"]), "pins": {}}] + case["assisted"]
        for spec in specs:
            text, requested = (original, 0) if spec["mode"] == "explicit" else make_variant(original, case, spec)
            path = HERE / "tepl/corpus" / case["name"] / (spec["mode"] + ".tepl")
            path.parent.mkdir(parents=True, exist_ok=True)
            relative = Path(os.path.relpath(pilot.DIALECT, path.parent)).as_posix()
            pilot.require(text.count('"dialects/hlo.tepl"') == 1, "unexpected dialect import")
            text = text.replace('"dialects/hlo.tepl"', '"' + relative + '"')
            path.write_text("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
            jobs.append({"case": case["name"], "mode": spec["mode"], "file": str(path.relative_to(ROOT)),
                         "policy": case["policy"], "control": False, "attribute_scope": case["attribute_scope"],
                         "category": case["category"], "catalog_id": case["catalog_id"],
                         "infer": spec.get("infer", []), "pins": spec.get("pins", {}),
                         "requested_fields": requested, "metrics": pilot.code_metrics(path.read_text()),
                         "expected_constructor_fields": [f for n in spec.get("infer", []) for f in case["rhs_descriptors"][n]]})
    control = manifest["control"]
    original = next(j for j in jobs if j["case"] == control["source_case"] and j["mode"] == "explicit")
    text = (ROOT / original["file"]).read_text()
    pilot.require(text.count(control["replace"]) == 1, "ambiguous control")
    path = (ROOT / original["file"]).with_name("incorrect_control.tepl")
    path.write_text(text.replace(control["replace"], control["with"]))
    jobs.append({**original, "mode": "incorrect_control", "file": str(path.relative_to(ROOT)), "control": True})
    return jobs


def best_assisted(rows, name):
    candidates = [r for r in rows if r["case"] == name and r["mode"].startswith("assisted") and r["status"] == "proved"]
    return min(candidates, key=lambda r: (r["metrics"]["attribute_tokens"], -r["requested_fields"], r["mode"])) if candidates else None


def constructor_kind(row):
    expressions = [c["expression"] for c in row["constructors"] if c["field"] not in
                   {f for values in row.get("pins", {}).values() for f in values}]
    if all(LITERAL.fullmatch(e) for e in expressions):
        return "constant"
    if all(LITERAL.fullmatch(e) or re.fullmatch(r"@\w+\.\w+", e) for e in expressions):
        return "reuse"
    return "computed"


def identity_witness(case, row):
    """Conservatively recognize unchanged unary operations with copied fields."""
    if row["status"] != "proved" or row["mode"] != "full_auto":
        return False
    source = pilot.uncomment((ROOT / case["source"]).read_text())
    start, end = pilot.rhs_span(source)
    lhs = re.search(r"\(([\w.]+)\[@(\w+)\]\s+(\w+)\)\s*$", source[:source.index("=>")])
    if not lhs:
        return False
    rhs = source[start:end].strip()
    for left, right in reversed(pilot.annotations(rhs)):
        rhs = rhs[:left] + rhs[right:]
    plain_lhs = f"({lhs[1]} {lhs[3]})"
    return (pilot.tokens(rhs) == pilot.tokens(plain_lhs)
            and bool(row["constructors"])
            and all(c["expression"] == f"@{lhs[2]}.{c['field']}" for c in row["constructors"]))


def report(data):
    rows = data["results"]
    lookup = {(r["case"], r["mode"]): r for r in rows}
    cases = data["cases"]
    primary = [c for c in cases if c["attribute_scope"] == "axis_extent"]
    supplementary = [c for c in cases if c["attribute_scope"] != "axis_extent"]
    baseline = {c["name"] for c in primary if lookup[c["name"], "explicit"]["status"] == "proved"}
    full = {c["name"] for c in primary if lookup[c["name"], "full_auto"]["status"] == "proved"}
    assisted = {c["name"] for c in primary if best_assisted(rows, c["name"])}
    identities = {c["name"] for c in primary if identity_witness(c, lookup[c["name"], "full_auto"])}
    lines = ["# Automatic attribute inference: complete XLA cohort", "", data["selection"], "",
             "## Corpus and primary results", "",
             "The source evaluation has 209 non-control cases: 98 have no RHS attributes and 111 have attribute work. Of the latter, 69 lack required operation semantics. All remaining 42 are checked here: 38 axis/extent cases in the primary cohort and four scalar-setting cases in a supplementary cohort. All numbers count concrete cases, not entire catalog schemas.", "",
             "| Measurement | Result |", "|---|---:|",
             f"| Primary cases attempted | {len(primary)} |",
             f"| Explicit baselines proved | {len(baseline)}/{len(primary)} |",
             f"| Full auto, all primary cases | {len(full)}/{len(primary)} |",
             f"| Full auto, conditioned on explicit proof | {len(full & baseline)}/{len(baseline)} |",
             f"| Additional cases recovered by assistance | {len(assisted - full)} |",
             f"| Full or assisted, all primary cases | {len(full | assisted)}/{len(primary)} |",
             f"| Full or assisted, conditioned on explicit proof | {len((full | assisted) & baseline)}/{len(baseline)} |",
             f"| Inference successes with unproved explicit baseline | {len((full | assisted) - baseline)} |", "",
             f"The primary cohort spans {len({c['catalog_id'] for c in primary})} catalog schemas; full-auto successes cover {len({c['catalog_id'] for c in primary if c['name'] in full})} schemas. A concrete specialization does not prove its whole catalog schema.", "",
             "Assistance is tested only after full-auto failure. Its fraction is additional coverage under the stated bounded policy, not completeness over every possible user hint.", "",
             "## Primary results by family", "",
             "| Family | Cases | Explicit proved | Full auto | Additional assisted | Full or assisted |",
             "|---|---:|---:|---:|---:|---:|"]
    for category in sorted({c["category"] for c in primary}):
        names = {c["name"] for c in primary if c["category"] == category}
        lines.append(f"| {category} | {len(names)} | {len(names & baseline)} | {len(names & full)} | {len(names & (assisted-full))} | {len(names & (full|assisted))} |")
    kinds = Counter(constructor_kind(lookup[name, "full_auto"]) for name in full)
    lines += ["", "## What was constructed", "",
              "This syntactic classification describes the printed constructor programs. Constants do not demonstrate arbitrary-rank reconstruction; reuse may contain constant fields; computed programs contain operations beyond field copies and literal lists.", "",
              "| Full-auto constructor class | Cases |", "|---|---:|"]
    for kind in ("reuse", "constant", "computed"):
        lines.append(f"| {kind} | {kinds[kind]} |")
    successful = [lookup[n, "full_auto"] for n in full]
    paired = [lookup[n, "full_auto"] for n in full & baseline]
    paired_old = sum(lookup[r["case"], "explicit"]["metrics"]["attribute_tokens"] for r in paired)
    paired_new = sum(r["metrics"]["attribute_tokens"] for r in paired)
    old_tokens = sum(lookup[r["case"], "explicit"]["metrics"]["attribute_tokens"] for r in successful)
    new_tokens = sum(r["metrics"]["attribute_tokens"] for r in successful)
    lines += ["", f"For the **{len(paired)} pairs where both explicit and full-auto versions verify**, RHS attribute/derive tokens decrease from **{paired_old} to {paired_new}** ({100*(paired_old-paired_new)/paired_old:.1f}% reduction).",
              f"Across all {len(successful)} full-auto successes, the written explicit-reference token count decreases from {old_tokens} to {new_tokens} ({100*(old_tokens-new_tokens)/old_tokens:.1f}%). That broader comparison includes {len(successful)-len(paired)} unproved explicit references.",
              "These counts exclude unchanged guards, shared operation semantics and failed inferred cases; they are code-size proxies, not usability results.", "",
              "## Witness quality", "",
              "Inference constructs a semantically valid witness, which need not equal the handwritten descriptor or implement its optimization policy.",
              f"A conservative unchanged-unary-operation audit identifies **{len(identities)} identity witness** among the full-auto successes: " + (", ".join(sorted(identities)) or "none") + ".",
              f"Thus {len(full)-len(identities)} primary full-auto successes remain after excluding those detected identity witnesses. This audit is not a general profitability analysis.",
              "In `catalog_indexing_slice_singleton_stride`, inference copies `@s.starts`, `@s.sizes` and `@s.strides`; the intended explicit rule replaces singleton-axis strides with one. The inferred version proves equivalence but leaves the original slice unchanged. It is included in semantic-witness coverage and identified separately from optimization realization.",
              "The other new full-auto success, `catalog_broadcast_transpose`, constructs a replacement transpose. Both written explicit versions remain unproved because their comprehensions exceed the checker's supported metadata fragment.", "",
              "## Per-case primary results", "",
              "| Case | Explicit | Full auto | Best successful assistance | Attribute tokens explicit → selected proved variant |",
              "|---|---|---|---|---|"]
    for c in primary:
        name = c["name"]
        explicit, inferred = lookup[name, "explicit"], lookup[name, "full_auto"]
        extra = best_assisted(rows, name)
        selected = inferred if inferred["status"] == "proved" else extra
        metric = f"{explicit['metrics']['attribute_tokens']} → {selected['metrics']['attribute_tokens']}" if selected else "unproved"
        assistance_text = extra["mode"] if extra else "not needed" if inferred["status"] == "proved" else "none proved"
        lines.append(f"| {name} | {explicit['status']} | {inferred['status']} | {assistance_text} | {metric} |")
    lines += ["", "## Supplementary scalar-setting cohort", "",
              "These four comparison-setting cases are measured and disclosed separately; they do not change the primary axis/extent denominator.", "",
              "| Case | Explicit | Full auto |", "|---|---|---|"]
    for c in supplementary:
        name = c["name"]
        lines.append(f"| {name} | {lookup[name, 'explicit']['status']} | {lookup[name, 'full_auto']['status']} |")
    attempts = [r for r in rows if r["mode"].startswith("assisted") and r["status"] != "not_needed"]
    lines += ["", "## Assistance evidence", "",
              f"Executed {len(attempts)} predeclared assisted variants; {sum(r['status'] == 'proved' for r in attempts)} proved. Multiple variants of one case count only once toward additional case coverage.",
              "Best means fewest remaining attribute tokens, then most requested fields, then stable variant name. All alternatives, failures and skipped variants are in results/corpus.json.", ""]
    for c in primary:
        chosen = best_assisted(rows, c["name"])
        if chosen:
            lines += [f"### {c['name']}: {chosen['mode']}", "", "```text", "User pins: " + json.dumps(chosen["pins"], sort_keys=True)]
            lines += [f"{v['field']} = {v['expression']}" for v in chosen["constructors"]]
            lines += ["```", ""]
    lines += ["## Unproved full-auto cases", "", "An unproved result is not an impossibility claim. Exact diagnostics are retained below and in raw evidence.", ""]
    for c in cases:
        r = lookup[c["name"], "full_auto"]
        if r["status"] != "proved":
            lines += [f"### {c['name']}", "", "```text", r["stderr"].strip() or r["status"], "```", ""]
    control = next(r for r in rows if r["control"])
    lines += ["## Validation and limits", "",
              f"The incorrect permutation-order control is **{control['status']}**, as expected. Acceptance requires an explicit proved status, true additional guard and completed requested constructors.",
              "All variants preserve original guards, tensor restrictions, operation structure and local dialect semantics. Some source cases specialize ranks, layouts or attribute domains; case scopes are retained in the manifest.",
              "No compiler changes, stronger applicability guards or operation-semantic substitutions were introduced to improve coverage.",
              "This run completes the classified XLA attribute cohort. It does not complete the planned general-symbolic, independent numerical, scalability or ablation experiments.",
              "Single CLI wall times are reproducibility evidence, not performance results. Algebraic verification does not establish complete IEEE or XLA backend equivalence.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="freeze all cases and assistance before execution")
    parser.add_argument("--check", action="store_true", help="verify saved hashes and report without running TEPL")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    pilot.require(not (args.prepare and args.check) and args.timeout > 0, "invalid options")
    if args.prepare:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        jobs = prepare(manifest)
        print(f"Prepared {len(manifest['cases'])} cases and {len(jobs)} planned variants/control; no TEPL checks executed.")
        return
    compiler = args.compiler.resolve(strict=True)
    if args.check:
        data = json.loads(RESULT.read_text())
        pilot.require(pilot.digest(compiler) == data["compiler_sha256"], "compiler changed")
        for name, expected in data["dependencies"].items():
            pilot.require(pilot.digest(ROOT / name) == expected, "stale corpus dependency: " + name)
        pilot.require(REPORT.read_text() == report(data), "corpus report is stale")
        print("Complete corpus hashes and report are current; no TEPL checks executed.")
        return
    manifest = json.loads(MANIFEST.read_text())
    pilot.require(manifest == build_manifest(), "manifest differs from deterministic selection; prepare again")
    jobs = prepare(manifest)
    paths = pilot.dependency_paths(jobs, manifest) + [MANIFEST, CLASSIFICATION, Path(__file__).resolve()]
    dependencies = {str(p.relative_to(ROOT)): pilot.digest(p) for p in paths}
    compiler_hash = pilot.digest(compiler)
    rows, full_status = [], {}
    for job in jobs:
        if job["mode"].startswith("assisted") and full_status[job["case"]] == "proved":
            rows.append({**job, "status": "not_needed", "stdout": "", "stderr": "", "constructors": []})
            continue
        row = pilot.run_job(compiler, job, args.timeout)
        rows.append(row)
        if job["mode"] == "full_auto":
            full_status[job["case"]] = row["status"]
        print(f"{row['case']} / {row['mode']}: {row['status']}", flush=True)
    pilot.require(pilot.digest(compiler) == compiler_hash and all(pilot.digest(ROOT/p) == h for p, h in dependencies.items()),
                  "compiler or inputs changed during sweep")
    data = {"schema_version": 1, "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "tepl_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines(),
            "compiler_sha256": compiler_hash, "environment": platform.platform(), "python": platform.python_version(),
            "timeout_seconds_per_variant": args.timeout, "selection": manifest["selection"],
            "dependencies": dependencies, "cases": manifest["cases"], "results": rows}
    RESULT.write_text(json.dumps(data, indent=2) + "\n")
    REPORT.write_text(report(data))
    pilot.require(all(r["status"] not in ("error", "timeout") for r in rows), "infrastructure error or timeout; inspect evidence")
    pilot.require(next(r for r in rows if r["control"])["status"] == "not_proved", "incorrect control was not rejected")
    print("Complete attribute cohort saved to reports/corpus.md.")


if __name__ == "__main__":
    main()
