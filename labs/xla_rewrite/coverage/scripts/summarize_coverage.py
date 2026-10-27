#!/usr/bin/env python3
"""Validate the catalog assessment and generate its coverage report."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from . import hlo_support
    from .paths import HERE, ROOT, TEPL, validate_frozen_corpus
else:
    import hlo_support
    from paths import HERE, ROOT, TEPL, validate_frozen_corpus
STATUSES = ("candidate", "restricted", "proof_gap", "unsupported")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(catalog, assessment, manifest, measured):
    hlo_support.validate_support()
    ids = {r["id"] for r in catalog}
    assessed = [r["id"] for r in assessment["rules"]]
    require(len(assessed) == len(set(assessed)), "duplicate assessment IDs")
    require(set(assessed) == ids, "assessments must cover exactly the catalog")
    require(assessment["xla_revision"] == read(
        HERE.parent / "catalog/source_manifest.json")["revision"],
        "XLA revisions differ")
    capabilities = assessment["capabilities"]
    for capability in capabilities.values():
        require(capability["evidence"], "capability needs evidence")
        for evidence in capability["evidence"]:
            require((ROOT / evidence).is_file(), f"missing evidence: {evidence}")
    for row in assessment["rules"]:
        require(row["assessment"] in STATUSES, "unknown assessment status")
        require(row["reason"] and row["requires"], "missing assessment reason")
        require(set(row["requires"]) <= set(capabilities), "unknown capability")
        require(row["route"] in assessment["routes"], "unknown implementation route")
        if row["assessment"] == "restricted":
            require(row["restriction"], "restricted assessment needs a restriction")
    names = [c["name"] for c in manifest["cases"]]
    require(len(names) == len(set(names)), "duplicate case names")
    files = set()
    for case in manifest["cases"]:
        require(case["policy"] in {"strict", "algebraic"}, "unknown policy")
        require(case["control"] == (case["catalog_id"] is None),
                "only controls may omit a catalog ID")
        require(case["control"] or case["catalog_id"] in ids, "unknown case catalog ID")
        path = (HERE / case["file"]).resolve()
        files.add(path.resolve())
        require(path.is_file() and case["scope"], "case needs file and scope")
        require(not re.search(r"\bauto\b", path.read_text()),
                f"explicit case contains auto: {case['name']}")
        require('from "dialects/hlo.tepl" import HLO as hlo;' in path.read_text(),
                "semantic cases must import the local HLO dialect from the project root")
    require(files == {p.resolve() for p in (TEPL / "rules/semantic").glob("*.tepl")},
            "unlisted or missing explicit case files")
    expression_cases = read(HERE / "data/compilation_cases.json")["cases"]
    counterparts = {c.get("expression_case"): c for c in manifest["cases"]
                    if not c["control"]}
    require(set(counterparts) == {c["name"] for c in expression_cases},
            "each compilation case needs one semantic counterpart")
    for expression in expression_cases:
        counterpart = counterparts[expression["name"]]
        require(counterpart["catalog_id"] == expression["catalog_id"],
                "semantic counterpart has a different catalog schema")
        semantic_source = (HERE / counterpart["file"]).read_text()
        expression_source = (HERE / expression["file"]).read_text().replace(
            '"dialects/hlo_structural.tepl"', '"dialects/hlo.tepl"')
        require(semantic_source == expression_source,
                "semantic counterpart must preserve the written compilation rule")
    records = measured["results"]
    result_names = [r["case"] for r in records]
    require(len(result_names) == len(set(result_names))
            and set(result_names) == set(names), "results must cover each case once")
    require(re.fullmatch(r"[0-9a-f]{40}", measured["tepl_revision"]) is not None,
            "invalid measured TEPL revision")
    cases = {c["name"]: c for c in manifest["cases"]}
    for result in records:
        case = cases[result["case"]]
        require(result["catalog_id"] == case["catalog_id"], "case/result ID mismatch")
        require(result["file_sha256"] == sha(HERE / case["file"]),
                f"stale result for {case['name']}; rerun run_checks.py")
        require(result["status"] in {"proved", "missing_semantics", "not_proved", "timeout"},
                "unknown measured status")
        if case["control"]:
            require(result["status"] == "not_proved", "negative control must fail")
        if result["status"] == "proved":
            require(result["returncode"] == 0, "proved result must exit successfully")
            require(not result.get("missing_semantics") and result.get("proof_evidence")
                    and any("status=proved" in line for line in result["proof_evidence"])
                    and "additional-guard: true" in result["proof_evidence"],
                    "proved result needs semantic proof evidence, not structural acceptance")
        if result["status"] == "missing_semantics":
            require(result.get("missing_semantics"),
                    "missing semantic result needs the operation names")
    for name, expected in measured.get("compiler_source_hashes", {}).items():
        require(sha(ROOT / name) == expected, f"stale compiler source: {name}")
    for name, expected in measured["dependencies"].items():
        require(sha(ROOT / name) == expected, f"stale measured dependency: {name}")

    reasoning_comparison(measured)


def reasoning_comparison(measured):
    baseline = read(HERE / "baselines/reasoning_baseline.json")
    validate_frozen_corpus(baseline)
    previous = {r["case"] for r in baseline["results"] if r["status"] == "proved"}
    current = {r["case"] for r in measured["results"] if r["status"] == "proved"}
    lost = previous - current
    require(not lost, "previously verified cases regressed: " + ", ".join(sorted(lost)))
    return baseline, previous, current - previous


def clean(text):
    return text.replace("|", "\\|").replace("\n", " ")


def failure_summary(result):
    diagnostic = result["diagnostics"]
    if "comprehension is outside the pointwise metadata fragment" in diagnostic:
        return "The comprehension needs reasoning beyond the supported pointwise sequence fragment."
    if "checked metadata evaluation is not established" in diagnostic:
        return "Checked evaluation of this metadata operation is not established."
    if "metadata definedness:" in diagnostic:
        return "The original derived metadata expression needs an indexing, arithmetic or attribute-range proof."

    if "semantic graphs support tensor captures and operation applications" in diagnostic:
        return "Semantic graph lowering cannot interpret the literal node in this rule."
    if "unsupported semantic-rule condition" in diagnostic:
        return "The semantic checker cannot lower this guard (for example, a list comprehension)."
    if "unsupported semantic-rule local expression" in diagnostic:
        return "The semantic checker cannot lower this local metadata expression."
    if "unsupported semantic descriptor field expression" in diagnostic:
        return "The semantic checker cannot interpret this explicit computed descriptor."
    if "coordinate correspondence:" in diagnostic:
        return "Coordinate correspondence does not establish the required scalar-broadcast mapping."
    if "normalized coefficients, tensor accesses, or reduction domains differ" in diagnostic:
        return "Normalization does not establish scalar-value, tensor-access and reduction-domain equality."
    if "same_set(" in diagnostic:
        return "The checker does not establish the required equality of axis sets for the explicit attributes."
    if "is_valid_extents(" in diagnostic:
        return "The checker does not establish nonnegative extents for the explicitly constructed shape list."
    if "legality obligation: ==(" in diagnostic:
        return "The checker does not establish equality of the explicitly reconstructed shape/extents."
    return result["status"] + "; see the recorded diagnostics."


def render(catalog, assessment, manifest, measured):
    cases = {c["name"]: c for c in manifest["cases"]}
    records = [r for r in measured["results"] if not cases[r["case"]]["control"]]
    passed = [r for r in records if r["status"] == "proved"]
    passed_ids = {r["catalog_id"] for r in passed}
    attempted_ids = {r["catalog_id"] for r in records}
    counts = Counter(r["status"] for r in records)
    support = hlo_support.validate_support()
    supported = [name for name, row in support["operations"].items() if row["semantic"]]
    missing = [name for name, row in support["operations"].items() if not row["semantic"]]
    out = ["# Semantic Verification of HLO Rewrites", "",
           "Every explicit case in [compilation_cases.json](../data/compilation_cases.json) has "
           "a verification counterpart in [rules/](../../tepl/rules/semantic/). The cases use the local "
           "[HLO dialect](../../tepl/dialects/hlo.tepl), with project-root imports configured by "
           "[tepl.toml](../../tepl/tepl.toml). No case imports another project's dialect.", "",
           "## Result", "",
           f"The sweep tested **{len(records)} explicit rule cases**, spanning "
           f"**{len(attempted_ids)} of {len(catalog)} catalog schemas**:", "",
           "| Outcome | Rule cases | Meaning |",
           "|---|---:|---|",
           f"| Accepted by semantic verification | {len(passed)} | Explicit proof status is `proved`, with no additional guard. |",
           f"| Missing operation semantics | {counts['missing_semantics']} | At least one required operation has no faithful definition in the current semantic fragment. |",
           f"| Not proved with the declared semantics | {counts['not_proved']} | Required operation semantics exist, but construction, legality or equivalence was not established. |",
           f"| Timeout | {counts['timeout']} | The per-case time limit was exceeded. |", "",
           f"At least one case was verified for **{len(passed_ids)} catalog schemas**. "
           "A specialization does not prove all variants of its catalog schema. "
           "The deliberately incorrect subtraction-to-addition control was rejected "
           "and is excluded from the table.", "",
           f"The remaining **{len(catalog)-len(attempted_ids)} catalog schemas** have no "
           "concrete rule in this corpus; they are not counted as attempted or rejected.", "",
           "All tested rules supply RHS attributes explicitly and use no `auto`. "
           "Proofs use the ideal `algebraic` arithmetic policy. They do not establish "
           "IEEE rounding, NaN, signed-zero, fixed-width overflow or complete XLA "
           "backend equivalence. A rejected case means unproved, not mathematically false.", "",
           "## Results by category", "",
           "| Category | Catalog schemas | Cases tested | Verified cases | Schemas with a verified case | Missing semantics | Not proved | Timeout |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category in dict.fromkeys(r["category"] for r in catalog):
        ids = {r["id"] for r in catalog if r["category"] == category}
        tested = [r for r in records if r["catalog_id"] in ids]
        c = Counter(r["status"] for r in tested)
        out.append(f"| {category} | {len(ids)} | {len(tested)} | {c['proved']} | "
                   f"{len({r['catalog_id'] for r in tested if r['status']=='proved'})} | "
                   f"{c['missing_semantics']} | {c['not_proved']} | {c['timeout']} |")
    baseline, previous, added = reasoning_comparison(measured)
    comparison = ["", "## General reasoning comparison", "",
                  f"The frozen baseline verified **{len(previous)} cases**. The general "
                  f"reasoning extension verifies **{len(passed)}**, adding **{len(added)}** "
                  "with **no previous proof lost**. Rule bodies, conditions, rank "
                  "constraints and both HLO dialects are unchanged. Manifest edits "
                  "only relocate case files, as recorded in "
                  "[layout_migration.json](../baselines/layout_migration.json); "
                  "[reasoning_baseline.json](../baselines/reasoning_baseline.json) preserves their hashes "
                  "and the original per-case outcomes.", "",
                  "The compiler now lowers metadata indexing and pointwise comprehensions, "
                  "propagates integer intervals and source extent inequalities, composes "
                  "ordered sequence mappings, and combines reversals by selector parity. "
                  "Original derived metadata programs retain checked-evaluation and final "
                  "attribute-range obligations. No case names or HLO-specific identities "
                  "are used by the reasoning implementation.", "",
                  "Examples include "
                  "[reverse composition](../../tepl/rules/semantic/expr_reverse_compose_general.tepl), "
                  "[slice composition](../../tepl/rules/semantic/expr_slice_compose_general.tepl), and "
                  "[dot operand transposes](../../tepl/rules/semantic/catalog_dot_transpose_both_operands.tepl). "
                  "The baseline and current result JSON retain all per-case outcomes. "
                  "This comparison measures proof coverage, not compiler speed.", ""]
    out[out.index("## Results by category"):out.index("## Results by category")] = comparison
    out.extend(["", "## Local HLO semantics", "",
                f"The local dialect declares **{len(support['operations'])} operations** "
                f"for the evaluation and defines value/axis semantics for **{len(supported)}**: "
                + ", ".join(f"`{name}`" for name in supported) + ".", "",
                "These cover polynomial arithmetic, registered pointwise scalar "
                "functions, sum reductions, dot, transpose, "
                "axis-insertion broadcast, reverse, positive-size static slice, and "
                "valid fixed-layout ungrouped convolution. `zero_like` is a helper "
                "for shaped zero. HLO attributes are flattened for this experiment; "
                "this is not a complete HLO importer.", "",
                "Broadcast retains existing extents and inserts new axes. Slice sizes "
                "are positive. The simple convolution has positive strides/dilations "
                "and no padding. `reduce_sum` has an implicit zero initializer. "
                "Source assumptions and case restrictions remain in the manifests.", "",
                "Scalar functions share one representation and recursive coordinate "
                "tracking. Arguments normalize using existing polynomial rules; "
                "other functions stay intact until an implemented law applies. "
                "The current laws include min/max associativity, commutativity, "
                "idempotence and absorption; Boolean identities; select simplification; "
                "integer powers; and selected exp/log/sqrt/abs identities. Numeric "
                "domains are checked conservatively: generic quotient reassociation "
                "or log/sqrt rewrites without the required domain facts remain unproved. "
                "The registry does not make every identity involving its functions provable.", "",
                "Logical operations in this profile are Boolean, not integer bitwise "
                "operations. Select uses a predicate with the same shape as its arms. "
                "Compare uses EQ=0, NE=1, GE=2, GT=3, LE=4, LT=5. These are explicit "
                "local representations, not XLA's native attribute encoding.", "",
                "### Semantics the current language cannot express", "",
                f"The other **{len(missing)} operations** remain declarations. "
                "Their absence is intentional: substituting identity, sum, or the "
                "desired RHS for their actual meaning would produce invalid evidence. "
                "[hlo_support.json](../data/hlo_support.json) records each operation's reason.", "",
                "| Missing capability | Operations | Explanation |",
                "|---|---|---|"])
    groups = defaultdict(list)
    for name in missing:
        groups[support["operations"][name]["category"]].append(name)
    for category, names in groups.items():
        reason = support["operations"][names[0]]["reason"]
        out.append(f"| {category.replace('_',' ')} | "
                   + ", ".join(f"`{name}`" for name in names)
                   + " | " + clean(reason) + " |")
    out.extend(["", "The value vocabulary includes `Const`, operand/read, `Add`, binary "
                "`Product`, sum `Reduce`, and typed registered scalar applications. "
                "Graph numeric literals are also supported. Calculated coordinates "
                "support addition, scaling, offsets, selection and reversal. "
                "Required scalar numeric attributes can parameterize functions, such "
                "as compare's direction. `Const` still requires an integer literal; "
                "it cannot read a constant payload from an attribute. Scalar `select` "
                "does not supply coordinate predicates or a general reduction binder.", "",
                "## Unproved cases with operation semantics", "",
                "These cases reached the semantic path with definitions for all "
                "required operations. Their diagnostics distinguish unsupported "
                "attribute expressions from unresolved coordinate/legality/value proofs.", "",
                "| Case | Limitation |",
                "|---|---|"])
    for result in records:
        if result["status"] in {"not_proved", "timeout"}:
            out.append(f"| [{result['case']}](../{cases[result['case']]['file']}) | "
                       + clean(failure_summary(result)) + " |")
    out.extend(["", "## Reproduction", "",
                "[verification.json](../results/verification.json) records diagnostics, "
                "required/missing operations, proof evidence and file/dependency "
                "hashes, including the semantic compiler sources and untracked additions. "
                "The revision and semantic source hashes identify the measured compiler. [verification_cases.json](../data/verification_cases.json) records scope and arithmetic policy. "
                "The validators reject stale inputs and missing or duplicate cases. "
                "[Language coverage](compilation.md) separately measures ordinary "
                "compilation using [the structural HLO profile](../../tepl/dialects/hlo_structural.tepl).", "",
                f"- TEPL revision: `{measured['tepl_revision']}`.",
                f"- XLA revision: `{assessment['xla_revision']}`.",
                f"- Compiler SHA-256: `{measured['compiler_sha256']}`.",
                f"- Environment: {measured['environment']}; Python {measured['python']}.",
                f"- Timeout: {measured['timeout_seconds_per_case']} seconds per case.", "",
                "See the [case outcome table](synthesis_cases.md) for explicit and inferred "
                "results, and the [reproduction commands](../README.md#reproduce) to rerun "
                "the evaluation.", ""])
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        catalog = read(HERE.parent / "catalog/rules.json")["rules"]
        assessment = read(HERE / "data/assessments.json")
        manifest = read(HERE / "data/verification_cases.json")
        measured = read(HERE / "results/verification.json")
        validate(catalog, assessment, manifest, measured)
        report = render(catalog, assessment, manifest, measured)
        destination = HERE / "reports/verification.md"
        if args.check:
            require(destination.read_text() == report,
                    "report is stale; run summarize_coverage.py")
        else:
            destination.write_text(report)
        print(f"Validated {len(catalog)} assessments and {len(manifest['cases'])} cases.")
        if __package__:
            from . import expression_coverage
        else:
            import expression_coverage
        expression_coverage.generate(args.check)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
