#!/usr/bin/env python3
"""Measure archived synthesis cases and render their specification-size report."""

import argparse
import hashlib
import json
import os
import re
import statistics
from pathlib import Path

if __package__:
    from .paths import HERE, ROOT
    from .synthesis_cases import Rule
else:
    from paths import HERE, ROOT
    from synthesis_cases import Rule


# A common lexical convention for both languages, rather than model/BPE tokens.
# Preserve strings before recognizing comments; retain newlines when erasing them.
LEXER = re.compile(
    r'(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
    r'|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<identifier>[A-Za-z_]\w*)'
    r'|(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    r'|(?P<operator><=>|>>=|<<=|\.\.\.|::|->|=>|==|!=|<=|>=|&&|\|\||'
    r'\+\+|--|\+=|-=|\*=|/=|%=|<<|>>|\.\.)'
    r'|(?P<other>[^\s])'
)
RESULT = HERE / "results/rule_specification_complexity.json"
REPORT = HERE / "reports/rule_specification_complexity.md"
SOURCES = HERE / "data/rule_specification_sources.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(source):
    return [m[0] for m in LEXER.finditer(source) if m.lastgroup != "comment"]


def metrics(source):
    cleaned = LEXER.sub(
        lambda m: "\n" * m[0].count("\n") if m.lastgroup == "comment" else m[0],
        source,
    )
    return {"lines": sum(bool(line.strip()) for line in cleaned.splitlines()),
            "tokens": len(tokens(source))}


def body(source):
    rule = Rule(source)
    start = rule.values.index("rule")
    opening = rule.values.index("{", start)
    end = rule.closing(opening, "{", "}")
    return source[rule.tokens[start].start():rule.tokens[end].end()]


def span_lines(ranges, size):
    selected = set()
    for first, last in ranges:
        if not 1 <= first <= last <= size:
            raise ValueError(f"invalid inclusive source range: {first}-{last}")
        selected.update(range(first, last + 1))
    return selected


def excerpt(lines, selected):
    return "\n".join(lines[i - 1] for i in sorted(selected)) + "\n"


def totals(rows, key):
    return {metric: sum(row[key][metric] for row in rows)
            for metric in ("lines", "tokens")}


def reduction(before, after):
    return 100 * (before - after) / before


def measure():
    synthesis_path = HERE / "results/synthesis.json"
    classification_path = HERE / "data/synthesis_classification.json"
    verification_path = HERE / "results/verification.json"
    catalog_path = HERE.parent / "catalog/rules.json"
    data = read(synthesis_path)
    classification = read(classification_path)
    catalog = {r["id"]: r for r in read(catalog_path)["rules"]}
    source_map = read(SOURCES)
    upstream_path = ROOT / source_map["source_file"]
    if sha(upstream_path) != source_map["source_sha256"]:
        raise ValueError("XLA source differs from the reviewed snapshot")
    upstream_lines = upstream_path.read_text().splitlines()
    supported = [r for r in data["results"] if r["category"] <= 2]
    eligible = [r for r in supported if r["rhs_descriptors"]]
    accepted = [r for r in eligible if r["outcome"] == "full_auto"]
    if any(r["outcome"] == "partial_auto" for r in eligible):
        raise ValueError("report selection needs updating for partial-only successes")
    if not accepted:
        raise ValueError("no measured successful full-auto cases")
    represented = {r["catalog_id"] for r in accepted}
    if represented != set(source_map["schemas"]):
        raise ValueError("reviewed XLA mappings must cover exactly the selected schemas")
    if classification["cases"] != [
            {key: r[key] for key in classification["cases"][i]}
            for i, r in enumerate(data["results"])]:
        raise ValueError("archived classifications differ from synthesis results")

    dependencies = {}

    def depend(path):
        dependencies[path.relative_to(ROOT).as_posix()] = sha(path)

    for path in (synthesis_path, classification_path, verification_path,
                 catalog_path, SOURCES, upstream_path, Path(__file__).resolve(),
                 HERE / "scripts/synthesis_cases.py", HERE / "scripts/paths.py",
                 HERE / "scripts/hlo_support.py",
                 HERE.parent / "tepl/dialects/hlo.tepl"):
        depend(path)
    cases = []
    for row in accepted:
        explicit_path = HERE.parent / "tepl/rules/semantic" / (row["case"] + ".tepl")
        auto_path = ROOT / row["successful_variant"]
        if sha(explicit_path) != row["file_sha256"]:
            raise ValueError("explicit case changed since measurement: " + row["case"])
        attempts = [a for a in row["attempts"]
                    if a["mode"] == "full_auto" and a["status"] == "success"]
        if len(attempts) != 1:
            raise ValueError("need one successful full-auto attempt per selected case")
        attempt = attempts[0]
        if (sha(auto_path) != attempt["file_sha256"]
                or attempt["file"] != row["successful_variant"]
                or attempt["returncode"] != 0
                or attempt["compile_returncode"] != 0
                or attempt["validation_returncode"] != 0
                or not attempt["validated_json"]
                or "status=proved" not in attempt["stdout"]
                or "additional-guard: true" not in attempt["stdout"]
                or attempt["constructors"] != row["rhs_descriptors"]):
            raise ValueError("invalid archived acceptance evidence: " + row["case"])
        explicit, auto = explicit_path.read_text(), auto_path.read_text()
        parsed = Rule(explicit)
        generated = parsed.auto_source(range(row["rhs_descriptors"]))
        if tokens(generated) != tokens(auto):
            raise ValueError("auto variant changes more than descriptor erasure: " + row["case"])
        if len(parsed.attribute_sites()) != row["rhs_descriptors"]:
            raise ValueError("RHS descriptor count mismatch")
        if any(op.attribute is not None for op in Rule(auto).rhs.operations()):
            raise ValueError("full-auto variant still has explicit RHS descriptors")
        depend(explicit_path)
        depend(auto_path)
        spans = source_map["schemas"][row["catalog_id"]]
        lines = span_lines(spans["ranges"], len(upstream_lines))
        explicit_body, auto_body = body(explicit), body(auto)
        cases.append({
            "case": row["case"], "catalog_id": row["catalog_id"],
            "category": row["category"], "scope": row["scope"],
            "equation": catalog[row["catalog_id"]]["rewrite"],
            "explicit_status": row["explicit_status"],
            "inferred_descriptors": row["rhs_descriptors"],
            "explicit_file": explicit_path.relative_to(ROOT).as_posix(),
            "auto_file": auto_path.relative_to(ROOT).as_posix(),
            "explicit_body": metrics(explicit_body), "auto_body": metrics(auto_body),
            "explicit_file_metrics": metrics(explicit), "auto_file_metrics": metrics(auto),
            "explicit_derive_statements": len(parsed.statements()),
            "auto_derive_statements": len(Rule(auto).statements()),
            "xla_ranges": spans["ranges"], "xla_scope_note": spans["note"],
            "xla_excerpt": metrics(excerpt(upstream_lines, lines)),
        })
    xla_lines, helper_lines = set(), set()
    for entry in source_map["schemas"].values():
        xla_lines.update(span_lines(entry["ranges"], len(upstream_lines)))
    for entry in source_map["shared_helpers"]:
        helper_lines.update(span_lines(entry["ranges"], len(upstream_lines)))
    compiler_drift = sorted(
        path for path, expected in data["dependencies"].items()
        if path.startswith("src/")
        and (not (ROOT / path).exists() or sha(ROOT / path) != expected))
    # Track current compiler hashes so --check also notices changes to this caveat.
    for path in data["dependencies"]:
        if path.startswith("src/") and (ROOT / path).exists():
            depend(ROOT / path)
    result = {
        "schema_version": 1, "archived_run_at_utc": data["run_at_utc"],
        "compiler_sha256": data["compiler_sha256"],
        "archived_tepl_revision": data["tepl_revision"],
        "compiler_source_drift": compiler_drift,
        "selection": "All successful full-auto cases from the archived synthesis run",
        "metric": "Nonblank physical lines after comment removal; common lexical tokens, not BPE tokens",
        "coverage": {
            "cases": len(data["results"]),
            "schemas": len({r["catalog_id"] for r in data["results"]}),
            "supported": len(supported), "missing_semantics": len(data["results"]) - len(supported),
            "explicit_proofs_supported": sum(r["explicit_status"] == "proved" for r in supported),
            "attribute_bearing": len(eligible), "attribute_free": len(supported) - len(eligible),
            "inferred": len(cases), "inferred_schemas": len(represented),
            "theorem_eligible": sum(r["category"] == 1 for r in eligible),
            "theorem_inferred": sum(r["category"] == 1 for r in cases),
            "beyond_eligible": sum(r["category"] == 2 for r in eligible),
            "beyond_inferred": sum(r["category"] == 2 for r in cases),
            "inference_only_proofs": sum(r["explicit_status"] != "proved" for r in cases),
        },
        "totals": {key: totals(cases, key) for key in
                   ("explicit_body", "auto_body", "explicit_file_metrics", "auto_file_metrics")},
        "descriptors_removed": sum(r["inferred_descriptors"] for r in cases),
        "derive_statements_removed": sum(r["explicit_derive_statements"] - r["auto_derive_statements"] for r in cases),
        "xla_unique_excerpts": metrics(excerpt(upstream_lines, xla_lines)),
        "xla_additional_local_helpers": metrics(excerpt(upstream_lines, helper_lines - xla_lines)),
        "xla_with_local_helpers": metrics(excerpt(upstream_lines, xla_lines | helper_lines)),
        "cases": cases,
        "unsuccessful_cases": [r["case"] for r in eligible if r["outcome"] not in {"full_auto", "partial_auto"}],
        "dependencies": dependencies,
    }
    for name in ("lines", "tokens"):
        reductions = [reduction(r["explicit_body"][name], r["auto_body"][name]) for r in cases]
        result.setdefault("per_case_reductions", {})[name] = {
            "median_percent": statistics.median(reductions),
            "min_percent": min(reductions), "max_percent": max(reductions)}
    return result


def link(path):
    return Path(os.path.relpath(ROOT / path, REPORT.parent)).as_posix()


def frac(n, d):
    return f"{n}/{d} ({100*n/d:.1f}%)"


def render(result):
    c, t = result["coverage"], result["totals"]
    source_map = read(SOURCES)
    upstream = (ROOT / source_map["source_file"]).read_text().splitlines()
    e, a = t["explicit_body"], t["auto_body"]
    x = result["xla_unique_excerpts"]
    helpers = result["xla_additional_local_helpers"]
    together = result["xla_with_local_helpers"]
    line_reduction = reduction(e["lines"], a["lines"])
    token_reduction = reduction(e["tokens"], a["tokens"])
    out = [
        "# XLA Rewrite Specification Size and Attribute Synthesis", "",
        f"Across **{c['inferred']} successfully synthesized concrete cases representing "
        f"{c['inferred_schemas']} catalog schemas**, inference reduces TEPL rule specifications "
        f"from **{e['lines']:,} to {a['lines']:,} noncomment lines ({line_reduction:.1f}%)** "
        f"and from **{e['tokens']:,} to {a['tokens']:,} lexical tokens ({token_reduction:.1f}%)**. "
        f"It removes **{result['descriptors_removed']} explicit RHS descriptor occurrences**, "
        "including copied descriptors and computed descriptors, while preserving the equations, "
        "input declarations and written conditions.", "",
        "## Evaluation population", "",
        f"The implementation contains **{c['cases']} concrete cases representing {c['schemas']} "
        f"XLA rewrite schemas**. Of these, **{c['supported']} cases** have the operation semantics "
        f"required by the verifier; **{c['missing_semantics']}** lack required semantics. Among "
        f"supported cases, **{c['attribute_bearing']} contain RHS descriptors to synthesize**. "
        f"The other **{c['attribute_free']}** require no RHS descriptors: their replacement uses "
        "attribute-free operations, such as elementwise addition, or returns an existing matched "
        "value, as in `add(X, 0) => X`.", "",
        "| Measurement | Result |", "|---|---:|",
        f"| Semantically supported cases | {frac(c['supported'], c['cases'])} |",
        f"| Supported cases proved with explicit attributes | {frac(c['explicit_proofs_supported'], c['supported'])} |",
        f"| Supported cases containing RHS descriptors | {frac(c['attribute_bearing'], c['supported'])} |",
        f"| Successful full attribute inference | **{frac(c['inferred'], c['attribute_bearing'])}** |",
        f"| Attribute-bearing cases classified within the theorem fragment | {frac(c['theorem_eligible'], c['attribute_bearing'])} |",
        f"| Successful inference within that fragment | **{frac(c['theorem_inferred'], c['theorem_eligible'])}** |",
        f"| Successful inference beyond that fragment | **{frac(c['beyond_inferred'], c['beyond_eligible'])}** |", "",
        "The theorem classification uses the written operations and read-once/component structure. "
        "The mathematical guarantee also assumes extents at least two and ideal arithmetic; the "
        "experiment preserves the original case conditions and does not add those extent restrictions. "
        "The contraction/permutation implementation has an ordered complete path; sum/addition cases "
        "can use general verification. Thus 10/10 is an empirical result on classified examples, "
        "not an exhaustive test of completeness. Some excluded operations could potentially be "
        "re-encoded using theorem primitives; that alternative encoding was not evaluated.", "",
        f"Each inference success in the archived run required `status=proved`, "
        "`additional-guard: true`, the expected inferred descriptor count, and successful executable "
        f"generation and JSON validation. In **{c['inference_only_proofs']} cases**, an inferred "
        "implementation verifies although the original explicit formulas remain unproved. "
        "The seven unsuccessful attribute-bearing cases remain in the coverage denominator; code-size "
        "reductions below are conditional on successful full inference.", "",
        "## Measured representation sizes", "",
        "| Representation | Total noncomment lines | Total lexical tokens |",
        "|---|---:|---:|",
        f"| XLA implementation excerpts, shared lines counted once¹ | {x['lines']:,} | {x['tokens']:,} |",
        f"| TEPL with explicit RHS attributes, {c['inferred']} rule bodies | {e['lines']:,} | {e['tokens']:,} |",
        f"| TEPL with inferred RHS attributes, the same {c['inferred']} rule bodies | **{a['lines']:,}** | **{a['tokens']:,}** |", "",
        f"¹ The XLA row is manually selected source context for the same {c['inferred_schemas']} represented schemas. "
        "Its branches/helpers retain more general ranks, operand variants, graph traversal and "
        "policy than several concrete TEPL cases. They are not 36 independent scope-equivalent "
        "implementations. General visitor setup and library implementations are excluded. "
        "The source row is descriptive; dividing it by the TEPL rows would not isolate a language "
        "or inference benefit. The paired TEPL comparison does hold rule scope fixed.", "",
        f"Within TEPL, synthesis removes **{e['lines']-a['lines']:,} noncomment lines** and "
        f"**{e['tokens']-a['tokens']:,} tokens**. The source contains "
        f"**{result['derive_statements_removed']} removed `derive` statements**, counting both "
        "descriptor assignments and their local metadata calculations. Descriptor occurrences "
        "include inline reuse such as `dot[attr_inner.attrs]`, so not every inferred descriptor "
        "corresponds to a separate assignment or a removed physical line.", "",
        "| Scope of paired TEPL measurement | Explicit lines | Inferred lines | Line reduction | Explicit tokens | Inferred tokens | Token reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("Rule bodies; primary measurement", result["cases"]),
              ("Theorem-fragment rule bodies", [r for r in result["cases"] if r["category"] == 1]),
              ("Beyond-fragment rule bodies", [r for r in result["cases"] if r["category"] == 2])]
    for name, rows in groups:
        before, after = totals(rows, "explicit_body"), totals(rows, "auto_body")
        out.append(f"| {name} | {before['lines']:,} | {after['lines']:,} | "
                   f"{reduction(before['lines'],after['lines']):.1f}% | {before['tokens']:,} | "
                   f"{after['tokens']:,} | {reduction(before['tokens'],after['tokens']):.1f}% |")
    before, after = t["explicit_file_metrics"], t["auto_file_metrics"]
    out.append(f"| Whole rule files, including imports | {before['lines']:,} | {after['lines']:,} | "
               f"{reduction(before['lines'],after['lines']):.1f}% | {before['tokens']:,} | "
               f"{after['tokens']:,} | {reduction(before['tokens'],after['tokens']):.1f}% |")
    out += ["", "| Per-case reduction in rule-body size | Median | Minimum | Maximum |",
            "|---|---:|---:|---:|"]
    for name in ("lines", "tokens"):
        stats = result["per_case_reductions"][name]
        out.append(f"| {name.capitalize()} | {stats['median_percent']:.1f}% | "
                   f"{stats['min_percent']:.1f}% | {stats['max_percent']:.1f}% |")
    out += ["", "Physical lines depend on formatting. Some metadata formulas occupy one long "
            "line, while inline attribute erasure can reduce tokens without removing a line. "
            "Tokens therefore provide a useful second measurement. Code size measures "
            "specification burden, not development time, maintainability, proof-checking speed "
            "or optimized-program performance.", "",
            "## What logic disappears", ""]
    by_case = {r["case"]: r for r in result["cases"]}
    descriptions = [
        ("transpose_compose", "Transpose composition",
         "XLA builds a replacement transpose and calls `ComposePermutations`. Explicit TEPL "
         "writes `gather(inner.permutation, outer.permutation)` in a descriptor; inferred TEPL "
         "leaves the replacement permutation unspecified. Here the benefit is removal of a "
         "small but necessary metadata expression."),
        ("catalog_broadcast_transpose_broadcast", "Transpose of a broadcast",
         "The XLA helper inverts the permutation, maps broadcast axes, checks ordering/layout "
         "policy, and builds the replacement. The explicit TEPL algebraic case computes an "
         "inverse map, surviving/new axis positions, and output extents. Inference removes "
         "those calculations, leaving the input declaration and graph equation. The explicit "
         "formula is unproved; the inferred construction has an archived proof. Source layout "
         "policy remains outside this TEPL case."),
        ("dot_swap_operands", "Swapping dot operands",
         "XLA swaps dimension-number lists, constructs a dot, and computes an output permutation "
         "from batch/free axis counts. The TEPL case selects ordinary rank-two matrix products; "
         "inference removes the two RHS descriptors. Its LHS rank/axis conditions remain written. "
         "XLA's parameter-placement and precision policy are not implemented by this algebraic identity."),
    ]
    for name, title, description in descriptions:
        row = by_case[name]
        source = (ROOT / row["explicit_file"]).read_text()
        auto = (ROOT / row["auto_file"]).read_text()
        out += [f"### {title}", "", description, ""]
        example_ranges = {
            "transpose_compose": [[10248, 10254]],
            "catalog_broadcast_transpose_broadcast": [[2822, 2829], [2859, 2862]],
            "dot_swap_operands": [[2967, 2985]],
        }
        pieces = ["\n".join(upstream[lo - 1:hi]) for lo, hi in example_ranges[name]]
        out += ["Selected XLA metadata/construction excerpt "
                "(surrounding logic is omitted here; measured ranges are listed below):", "",
                "```cpp", "\n// ... surrounding source omitted ...\n".join(pieces), "```", "",
                f"Measured rule body: **{row['explicit_body']['lines']} → {row['auto_body']['lines']} "
                f"lines; {row['explicit_body']['tokens']} → {row['auto_body']['tokens']} tokens**.", "",
                f"Explicit TEPL ([source]({link(row['explicit_file'])})):", "", "```tepl",
                body(source), "```", "",
                f"Inferred TEPL ([source]({link(row['auto_file'])})):", "", "```tepl",
                body(auto), "```", ""]
    dialect = HERE.parent / "tepl/dialects/hlo.tepl"
    dialect_size = metrics(dialect.read_text())
    out += [
        "## Counting and source attribution", "",
        "- **Selection:** every successful full-auto case in the saved synthesis results, with "
        "no selection based on code-size reduction. Multiple concrete cases for one schema remain "
        "separate in the paired TEPL measurements.",
        "- **TEPL rule bodies:** count from `rule` through its closing brace, including input "
        "declarations, both graph patterns, `where` conditions and `derive` blocks. Exclude imports, "
        "blank lines and comments. Whole-file totals above show the effect of including imports.",
        "- **Lines:** physical lines containing code after string-aware removal of `//` and "
        "`/* ... */` comments. Braces on their own lines count. No reformatting or manual shrinking "
        "is performed.",
        "- **Tokens:** use the same lexical convention for C++ and TEPL: identifiers, numbers, "
        "quoted strings, recognized multicharacter operators and remaining punctuation. Whitespace "
        "and comments do not count. These are reproducible lexical units, not compiler-specific "
        "token streams or language-model tokens.",
        "- **XLA excerpts:** manually identify relevant branches or dedicated helpers rather than "
        "counting entire visitors such as `HandleDot`. Ranges are inclusive and pinned by the "
        "snapshot hash. Union selected lines across all cases before counting: shared branches "
        "are counted once. Per-case XLA counts in the appendix overlap and must not be summed.",
        "- **Additional support:** the main XLA row excludes generic libraries and common visitor "
        "setup. A selected inventory of ten local helpers adds "
        f"**{helpers['lines']} lines / {helpers['tokens']:,} tokens**; the deduplicated union with "
        f"the excerpts is **{together['lines']:,} lines / {together['tokens']:,} tokens**. "
        "This is not a complete transitive dependency count. External helpers such as "
        "`ComposePermutations`, `InversePermutation`, shape utilities and graph-construction APIs "
        "are not expanded.",
        "- **Shared TEPL support:** the local semantic HLO dialect is "
        f"**{dialect_size['lines']:,} noncomment lines / {dialect_size['tokens']:,} tokens**, "
        "reported once and excluded from per-rule totals. The compiler, solver and shared "
        "attribute-construction machinery are also excluded. The comparison measures code "
        "written per rewrite after this shared infrastructure is available.", "",
        "The directly attributable inference saving is the explicit-to-inferred TEPL difference. "
        "The XLA excerpts illustrate the imperative work associated with these transformations, "
        "but their additional generality prevents an equal-functionality percentage-reduction "
        "claim. A future scope-matched XLA baseline would need to implement the same scoped "
        "equations and assumptions explicitly.", "",
        "## Paper-ready result", "",
        f"> We evaluate attribute synthesis on {c['cases']} concrete rewrite cases representing "
        f"{c['schemas']} XLA schemas. Of the {c['supported']} cases with supported operation "
        f"semantics, {c['attribute_bearing']} contain RHS attribute descriptors. Synthesis succeeds "
        f"on {c['inferred']}/{c['attribute_bearing']} cases "
        f"({100*c['inferred']/c['attribute_bearing']:.1f}%), including 10/10 cases classified "
        "within the theorem fragment and 26/33 beyond it. Across the 36 successful cases, "
        f"inference eliminates {result['descriptors_removed']} explicit descriptor occurrences "
        f"and reduces rule-body size from {e['lines']} to {a['lines']} noncomment lines "
        f"({line_reduction:.1f}%) and from {e['tokens']:,} to {a['tokens']:,} lexical tokens "
        f"({token_reduction:.1f}%), preserving the written equations and conditions. "
        "Four inferred implementations verify although the original explicit metadata formulas "
        "remain unproved. These results quantify reduced per-rewrite specification burden "
        "under the evaluated algebraic model.", "",
        "## Per-case measurements", "",
        "TEPL columns count rule bodies. `T` denotes the theorem-fragment classification; `G` "
        "denotes supported cases beyond it. Source links lead to the actual measured variants. "
        "The XLA column gives excerpt lines/tokens and overlaps between cases.", "",
        "| Case | Scope | RHS descriptors inferred | Explicit lines / tokens | Inferred lines / tokens | XLA excerpt lines / tokens |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for row in result["cases"]:
        eb, ab, xb = row["explicit_body"], row["auto_body"], row["xla_excerpt"]
        out.append(f"| [{row['case']}]({link(row['auto_file'])}) | "
                   f"{'T' if row['category']==1 else 'G'} | {row['inferred_descriptors']} | "
                   f"[{eb['lines']} / {eb['tokens']}]({link(row['explicit_file'])}) | "
                   f"{ab['lines']} / {ab['tokens']} | {xb['lines']} / {xb['tokens']} |")
    out += ["", "Unsuccessful eligible cases: " + ", ".join(
        f"`{case}`" for case in result["unsuccessful_cases"]) + ".", "",
        "## XLA source mapping and scope differences", "",
        f"All ranges refer to [{Path(source_map['source_file']).name}]"
        f"({link(source_map['source_file'])}) in the pinned XLA revision "
        "`e02127ba9c9c48912b0e6776fe5bfe3843d81042`. The machine-readable "
        "[source map](../data/rule_specification_sources.json) records every inclusive range.", "",
        "| Catalog schema | Source lines | Attribution and scope |",
        "|---|---|---|",
    ]
    for schema, entry in source_map["schemas"].items():
        ranges = ", ".join(f"{lo}–{hi}" for lo, hi in entry["ranges"])
        out.append(f"| `{schema}` | {ranges} | {entry['note']} |")
    out += ["", "Selected local helper ranges, counted separately:", ""]
    for helper in source_map["shared_helpers"]:
        ranges = ", ".join(f"{lo}–{hi}" for lo, hi in helper["ranges"])
        out.append(f"- `{helper['name']}`: {ranges}.")
    out += ["", "## Provenance and reproduction", "",
        f"Inference outcomes refer to the saved run at **{result['archived_run_at_utc']}**, "
        f"with compiler revision `{result['archived_tepl_revision']}` and compiler SHA-256 "
        f"`{result['compiler_sha256']}`. This measurement verifies each selected explicit and "
        "inferred rule against its archived file hash, checks the saved proof/acceptance evidence, "
        "and checks that generated variants differ only by the existing descriptor-erasure "
        "transformation. It does not rerun the semantic sweep.", ""]
    if result["compiler_source_drift"]:
        out += [f"**Current compiler freshness:** {len(result['compiler_source_drift'])} compiler "
                "source dependencies differ from the archived run. The success rates describe "
                "that archived compiler, not a fresh measurement of the current compiler. "
                "The selected rule files still match their archived hashes. Changed paths:", ""]
        out.extend(f"- `{path}`" for path in result["compiler_source_drift"])
        out.append("")
    else:
        out += ["All recorded compiler source dependencies match the archived run.", ""]
    out += ["From the repository root:", "", "```sh",
            "python3 labs/xla_rewrite/coverage/scripts/measure_rule_specification.py",
            "python3 labs/xla_rewrite/coverage/scripts/measure_rule_specification.py --check",
            "python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_rule_specification.py'",
            "```", "",
            "The runner writes [measurements and input hashes](../results/rule_specification_complexity.json) "
            "and this report. `--check` recomputes both without writing and detects stale outputs. "
            "For a fresh compiler evaluation, follow the [existing experiment reproduction steps]"
            "(../README.md#reproduce), review any changed population/source mappings, then rerun "
            "this measurement.", ""]
    return "\n".join(line.rstrip() for line in "\n".join(out).split("\n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check outputs without writing")
    args = parser.parse_args()
    result = measure()
    outputs = {RESULT: json.dumps(result, indent=2) + "\n", REPORT: render(result)}
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit("stale measurement output: " + str(path))
        else:
            path.write_text(content)
    print("Rule specification measurements " + ("are current." if args.check else "written."))


if __name__ == "__main__":
    main()
