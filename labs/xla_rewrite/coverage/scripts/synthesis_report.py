#!/usr/bin/env python3
"""Validate measurements and render the XLA attribute synthesis report."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __package__:
    from .paths import HERE, ROOT, TEPL
    from . import migrate_syntax
else:
    from paths import HERE, ROOT, TEPL
    import migrate_syntax


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def successful(row, mode):
    return any(a["mode"] == mode and a["status"] == "success" for a in row["attempts"])


def validate(data, classification):
    migrate_syntax.validate()
    rows = data["results"]
    expected = {r["case"]: r for r in classification["cases"]}
    if len(rows) != len(expected) or {r["case"] for r in rows} != set(expected):
        raise ValueError("synthesis results must cover each classified case exactly once")
    for row in rows:
        if any(row[key] != expected[row["case"]][key] for key in expected[row["case"]]):
            raise ValueError("classification changed after measurement")
        if row["category"] == 3 and row["attempts"]:
            raise ValueError("unsupported cases must not enter auto measurements")
        if not row["rhs_descriptors"] and row["attempts"]:
            raise ValueError("attribute-free cases must not count as inference")
        full, partial = successful(row, "full_auto"), successful(row, "partial_auto")
        expected_outcome = "full_auto" if full else "partial_auto" if partial else None
        if expected_outcome and row["outcome"] != expected_outcome:
            raise ValueError("full/partial overlap must count once")
        if row["outcome"] in {"full_auto", "partial_auto"} and not expected_outcome:
            raise ValueError("synthesis success lacks a successful attempt")
        for attempt in row["attempts"]:
            if attempt["status"] != "success":
                continue
            if (attempt["returncode"] != 0 or not attempt["validated_json"]
                    or attempt.get("compile_returncode") != 0
                    or attempt.get("validation_returncode") != 0
                    or attempt["constructors"] != len(attempt["selected"])
                    or "status=proved" not in attempt["stdout"]
                    or "additional-guard: true" not in attempt["stdout"]):
                raise ValueError("success requires proof, inferred constructors, compilation, and JSON validation")
    for path, expected_hash in data["dependencies"].items():
        if sha(ROOT / path) != expected_hash:
            raise ValueError("stale synthesis dependency: " + path)


def fraction(numerator, denominator):
    return f"{numerator}/{denominator} ({100*numerator/denominator:.1f}%)" if denominator else "—"


def link(path):
    return Path(os.path.relpath(ROOT / path, HERE / "reports")).as_posix()


def render(data, classification):
    rows = data["results"]
    supported = [r for r in rows if r["category"] <= 2]
    eligible = [r for r in supported if r["rhs_descriptors"]]
    accepted = [r for r in eligible if r["outcome"] in {"full_auto", "partial_auto"}]
    full = [r for r in eligible if successful(r, "full_auto")]
    partial = [r for r in eligible if successful(r, "partial_auto")]
    partial_only = [r for r in accepted if r["outcome"] == "partial_auto"]
    explicit_attr = [r for r in eligible if r["explicit_status"] == "proved"]
    explicit_automated = [r for r in explicit_attr if r in accepted]
    theorem_baseline = [r for r in explicit_attr if r["category"] == 1]
    theorem_automated = [r for r in theorem_baseline if r in accepted]
    recovered = [r for r in accepted if r["explicit_status"] != "proved"]
    attr_free = [r for r in supported if not r["rhs_descriptors"]]
    attr_free_proved = [r for r in attr_free if r["explicit_status"] == "proved"]
    practical = [r for r in supported if r["explicit_status"] == "proved" or r in accepted]
    compilation = read(HERE / "results/compilation.json")
    old = read(HERE / "baselines/pre_syntax/verification.json")
    explicit = read(HERE / "results/verification.json")
    old_proved = {r["case"] for r in old["results"] if r["status"] == "proved"}
    new_proved = {r["case"] for r in explicit["results"] if r["status"] == "proved"}
    inference_schemas = {r["catalog_id"] for r in accepted}
    o = ["# XLA Attribute Synthesis Evaluation", "",
         f"TEPL automatically infers RHS attributes for **{fraction(len(explicit_automated),len(explicit_attr))}** "
         "of the attribute-bearing cases whose explicit baseline already verifies. "
         f"This includes **{len(theorem_automated)}/{len(theorem_baseline)} verified theorem-fragment cases**. "
         f"Across all supported attribute-bearing cases, including cases with unproved explicit baselines, "
         f"auto succeeds on **{fraction(len(accepted),len(eligible))}**, spanning "
         f"**{len(inference_schemas)} catalog schemas**.", "",
         f"The experiment infers RHS attributes for **{len(accepted)} cases** "
         f"in total. Full auto succeeds on **{len(full)} cases**; "
         f"partial auto succeeds on **{len(partial)}**, including **{len(partial_only)} partial-only successes**. "
         "Overlapping full and partial results count once.", "",
         f"Combining verified explicit rules, inferred rules, and attribute-free identities gives "
         f"**{fraction(len(practical),len(supported))} supported cases** with a semantic proof. "
         f"**{len(recovered)} cases gain a verified implementation through inference even though their explicit metadata formulas remain unproved.**", "",
         "## Experiment setup", "",
         "The evaluation follows three steps: migrate the existing corpus, freeze theorem-based categories, "
         "then measure explicit, full-auto, and partial-auto variants. It uses the ideal `algebraic` policy. "
         "A successful auto variant has an explicit semantic proof, the expected number of inferred descriptors, "
         "successful executable compilation, and validated JSON. Equations, input declarations, and written "
         "conditions are preserved. No suggested restrictions are added.", "",
         "The corpus contains **239 catalog schemas**, **209 concrete cases representing 199 schemas**, "
         "and one negative control. Concrete cases retain the scopes recorded in the existing manifest: "
         "some specialize ranks, arithmetic operations, layouts, or constant representations. Schema coverage "
         "means at least one measured case, not every XLA variant.", "",
         "## Step 1 Syntax migration", "",
         "**136 legacy semantic cases and their 136 structural counterparts were migrated.** "
         "Node labels and `.attrs` replace `@` bindings; original equations, conditions, attribute sharing, "
         "and metadata formulas are preserved. Original text and hashes are retained in "
         "[migration snapshots](../baselines/pre_syntax/) and the [migration record](../data/syntax_migration.json).", "",
         f"All **{len(compilation['results'])} structural cases compile and their executable JSON validates**. "
         f"The semantic baseline verifies **{len(new_proved)} cases**, retaining "
         f"**{len(old_proved & new_proved)}/{len(old_proved)} previously verified cases**. "
         "The deliberately incorrect subtraction-to-addition control is rejected.", "",
         "## Step 2 Classification before testing", "",
         "[synthesis_classification.json](../data/synthesis_classification.json) is frozen before inference runs. "
         "Classification uses declarations and graph structure, never proof outcomes. Metadata functions such "
         "as `gather(...)` are distinguished from HLO graph operations.", "",
         "| Category | Catalog schemas | Concrete cases | Cases with RHS descriptors | Attribute-free cases |",
         "|---|---:|---:|---:|---:|"]
    names = {1: "1 Theorem fragment", 2: "2 Supported beyond theorem", 3: "3 Unsupported"}
    for category in (1, 2, 3):
        group = [r for r in rows if r["category"] == category]
        schemas = [r for r in classification["schemas"] if r["category"] == category]
        o.append(f"| {names[category]} | {len(schemas)} | {len(group)} | "
                 f"{sum(r['rhs_descriptors']>0 for r in group)} | {sum(not r['rhs_descriptors'] for r in group)} |")
    o += ["", "Category 1 follows the theorem's contraction, permutation, unary-sum, and separated-addition "
          "fragment, including read-once components and matching distinct input sets. The theorem's comparison "
          "domain assumes extents at least two and ideal arithmetic; this experiment adds no extent restriction "
          "to the original TEPL conditions. Accepted rules may therefore cover a broader domain. "
          "The current ordered complete path covers contraction/permutation; unary-sum and additive cases may "
          "use general verification. These measurements report coverage, not an exhaustive completeness evaluation.", "",
          "Category 2 has supported local semantics but lies outside those theorem assumptions. "
          "Category 3 requires missing semantics or other unsupported capabilities and is excluded from auto attempts. Forty catalog schemas "
          "have no concrete case; their porting state is recorded separately, and they contribute no measured successes. "
          "Classification describes the scoped algebraic evaluation model, not a complete HLO importer.", "",
          "## Step 3 Explicit full auto and partial auto", "",
          "Full auto removes all RHS descriptor definitions. Partial auto retains explicit descriptors on "
          "some nodes and infers the rest. Both modes are measured for cases with multiple RHS descriptors. "
          "Partial subsets are tried in descending number of inferred nodes, stopping after the first successful "
          "cardinality. Partial success means at least one whole descriptor was inferred; field-level pins are not used.", "",
          "| Category | Attribute cases | Explicit verified | Full auto | Partial auto including overlap | Partial only | Full or partial success |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for category in (1, 2):
        group = [r for r in eligible if r["category"] == category]
        o.append(f"| {names[category]} | {len(group)} | {sum(r['explicit_status']=='proved' for r in group)} | "
                 f"{sum(successful(r,'full_auto') for r in group)} | {sum(successful(r,'partial_auto') for r in group)} | "
                 f"{sum(r['outcome']=='partial_auto' for r in group)} | "
                 f"{fraction(sum(r['outcome'] in {'full_auto','partial_auto'} for r in group),len(group))} |")
    o += [f"| Total | {len(eligible)} | {len(explicit_attr)} | {len(full)} | {len(partial)} | {len(partial_only)} | {fraction(len(accepted),len(eligible))} |", "",
          f"The **{fraction(len(attr_free_proved),len(attr_free))} attribute-free cases** that verify "
          "are useful semantic verification coverage and are counted separately from attribute inference.", "",
          "### Comparison on the verified explicit baseline", "",
          "This comparison holds the availability of an explicit semantic proof fixed, isolating how much "
          "attribute-writing work can be automated on already verified cases. The overall table above retains "
          "all supported attribute-bearing cases, including those with an unproved explicit baseline.", "",
          "| Category | Explicit-verified attribute cases | Auto successes on those same cases |",
          "|---|---:|---:|"]
    for category in (1, 2):
        group = [r for r in explicit_attr if r["category"] == category]
        o.append(f"| {names[category]} | {len(group)} | {fraction(sum(r in accepted for r in group),len(group))} |")
    o += [f"| Total | {len(explicit_attr)} | {fraction(len(explicit_automated),len(explicit_attr))} |", "",
          "### Practical interpretation", "",
          "Theorem-fragment cases show how much the current implementation automates under the original "
          "XLA-case conditions. Beyond-theorem successes demonstrate practical reach of candidate construction "
          "plus independent verification. A failed proof is recorded as unproved; it does not establish that "
          "the rewrite is false or that no suitable attributes exist. All reported auto successes pass the same "
          "compiler acceptance checks used for ordinary rules.", "",
          "### Cases improved by inference", ""]
    for row in recovered:
        o.append(f"- [{row['case']}]({link(row['successful_variant'])}): {row['inferred_descriptors']} "
                 f"RHS descriptor(s) inferred; the original explicit formula was `{row['explicit_status']}`.")
    o += ["", "### Successful inferred cases", "",
          "| Case | Category | Full auto | Partial auto | Inferred descriptors in best variant | Original baseline scope |",
          "|---|---:|---|---|---:|---|"]
    for row in accepted:
        o.append(f"| [{row['case']}]({link(row['successful_variant'])}) | {row['category']} | "
                 f"{'yes' if successful(row,'full_auto') else 'no'} | {'yes' if successful(row,'partial_auto') else '—'} | "
                 f"{row['inferred_descriptors']}/{row['rhs_descriptors']} | {row['scope'].replace('|', '/')} |")
    before_path = HERE / "baselines/pre_axis_views/synthesis.json"
    if before_path.exists():
        before = read(before_path)
        previous = {r["case"] for r in before["results"]
                    if r["outcome"] in {"full_auto", "partial_auto"}}
        current = {r["case"] for r in accepted}
        previous_theorem = [r for r in before["results"]
                            if r["category"] == 1 and r["rhs_descriptors"]]
        current_theorem = [r for r in eligible if r["category"] == 1]
        previous_verification = read(HERE / "baselines/pre_axis_views/verification.json")
        previous_proofs = {r["case"] for r in previous_verification["results"]
                           if r["status"] == "proved"}
        o += ["", "## Shared axis reasoning improvement", "",
              "The compiler now reuses its coordinate-sequence expansion in set-membership reasoning. "
              "This relates registered selections and permutations to the same logical axes used by "
              "equality checking, including context-known block lengths. Coordinate witnesses remain "
              "stored in their original form. No operation names, rule names, XLA case IDs, or new "
              "applicability assumptions participate in the fix.", "",
              "The comparison uses the same migrated cases and frozen classifications. "
              "[Pre-fix measurements](../baselines/pre_axis_views/) preserve the previous compiler results.", "",
              "| Measurement | Before | After |",
              "|---|---:|---:|",
              f"| Explicit semantic proofs | {len(previous_proofs)} | {len(new_proved)} |",
              f"| Unique auto successes | {len(previous)} | {len(current)} |",
              f"| Theorem-fragment auto successes | {sum(r['case'] in previous for r in previous_theorem)}/{len(previous_theorem)} | {sum(r['case'] in current for r in current_theorem)}/{len(current_theorem)} |",
              f"| Previous semantic proofs retained | — | {len(previous_proofs & new_proved)}/{len(previous_proofs)} |",
              f"| Previous auto successes retained | — | {len(previous & current)}/{len(previous)} |", "",
              "Newly inferred cases:", ""]
        for row in accepted:
            if row["case"] not in previous:
                o.append(f"- [{row['case']}]({link(row['successful_variant'])}).")
        if previous - current or previous_proofs - new_proved:
            o += ["", "Cases losing acceptance in this run: " +
                  ", ".join(sorted((previous - current) | (previous_proofs - new_proved))) + "."]
    o += ["", "## Reproduction and artifacts", "",
          "```sh", "bazel build //:tepl",
          "python3 labs/xla_rewrite/coverage/scripts/migrate_syntax.py --check",
          "python3 labs/xla_rewrite/coverage/scripts/synthesis_cases.py",
          "python3 labs/xla_rewrite/coverage/scripts/run_checks.py",
          "python3 labs/xla_rewrite/coverage/scripts/expression_coverage.py --run",
          "python3 labs/xla_rewrite/coverage/scripts/run_synthesis.py",
          "python3 labs/xla_rewrite/coverage/scripts/synthesis_report.py",
          "python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_*.py'",
          "```", "",
          "[synthesis.json](../results/synthesis.json) contains every attempt, compiler diagnostics, "
          "constructor counts, proof output, artifact hashes, and source/input hashes. "
          "[synthesis_cases.md](synthesis_cases.md) lists all concrete outcomes, including unsupported "
          "and attribute-free cases. Generated [full-auto](../../tepl/rules/full_auto/) and "
          "[partial-auto](../../tepl/rules/partial_auto/) files preserve the source conditions.", "",
          f"- Run UTC: `{data['run_at_utc']}`.",
          f"- Compiler revision: `{data['tepl_revision']}`; working-tree source hashes are recorded.",
          f"- Timeout: {data['timeout_seconds_per_command']:g} seconds per compiler command; {data['jobs']} independent case workers.",
          "- Arithmetic model: ideal algebraic semantics. The measurements do not claim IEEE floating-point or full backend equivalence.", ""]
    return "\n".join(o)


def case_table(data):
    o = ["# XLA Synthesis Case Outcomes", "",
         "Each original case counts once. Auto success requires semantic proof, compilation, "
         "and executable JSON validation; unsupported cases are not attempted with auto.", "",
         "| Case | Category | RHS descriptors | Explicit baseline | Full auto | Partial auto | Outcome |",
         "|---|---:|---:|---|---|---|---|"]
    for row in data["results"]:
        o.append(f"| {row['case']} | {row['category']} | {row['rhs_descriptors']} | "
                 f"{row['explicit_status']} | {'success' if successful(row,'full_auto') else 'unproved' if row['attempts'] else '—'} | "
                 f"{'success' if successful(row,'partial_auto') else 'unproved' if any(a['mode']=='partial_auto' for a in row['attempts']) else '—'} | {row['outcome']} |")
    return "\n".join(o) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = read(HERE / "results/synthesis.json")
    classification = read(HERE / "data/synthesis_classification.json")
    validate(data, classification)
    for name, content in [("attribute_synthesis.md", render(data, classification)),
                          ("synthesis_cases.md", case_table(data))]:
        path = HERE / "reports" / name
        if args.check:
            if path.read_text() != content:
                raise ValueError("stale synthesis report: " + name)
        else:
            path.write_text(content)
    print("Validated and summarized", len(data["results"]), "synthesis cases.")


if __name__ == "__main__":
    main()
