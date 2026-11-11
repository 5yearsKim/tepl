#!/usr/bin/env python3
"""Refresh theorem-scope counts and measure the actual native core dispatch.

Writes a separate current report; never edits historical default-backend runs
or the paper. The artifact's marker and instruction kinds follow emit()'s
restricted-insertion / legacy-extension / unsigned-constructor dispatch.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from restricted_insertion import ROOT, migrate, run, sha

COVERAGE = Path(__file__).resolve().parents[1]
MAIN = {"main_theorem_form", "main_theorem_reduction"}
UNSIGNED = {"input", "permutation", "summation", "contraction", "addition"}


def report(result):
    counts = result["counts"]
    rows = [
        "# Current Constructive Core Coverage",
        "",
        "The paper audit classifies **21 cases as main-theorem scope: 13 direct "
        "forms + 8 restricted insertion reductions**. The current native "
        "`--core` backend compiles and portable-validates all 21. Of these, "
        "**18 use constructive metadata DAGs**, while **3 existing reversal "
        "forms still dispatch to specialized schemas**.",
        "",
        "| Classification | Cases | Measured native implementation path |",
        "|---|---:|---|",
        f"| Direct unsigned core | {counts['constructive_core']} | Existing constructive core DAG |",
        f"| Restricted insertion | {counts['restricted_insertion']} | Generic LHS normalization + constructive core DAG |",
        f"| Direct signed core forms | {counts['legacy_schema']} | Existing specialized reversal schemas |",
        "| Main-theorem scope total | 21 | 18 constructive + 3 specialized |",
        "| Additional structural candidates | 10 | Outside this main-core boundary; not rerun here |",
        "",
        "## Populations and interpretation",
        "",
        "The audit contains 38 axis-inference forms: 31 historically inferred "
        "and 7 not proved. Main-theorem scope is **21/38 (55.3%)**, or "
        "**21/31 (67.7%)** among historically successful forms. Actual "
        "constructive dispatch is **18/38 (47.4%)**, or **18/31 (58.1%)** "
        "among historically successful forms. The previous direct-form count "
        "13 is a mathematical classification; it was not evidence that all "
        "13 executed the unsigned constructive algorithm.",
        "",
        "## Case-level native dispatch",
        "",
        "| Case | Paper classification | Current path | Compile / portable validation |",
        "|---|---|---|---|",
    ]
    labels = {"constructive_core": "Constructive core", "restricted_insertion": "Generic insertion reduction",
              "legacy_schema": "Specialized schema"}
    for case in result["cases"]:
        rows.append(f"| {case['case']} | {case['classification']} | {labels[case['path']]} | Passed |")
    rows += [
        "",
        "## Evidence and limits",
        "",
        "[Current results](../results/core_classification.json) include compiler, "
        "audit, input, dialect, and dispatch-source hashes. The runner reads the "
        "paper inputs and migrates only old syntax/import qualification; source "
        "guards and supplied expressions are retained. Eight insertion forms "
        "also have bounded execution tests in `//tests:insertion_test`, including "
        "the recorded normalized unit insertion, scalar/singleton cases, "
        "incompatible extents, RHS boundary checks, and guard preservation. "
        "Compiling 21 forms is not a new full-domain completeness proof.",
        "",
        "The constructive extension requires exact-real semantics, original input "
        "extents at least two, positive supplied fresh extents, independent "
        "unknowns, and the restricted single-component insertion boundary. "
        "The broadcast/transpose entry uses the recorded companion. Arbitrary "
        "internal RHS insertion remains outside the generic path. Native "
        "`--core` support does not add C++/Rust generated-adapter support.",
        "",
        "To make **all 21** execute constructively, the remaining implementation "
        "step is to route the three pure reversal forms through the compiled "
        "signed core machinery already used inside the insertion reduction, "
        "and validate that dispatch separately.",
        "",
        "```sh",
        "bazel build //:tepl",
        "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/core_classification.py \\",
        "  --paper /path/to/TEPL-paper",
        "bazel test //tests:insertion_test --test_output=all",
        "```",
        "",
        f"Measured UTC: `{result['measured_utc']}`. Existing frozen measurements "
        "and paper files were not changed by this runner.",
        "",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    args = parser.parse_args()
    audit_path = args.paper / "evaluation/theorem-scope-audit.json"
    audit = json.loads(audit_path.read_text())
    selected = [r for r in audit["records"] if r["classification"] in MAIN]
    assert len(selected) == 21
    dialect = ROOT / "labs/attribute_inferences/xla_rewrite/tepl/dialects/hlo.tepl"
    cases = []
    with tempfile.TemporaryDirectory(prefix="tepl-core-coverage-") as temp:
        for item in selected:
            source_path = item.get("source", f"evaluation/unified-construction/inputs/{item['case']}.tepl")
            original = args.paper / source_path
            migrated = migrate(original.read_text())
            source = Path(temp) / (item["case"] + ".tepl")
            source.write_text(f'from "{dialect}" import HLO as hlo\n'
                              'use hlo::{broadcast,transpose,reverse,reduce_sum,dot}\n' + migrated)
            output = source.with_suffix(".json")
            run([args.compiler, "compile", source, "--core", "--semantic-policy", "algebraic", "-o", output])
            run([args.compiler, "validate", output])
            core = json.loads(output.read_text())["program"]["rules"][0]["core"]
            program = core["program"]
            kinds = {n["kind"] for side in ("source", "replacement") for n in program[side]}
            path = ("restricted_insertion" if program.get("restricted_insertion") else
                    "constructive_core" if kinds <= UNSIGNED else "legacy_schema")
            if item["classification"] == "main_theorem_reduction":
                assert path == "restricted_insertion" and len(core["nodes"]) == 2
            cases.append({"case": item["case"], "classification": item["classification"],
                          "source": source_path, "source_sha256": sha(original),
                          "migrated_source_sha256": sha(source), "path": path,
                          "instruction_kinds": sorted(kinds), "metadata_blocks": len(core["nodes"]),
                          "compiled": True, "portable_validated": True})
    counts = dict(Counter(c["path"] for c in cases))
    assert counts == {"constructive_core": 10, "legacy_schema": 3, "restricted_insertion": 8}, counts
    result = {"measured_utc": datetime.now(timezone.utc).isoformat(),
              "backend": "native --core; algebraic policy",
              "compiler_sha256": sha(args.compiler), "audit_sha256": sha(audit_path),
              "dialect_sha256": sha(dialect),
              "dispatch_sources": {p: sha(ROOT / p) for p in
                                   ["src/inference/emission.cc", "src/inference/insertion.cc",
                                    "src/core/attr_synth/ordered/extensions.cc"]},
              "populations": {"audited_axis_forms": len(audit["records"]),
                              "historically_inferred": sum(r["recorded_outcome"] == "inferred" for r in audit["records"]),
                              "main_theorem_scope": len(selected)},
              "counts": counts, "cases": cases}
    (COVERAGE / "results/core_classification.json").write_text(json.dumps(result, indent=2) + "\n")
    (COVERAGE / "reports/core_classification.md").write_text(report(result))
    print(json.dumps({"main_theorem_scope": len(cases), "native_paths": counts}))


if __name__ == "__main__":
    main()
