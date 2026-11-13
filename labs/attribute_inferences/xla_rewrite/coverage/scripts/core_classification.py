#!/usr/bin/env python3
"""Refresh theorem-scope counts and measure the actual native core dispatch.

Writes a separate current report; never edits historical default-backend runs
or the paper. The artifact's marker and instruction kinds follow emit()'s
restricted-insertion / specialized-schema / signed-constructor dispatch.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
import tempfile

from restricted_insertion import ROOT, migrate, run, sha

COVERAGE = Path(__file__).resolve().parents[1]
MAIN = {"main_theorem_form", "main_theorem_reduction"}
UNSIGNED = {"input", "permutation", "summation", "contraction", "addition"}


def report(result):
    counts = result["main_theorem_counts"]
    generic = counts.get("constructive_core", 0)
    normalized = counts.get("restricted_insertion", 0)
    rows = [
        "# Current Constructive Core Coverage", "",
        f"The paper audit classifies 21 forms within the main theorem. In this "
        f"measurement, **{generic} use generic construction** and **{normalized} "
        "use normalization followed by generic construction**. Compilation and "
        "portable validation pass for all 21. The three direct reversal forms "
        "now use the generic orientation predicate and reversal-axis builder.", "",
        "## Populations and interpretation", "",
        "All 38 audited axis forms are measured with the same compiler binary "
        "and algebraic policy using the native `--core` backend. The table "
        "records current core results separately from historical inference. "
        "Unresolved means this backend did not accept the input; it does not "
        "invalidate a historical success using another implementation path.", "",
        "| Current path | All audited forms | Main-theorem forms |",
        "|---|---:|---:|",
    ]
    for path, label in LABELS.items():
        rows.append(f"| {label} | {result['counts'].get(path, 0)} | {counts.get(path, 0)} |")
    rows += ["", "The historical success population is 31/38. The mathematical "
             "theorem classification is 21/38, or 21/31 historically successful "
             "forms. Current constructive coverage of that theorem population "
             "is 21/21; these are updated artifact measurements, not a "
             "reclassification of historical dispatch.", "",
             "## Case-level native dispatch", "",
             "| Case | Paper classification | Historical outcome | Current path | Compile / portable validation |",
             "|---|---|---|---|---|"]
    for case in result["cases"]:
        rows.append(f"| {case['case']} | {case['classification']} | "
                    f"{case['historical_outcome']} | {LABELS[case['path']]} | "
                    f"{'Passed' if case['portable_validated'] else 'Unresolved'} |")
    rows += ["", "## Original and companion inputs", "",
             "The 38-form audit, the paper's 31-success population, and the "
             "21-form theorem population use the recorded broadcast–transpose "
             "companion for `catalog_broadcast_transpose_broadcast`. Its "
             "supplied shape expression differs from the original frozen input. "
             "Each input's guards and supplied expressions are preserved during "
             "syntax migration. Acceptance of the companion is not acceptance "
             "of the frozen input. The latter is measured separately below "
             "and is excluded from the 38-form denominator.", ""]
    for c in result["companion_controls"]:
        rows.append(f"- Original frozen `{c['case']}`: "
                    f"{'passed' if c['portable_validated'] else 'unresolved'} "
                    f"under the same native core compiler; source `{c['source']}`.")
    rows += ["", "## Evidence and limits", "",
             "[Machine-readable results](../results/core_classification.json) "
             "record compiler and source hashes, Git revision, implementation "
             "hashes, per-case diagnostics, instruction kinds and DAG operators. "
             "The source hashes identify working-tree changes beyond the Git "
             "revision. Historical runs and paper files are preserved.", "",
             "`//tests:insertion_test` independently compares dense tensor "
             "results for the eight insertion reductions and pure signed core "
             "programs, including reversal parity, contraction transfer, "
             "eliminated axes and impossible orientations. Tests are bounded; "
             "compilation does not establish full-domain completeness.", "",
             "Completeness remains conditional on exact-real semantics, input "
             "extents at least two, positive supplied fresh extents, independent "
             "unknowns, and the restricted insertion boundary. Native `--core` "
             "measurements do not establish generated C++/Rust adapter coverage.", "",
             "```sh", "bazel build //:tepl",
             "python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/core_classification.py --paper /path/to/TEPL-paper",
             "bazel test //tests:insertion_test //tests:ordered_synthesis_test --test_output=errors",
             "```", "", f"Measured UTC: `{result['measured_utc']}`.", ""]
    return "\n".join(rows)


LABELS = {"constructive_core": "Generic construction",
          "restricted_insertion": "Normalization then generic construction",
          "legacy_schema": "Specialized schema", "unresolved": "Unresolved"}

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
    # Import the dialect's full vocabulary so unresolved cases fail on their
    # semantics rather than on missing operation imports.
    import re
    vocabulary = re.findall(r"^  op (\w+)", dialect.read_text(), re.M)
    vocabulary += re.findall(r"alias: (\w+)", dialect.read_text())
    views = dialect.with_name("axis_views.tepl")
    generic_kinds = UNSIGNED | {"reverse"}

    def measure(item, original, source_name, temp):
        source = Path(temp) / (item["case"] + ".tepl")
        source.write_text(f'from "{views}" import AxisViews as axis_view\n'
                          f'from "{dialect}" import HLO as hlo\n'
                          'use hlo::{' + ','.join(vocabulary) + '}\n' + migrate(original.read_text()))
        output = source.with_suffix(".json")
        case = {"case": item["case"], "classification": item["classification"],
                "historical_outcome": item.get("recorded_outcome"),
                "source": source_name, "source_sha256": sha(original),
                "migrated_source_sha256": sha(source), "path": "unresolved",
                "compiled": False, "portable_validated": False}
        try:
            run([compiler, "compile", source, "--core", "--semantic-policy", "algebraic", "-o", output])
            case["compiled"] = True
            run([compiler, "validate", output])
            case["portable_validated"] = True
        except RuntimeError as error:
            case["diagnostic"] = str(error).replace(str(source.parent), "<temporary-inputs>")
            return case
        core = json.loads(output.read_text())["program"]["rules"][0]["core"]
        program = core["program"]
        kinds = {n["kind"] for side in ("source", "replacement") for n in program[side]}
        path = ("restricted_insertion" if program.get("restricted_insertion") else
                "constructive_core" if kinds <= generic_kinds else "legacy_schema")
        case.update(path=path, instruction_kinds=sorted(kinds),
                    metadata_blocks=len(core["nodes"]),
                    dag_operators=sorted({n["op"] for block in core["nodes"] for n in block}))
        if "reverse" in kinds and path == "constructive_core":
            assert "orientation_check" in case["dag_operators"], case
        return case

    cases, controls = [], []
    with tempfile.TemporaryDirectory(prefix="tepl-core-coverage-") as temp:
        # Pin the binary: another build in the shared checkout may replace it.
        compiler = Path(temp) / "tepl"
        shutil.copyfile(args.compiler, compiler)
        compiler.chmod(0o755)
        compiler_hash = sha(compiler)
        for item in audit["records"]:
            name = item.get("source", f"evaluation/unified-construction/inputs/{item['case']}.tepl")
            cases.append(measure(item, args.paper / name, name, temp))
        # Separate original frozen control. This is not an extra audited form.
        frozen = ROOT / "labs/attribute_inferences/xla_rewrite/tepl/rules/axis/catalog_broadcast_transpose_broadcast.tepl"
        controls.append(measure({"case": "catalog_broadcast_transpose_broadcast_original",
                                 "classification": "frozen_original_control"},
                                frozen, str(frozen.relative_to(ROOT)), temp))
        assert sha(compiler) == compiler_hash, "compiler changed during measurement"
    counts = dict(Counter(c["path"] for c in cases))
    main_counts = dict(Counter(c["path"] for c in cases if c["classification"] in MAIN))
    assert main_counts == {"constructive_core": 13, "restricted_insertion": 8}, main_counts
    sources = ["src/inference/emission.cc", "src/inference/extensions.cc",
               "src/inference/extensions.h", "src/inference/insertion.cc",
               "src/inference/validate.cc", "src/executable/codec.cc",
               "src/core/attr_synth/ordered/contract.cc",
               "src/core/semantics/prove_rule.cc", "src/core/attr_synth/ordered/extensions.cc",
               "labs/attribute_inferences/xla_rewrite/coverage/scripts/core_classification.py"]
    result = {"measured_utc": datetime.now(timezone.utc).isoformat(),
              "backend": "native --core; algebraic policy",
              "git_revision": run(["git", "rev-parse", "HEAD"]).strip(),
              "compiler_sha256": compiler_hash, "audit_sha256": sha(audit_path),
              "dialect_sha256": sha(dialect), "axis_views_sha256": sha(views),
              "dispatch_sources": {p: sha(ROOT / p) for p in sources},
              "populations": {"audited_axis_forms": len(audit["records"]),
                              "historically_inferred": sum(r["recorded_outcome"] == "inferred" for r in audit["records"]),
                              "main_theorem_scope": len(selected)},
              "counts": counts, "main_theorem_counts": main_counts,
              "cases": cases, "companion_controls": controls}
    (COVERAGE / "results/core_classification.json").write_text(json.dumps(result, indent=2) + "\n")
    (COVERAGE / "reports/core_classification.md").write_text(report(result))
    print(json.dumps({"main_theorem_scope": len(selected), "native_paths": counts}))


if __name__ == "__main__":
    main()
