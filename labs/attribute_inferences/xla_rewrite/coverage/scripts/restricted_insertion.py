#!/usr/bin/env python3
"""Validate the paper's eight insertion reductions using the native core DAG.

This is a new engineering regression, not a replacement for frozen measurements.
The paper checkout is read only. No mathematical full-domain claim is inferred
from compilation or from the bounded execution tests in //tests:insertion_test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from paper_inputs import audit_path as resolve_audit, source_path as resolve_source

ROOT = Path(__file__).resolve().parents[5]
FIXTURES = ROOT / "tests/fixtures/restricted_insertion"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True,
                            timeout=60, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"{command}: {result.stderr or result.stdout}")
    return result.stdout


def migrate(raw):
    # Old paper syntax only; retain every guard and supplied expression.
    text = raw[raw.index("rule "):].replace(";", "")
    return re.sub(r"where\s*\{([^{}]*)\}", r"where \1", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, help="Optional external paper checkout; defaults to bundled inputs")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit_path = resolve_audit(args.paper)
    audit = json.loads(audit_path.read_text())
    records = {r["case"]: r for r in audit["records"]
               if r["classification"] == "main_theorem_reduction"}
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert len(records) == 8 and set(records) == {r["case"] for r in provenance}
    results = []
    dialect = ROOT / "labs/attribute_inferences/xla_rewrite/tepl/dialects/hlo.tepl"
    with tempfile.TemporaryDirectory(prefix="tepl-insertion-") as temp:
        for item in provenance:
            original = resolve_source(records[item["case"]]["source"], args.paper)
            fixture = FIXTURES / item["fixture"]
            assert records[item["case"]]["source"] == item["paper_source"]
            assert sha(original) == item["paper_sha256"], original
            assert sha(fixture) == item["fixture_sha256"], fixture
            migrated = migrate(original.read_text())
            expected = migrated.replace("hlo.", "TensorLang.")
            actual = fixture.read_text()
            assert actual[actual.index("rule "):] == expected, fixture
            source = Path(temp) / (item["case"] + ".tepl")
            # Preserve the HLO qualifier and aliases from the paper's imports.
            source.write_text(f'from "{dialect}" import HLO as hlo\n'
                              'use hlo::{broadcast,transpose,reverse,reduce_sum,dot}\n'
                              + migrated)
            output = source.with_suffix(".json")
            run([args.compiler, "compile", source, "--core", "--semantic-policy",
                 "algebraic", "-o", output])
            run([args.compiler, "validate", output])
            core = json.loads(output.read_text())["program"]["rules"][0]["core"]
            assert core["program"]["restricted_insertion"] is True
            assert len(core["nodes"]) == 2
            results.append({**item, "compiled": True, "portable_validated": True,
                            "path": "generic restricted insertion + constructive core",
                            "metadata_blocks": 2})
    result = {
        "scope": "eight audited restricted reductions; no unrestricted broadcast completeness claim",
        "compiler_sha256": sha(args.compiler),
        "audit_sha256": sha(audit_path), "dialect_sha256": sha(dialect),
        "cases": results,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
