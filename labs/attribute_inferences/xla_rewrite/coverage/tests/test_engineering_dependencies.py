"""Engineering provenance is independent of theory prose and paper exports."""

import copy
import json
import unittest

from _support import HERE
from scripts import run_synthesis, synthesis_report
from scripts.paths import ROOT


class EngineeringDependenciesTest(unittest.TestCase):
    def test_research_hashes_do_not_replace_implementation_integrity_checks(self):
        source = HERE / "scripts/run_synthesis.py"
        key = str(source.relative_to(ROOT))
        classification = dict(
            protocol="axis_only_v1",
            dependencies={
                key: run_synthesis.sha(source),
                "research/unavailable-proof.md": "historical hash",
            },
            cases=[],
        )
        data = dict(protocol="axis_only_v1", results=[],
                    dependencies=classification["dependencies"])
        run_synthesis.validate_classification(classification)
        synthesis_report.validate(data, classification)
        changed = copy.deepcopy(classification)
        changed["dependencies"][key] = "changed implementation"
        for validate in (
            lambda: run_synthesis.validate_classification(changed),
            lambda: synthesis_report.validate(data, changed),
        ):
            with self.assertRaisesRegex(ValueError, "stale classification dependency"):
                validate()

    def test_new_measurements_only_fingerprint_implementation_inputs(self):
        dependencies = run_synthesis.dependencies()
        self.assertIn(str((ROOT / "src/core/pipeline.cc").relative_to(ROOT)), dependencies)
        self.assertFalse(any(path.startswith("research/") for path in dependencies))

    def test_reports_do_not_require_theory_metadata(self):
        data = json.loads((HERE / "results/synthesis.json").read_text())
        for row in data["results"]:
            for key in tuple(row):
                if key.startswith("theorem_"):
                    del row[key]
        report = synthesis_report.render(data, {})
        table = synthesis_report.case_table(data)
        self.assertIn("Unresolved axis cases", report)
        self.assertIn("Axis fields", table)
        self.assertNotIn("Theorem", report + table)
        self.assertNotIn("research/", report + table)
