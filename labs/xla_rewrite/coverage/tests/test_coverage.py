"""Integrity checks that prevent inflated or stale coverage reports."""

import copy
import unittest

from _support import summarize_coverage as coverage


class CoverageIntegrityTest(unittest.TestCase):
    def setUp(self):
        here = coverage.HERE
        self.inputs = [
            coverage.read(here.parent / "catalog/rules.json")["rules"],
            coverage.read(here / "data/assessments.json"),
            coverage.read(here / "data/verification_cases.json"),
            coverage.read(here / "results/verification.json"),
        ]

    def test_recorded_data_is_consistent(self):
        coverage.validate(*self.inputs)

    def test_reasoning_preserves_frozen_corpus_and_previous_proofs(self):
        _, previous, added = coverage.reasoning_comparison(self.inputs[3])
        self.assertEqual(len(previous), 80)
        self.assertTrue(added)

    def test_lost_baseline_proof_is_rejected(self):
        baseline = coverage.read(coverage.HERE / "baselines/reasoning_baseline.json")
        name = next(r["case"] for r in baseline["results"] if r["status"] == "proved")
        result = next(r for r in self.inputs[3]["results"] if r["case"] == name)
        result["status"] = "not_proved"
        with self.assertRaisesRegex(ValueError, "previously verified cases regressed"):
            coverage.reasoning_comparison(self.inputs[3])

    def test_missing_and_duplicate_catalog_assessments_are_rejected(self):
        for duplicate in (False, True):
            inputs = copy.deepcopy(self.inputs)
            if duplicate:
                inputs[1]["rules"].append(inputs[1]["rules"][0])
            else:
                inputs[1]["rules"].pop()
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                coverage.validate(*inputs)

    def test_duplicate_measurements_are_rejected(self):
        self.inputs[3]["results"].append(self.inputs[3]["results"][0])
        with self.assertRaisesRegex(ValueError, "results must cover each case once"):
            coverage.validate(*self.inputs)

    def test_stale_case_measurement_is_rejected(self):
        self.inputs[3]["results"][0]["file_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "stale result"):
            coverage.validate(*self.inputs)

    def test_stale_shared_dependency_is_rejected(self):
        dependency = next(iter(self.inputs[3]["dependencies"]))
        self.inputs[3]["dependencies"][dependency] = "old"
        with self.assertRaisesRegex(ValueError, "stale measured dependency"):
            coverage.validate(*self.inputs)

    def test_failed_check_cannot_be_reported_as_proved(self):
        result = self.inputs[3]["results"][0]
        result.update(status="proved", returncode=1)
        with self.assertRaisesRegex(ValueError, "proved result must exit successfully"):
            coverage.validate(*self.inputs)

    def test_accepted_negative_control_is_rejected(self):
        control = next(c for c in self.inputs[2]["cases"] if c["control"])
        result = next(r for r in self.inputs[3]["results"]
                      if r["case"] == control["name"])
        result.update(status="proved", returncode=0)
        with self.assertRaisesRegex(ValueError, "negative control must fail"):
            coverage.validate(*self.inputs)

    def test_restricted_assessment_requires_a_declared_restriction(self):
        row = next(r for r in self.inputs[1]["rules"]
                   if r["assessment"] == "restricted")
        row["restriction"] = ""
        with self.assertRaisesRegex(ValueError, "needs a restriction"):
            coverage.validate(*self.inputs)

    def test_structural_acceptance_cannot_be_reported_as_a_proof(self):
        result = next(r for r in self.inputs[3]["results"]
                      if r["status"] == "missing_semantics")
        result.update(status="proved", returncode=0, proof_evidence=[])
        with self.assertRaisesRegex(ValueError, "semantic proof evidence"):
            coverage.validate(*self.inputs)

    def test_missing_semantics_must_name_the_operations(self):
        result = next(r for r in self.inputs[3]["results"]
                      if r["status"] == "missing_semantics")
        result["missing_semantics"] = []
        with self.assertRaisesRegex(ValueError, "operation names"):
            coverage.validate(*self.inputs)

    def test_each_compilation_case_has_a_verification_counterpart(self):
        cases = [c for c in self.inputs[2]["cases"] if not c["control"]]
        expected = coverage.read(coverage.HERE / "data/compilation_cases.json")["cases"]
        self.assertEqual({c["expression_case"] for c in cases},
                         {c["name"] for c in expected})
        self.assertEqual(len(cases), len(expected))


if __name__ == "__main__":
    unittest.main()
