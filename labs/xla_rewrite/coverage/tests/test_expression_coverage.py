"""Keep language acceptance and semantic proof as separate measurements."""

import unittest

from _support import expression_coverage as coverage


class ExpressionCoverageTest(unittest.TestCase):
    def setUp(self):
        here = coverage.HERE
        self.inputs = [
            coverage.read(here.parent / "catalog/rules.json")["rules"],
            coverage.read(here / "data/language_assessments.json"),
            coverage.read(here / "data/compilation_cases.json"),
            coverage.read(here / "results/compilation.json"),
        ]

    def test_recorded_language_data_is_consistent(self):
        coverage.validate(*self.inputs)

    def test_language_assessment_must_include_entire_catalog(self):
        self.inputs[1]["rules"].pop()
        with self.assertRaisesRegex(ValueError, "each catalog schema once"):
            coverage.validate(*self.inputs)

    def test_extension_requirements_must_be_explicit(self):
        row = next(r for r in self.inputs[1]["rules"] if r["status"] == "needs_extension")
        row["integration"] = []
        with self.assertRaisesRegex(ValueError, "integration requirement"):
            coverage.validate(*self.inputs)

    def test_compilation_cannot_be_labeled_semantic_proof(self):
        self.inputs[3]["results"][0]["status"] = "proved"
        with self.assertRaisesRegex(ValueError, "must not be recorded as semantic proof"):
            coverage.validate(*self.inputs)

    def test_measured_case_must_reference_correct_catalog_schema(self):
        self.inputs[3]["results"][0]["catalog_id"] = "another-schema"
        with self.assertRaisesRegex(ValueError, "case ID mismatch"):
            coverage.validate(*self.inputs)

    def test_changed_expression_requires_new_measurement(self):
        self.inputs[3]["results"][0]["file_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "stale expression measurement"):
            coverage.validate(*self.inputs)

    def test_original_192_targets_all_have_a_compiled_baseline(self):
        baseline = self.inputs[2]["baseline"]
        self.assertEqual(len(baseline["target_ids"]), 192)
        self.assertEqual(set(baseline["target_ids"]), set(baseline["cases"]))
        compiled = {r["case"] for r in self.inputs[3]["results"]
                    if r["status"] == "compiled"}
        self.assertTrue(set(baseline["cases"].values()) <= compiled)

    def test_missing_designated_rule_is_rejected(self):
        self.inputs[2]["baseline"]["cases"].pop(
            self.inputs[2]["baseline"]["target_ids"][0])
        with self.assertRaisesRegex(ValueError, "every target schema"):
            coverage.validate(*self.inputs)

    def test_checked_ir_alone_is_not_an_executable_compilation(self):
        self.inputs[3]["results"][0]["validated_json"] = False
        with self.assertRaisesRegex(ValueError, "validated executable JSON"):
            coverage.validate(*self.inputs)


if __name__ == "__main__":
    unittest.main()
