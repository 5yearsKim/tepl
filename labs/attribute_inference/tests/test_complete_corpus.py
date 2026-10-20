"""Preserve the complete denominator and compare every prepared inference variant."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot
import run_corpus


class CompleteCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = run_corpus.build_manifest()

    def test_denominators_include_unproved_and_supplementary_cases(self):
        cases = self.manifest["cases"]
        primary = [c for c in cases if c["attribute_scope"] == "axis_extent"]
        self.assertEqual(len(cases), 42)
        self.assertEqual(len(primary), 38)
        self.assertEqual(sum(c["recorded_explicit_status"] == "not_proved" for c in primary), 18)
        self.assertEqual(json.loads(run_corpus.MANIFEST.read_text()), self.manifest)

    def test_every_prepared_variant_preserves_structure_and_conditions(self):
        for case in self.manifest["cases"]:
            source = (pilot.ROOT / case["source"]).read_text()
            specs = [{"infer": list(case["rhs_descriptors"]), "pins": {}}] + case["assisted"]
            for spec in specs:
                with self.subTest(case=case["name"], mode=spec.get("mode", "full_auto")):
                    text, requested = run_corpus.make_variant(source, case, spec)
                    self.assertEqual(pilot.invariant(source), pilot.invariant(text))
                    self.assertGreater(requested, 0)

    def test_assistance_retains_dependencies(self):
        source = """rule r { X => X derive {
            let p = gather(@a.permutation, @b.permutation);
            @first = { permutation: p };
            @second = { ..@first };
            @unused = { permutation: [] };
        } }"""
        retained = "\n".join(run_corpus.retained_derive(source, {"@second"}))
        self.assertIn("let p", retained)
        self.assertIn("@first =", retained)
        self.assertIn("@second =", retained)
        self.assertNotIn("@unused", retained)

    def test_literal_pins_follow_spreads_but_respect_overrides(self):
        original = """rule r { X => X where { @source.dimensions == [] }
            derive { @result = { ..@source, new_extents: [2, 3] }; } }"""
        self.assertEqual(run_corpus.literal_fields(original, "result"),
                         {"dimensions": "[]", "new_extents": "[2, 3]"})
        overridden = original.replace("new_extents: [2, 3]", "dimensions: @other.dimensions")
        self.assertEqual(run_corpus.literal_fields(overridden, "result"), {})

    def test_no_assisted_variant_supplies_every_field(self):
        for case in self.manifest["cases"]:
            for spec in case["assisted"]:
                remaining = sum(len(case["rhs_descriptors"][name]) - len(spec["pins"].get(name, {}))
                                for name in spec["infer"])
                self.assertGreater(remaining, 0)
                if spec["mode"].startswith("assisted_operation"):
                    self.assertLess(len(spec["infer"]), len(case["rhs_descriptors"]))

    def test_identity_witness_audit_distinguishes_composition(self):
        cases = {c["name"]: c for c in self.manifest["cases"]}
        copied = {"mode": "full_auto", "status": "proved", "constructors": [
            {"field": name, "expression": "@s." + name} for name in ("starts", "sizes", "strides")]}
        self.assertTrue(run_corpus.identity_witness(cases["catalog_indexing_slice_singleton_stride"], copied))
        composition = {"mode": "full_auto", "status": "proved", "constructors": [
            {"field": "permutation", "expression": "gather(@inner.permutation, @outer.permutation)"}]}
        self.assertFalse(run_corpus.identity_witness(cases["transpose_compose"], composition))

    def test_saved_evidence_covers_every_predeclared_variant(self):
        if not run_corpus.RESULT.exists():
            self.skipTest("run the complete cohort to validate saved evidence")
        data = json.loads(run_corpus.RESULT.read_text())
        rows = [r for r in data["results"] if not r["control"]]
        lookup = {(r["case"], r["mode"]): r for r in rows}
        self.assertEqual(len(lookup), len(rows))
        for case in self.manifest["cases"]:
            modes = {"explicit", "full_auto"} | {s["mode"] for s in case["assisted"]}
            self.assertEqual({m for n, m in lookup if n == case["name"]}, modes)
            for mode in modes:
                if lookup[case["name"], mode]["status"] == "not_needed":
                    self.assertTrue(mode.startswith("assisted"))
                    self.assertEqual(lookup[case["name"], "full_auto"]["status"], "proved")


if __name__ == "__main__":
    unittest.main()
