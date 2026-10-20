"""Protect benchmark comparability and proof-evidence parsing."""

import importlib.util
import json
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("pilot", Path(__file__).resolve().parents[1] / "scripts/pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class PilotIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(pilot.MANIFEST.read_text())

    def test_all_variants_preserve_conditions_and_structure(self):
        for case in self.manifest["cases"]:
            source = (pilot.ROOT / case["source"]).read_text()
            specs = [{"infer": list(case["rhs_descriptors"])}] + case["assisted"]
            for spec in specs:
                with self.subTest(case=case["name"], mode=spec.get("mode", "full_auto")):
                    variant, requested = pilot.make_variant(source, case, spec)
                    self.assertEqual(pilot.invariant(source), pilot.invariant(variant))
                    self.assertGreater(requested, 0)
                    self.assertEqual(pilot.code_metrics(source)["guard_tokens"],
                                     pilot.code_metrics(variant)["guard_tokens"])

    def test_condition_changes_are_detected(self):
        case = next(c for c in self.manifest["cases"] if c["name"] == "scalar_broadcast_transpose")
        source = (pilot.ROOT / case["source"]).read_text()
        self.assertNotEqual(pilot.invariant(source), pilot.invariant(source.replace("== 2", "== 3")))

    def test_fully_pinned_descriptors_are_not_inference(self):
        case = self.manifest["cases"][0]
        source = (pilot.ROOT / case["source"]).read_text()
        with self.assertRaisesRegex(ValueError, "supplies every target"):
            pilot.make_variant(source, case, {"infer": ["combined"],
                                             "pins": {"combined": {"permutation": "[]"}}})

    def test_constructor_parser_excludes_coordinate_frame(self):
        output = """      constructor auto#2:
        permutation = gather(@inner.permutation, @outer.permutation)
      declared-conditions: true
      coordinate frame:
        input0 = group(extents=shape(Input(#0)))
"""
        self.assertEqual(pilot.constructor_fields(output),
                         [("permutation", "gather(@inner.permutation, @outer.permutation)")])

    def test_control_has_a_concrete_order_counterexample(self):
        inner, outer = [1, 0, 2], [0, 2, 1]
        correct = [inner[i] for i in outer]
        corrupted = [outer[i] for i in inner]
        self.assertEqual(correct, [1, 2, 0])
        self.assertEqual(corrupted, [2, 0, 1])
        self.assertNotEqual(correct, corrupted)


if __name__ == "__main__":
    unittest.main()
