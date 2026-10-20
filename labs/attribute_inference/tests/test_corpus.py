"""Check selection denominators and preserve comparisons before executing TEPL."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import classify_corpus
import pilot


class CorpusIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = classify_corpus.classify()

    def test_counts_distinguish_controls_and_attribute_work(self):
        rows = [r for r in self.data["cases"] if not r["control"]]
        self.assertEqual(len(rows), 209)
        attrs = [r for r in rows if r["rhs_descriptors"]]
        self.assertEqual(len(attrs), 111)
        supported = [r for r in attrs if not r["missing_semantics"]]
        self.assertEqual(len(supported), 42)
        eligible = [r for r in supported if r["attribute_scope"] == "axis_extent"]
        self.assertEqual(len(eligible), 38)
        self.assertEqual(sum(r["recorded_explicit_status"] == "proved" for r in eligible), 20)
        self.assertTrue(all(r["recorded_source_matches"] for r in rows))

    def test_batch_is_disjoint_and_includes_unproved_baselines(self):
        batch = classify_corpus.select_batch(self.data)
        previous = {c["name"] for c in json.loads(pilot.MANIFEST.read_text())["cases"]}
        self.assertEqual(len(batch["cases"]), 5)
        self.assertFalse(previous & {c["name"] for c in batch["cases"]})
        self.assertEqual(sum(c["recorded_explicit_status"] == "not_proved" for c in batch["cases"]), 3)

    def test_batch_variants_preserve_conditions_and_operations(self):
        for case in classify_corpus.select_batch(self.data)["cases"]:
            source = (pilot.ROOT / case["source"]).read_text()
            specs = [{"infer": list(case["rhs_descriptors"])}] + case["assisted"]
            for spec in specs:
                with self.subTest(case=case["name"], spec=spec):
                    variant, count = pilot.make_variant(source, case, spec)
                    self.assertEqual(pilot.invariant(source), pilot.invariant(variant))
                    self.assertGreater(count, 0)

    def test_ordinary_check_output_does_not_count_as_proof(self):
        job = {"file": "unused.tepl", "policy": "algebraic", "expected_constructor_fields": []}
        process = SimpleNamespace(returncode=0, stdout="CheckedProgram\n", stderr="")
        with patch.object(pilot.subprocess, "run", return_value=process):
            result = pilot.run_job(Path("/unused/tepl"), job, 1)
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
