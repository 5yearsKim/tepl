"""Evidence validation and pinned-source preparation regression checks."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import branch_hooks
import prepare_xla
import run_workloads

HERE = Path(__file__).resolve().parent


class ProtocolTest(unittest.TestCase):
    def test_port_inventory_and_source_hooks(self):
        registry = json.loads((HERE.parent / "xla_workload_count/support-registry.json").read_text())
        inventory = json.loads((HERE / "ports.json").read_text())
        expected = {int(name[3:]) for name in registry["rules"]}
        actual = {row["site"] for row in inventory["ports"]}
        self.assertEqual(len(expected), 38)
        self.assertEqual(expected, actual)
        self.assertEqual(expected, set(branch_hooks.PORTS) | {2744, 10114, 10118})
        base = prepare_xla.instrumented_base()
        self.assertEqual(hashlib.sha256(base.encode()).hexdigest(), registry["instrumented_source_sha256"])
        patched = branch_hooks.apply(prepare_xla.patch_source(base))
        self.assertEqual(branch_hooks.apply(patched), patched)
        for site in expected:
            self.assertIn(f"ObserveAlgebraicRewrite({site},", patched)
        # Preserve native commit semantics for in-place mutation and ownership.
        for text in ("/*relay_control_dependency=*/true", "ReplaceInstructionIfCompatible",
                     "/*preserve_frontend_attributes=*/false", "ReplaceOperandWith"):
            self.assertIn(text, patched)
        rules = (HERE / "branch_rules.tepl").read_text()
        for row in inventory["ports"]:
            if row["deployment"] == "explicit_structural":
                for name in row["rules"]:
                    self.assertRegex(rules, r"\brule\s+" + re.escape(name) + r"\b")

    def test_evidence_rejects_unaccounted_fallback(self):
        result = dict(final_graphs_equal=True, outputs=[dict(equal=True)] * 3,
                      rewrite_counts_equal=True, baseline_pass_invocations=3, integrated_pass_invocations=3,
                      rules=[dict(original_line=934, applications=4, tepl_applications=4)],
                      distinct_variants=1, total_applications=4, tepl_applications=4,
                      all_observed_variants_delegated=True, unported_sites=[])
        run_workloads.check_result(result, strict=True)
        bad = copy.deepcopy(result)
        bad["rules"][0]["tepl_applications"] = 3
        with self.assertRaisesRegex(ValueError, "TEPL application count"):
            run_workloads.check_result(bad, strict=True)
        bad["tepl_applications"] = 3
        with self.assertRaisesRegex(ValueError, "closure evidence"):
            run_workloads.check_result(bad, strict=True)
        bad.update(all_observed_variants_delegated=False, unported_sites=[934])
        run_workloads.check_result(bad, strict=False)
        with self.assertRaisesRegex(ValueError, "strict integration incomplete"):
            run_workloads.check_result(bad, strict=True)

    def test_graph_archive_preserves_full_constants(self):
        content = b"f32[3] constant({1.234567, 2.345678, 3.456789})\n" * 1000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.final.hlo"
            path.write_bytes(content)
            self.assertEqual(run_workloads.archive_graph(path), hashlib.sha256(content).hexdigest())
            self.assertFalse(path.exists())
            with gzip.open(path.with_suffix(".hlo.gz"), "rb") as archive:
                self.assertEqual(archive.read(), content)


if __name__ == "__main__":
    unittest.main()
