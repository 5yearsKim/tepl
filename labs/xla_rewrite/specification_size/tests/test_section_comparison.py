"""Check literal sections and consistent 30/31-case accounting."""

import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
spec = importlib.util.spec_from_file_location("section_comparison", HERE / "scripts/section_comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class SectionTest(unittest.TestCase):
    def test_literal_blocks_include_all_syntax_and_keep_constraints_separate(self):
        source = '''from "dialects/hlo.tepl" import HLO as hlo;
        rule r { X: f32[N]
          (a: hlo.reverse X) => (b: hlo.reverse[a.attrs] X)
          where { let n = len(X.shape); n > 0 }
          derive { let d = [0]; b.attrs = {dimensions: d}; }
        }'''
        parts = comparison.sections(source)
        count = lambda s: len(comparison.m.tokens(s))
        self.assertEqual(parts["matching"], count('(a: hlo.reverse X)'))
        self.assertEqual(parts["construction"], count('(b: hlo.reverse[a.attrs] X)'))
        self.assertEqual(parts["legality"], count('where { let n = len(X.shape); n > 0 }'))
        self.assertEqual(parts["attributes"], count('derive { let d = [0]; b.attrs = {dimensions: d}; }'))
        self.assertEqual(parts["other"], count('rule r { X: f32[N] => }'))
        self.assertEqual(sum(parts.values()), count(comparison.m.rule_text(source)))

    def test_missing_blocks_cost_zero(self):
        parts = comparison.sections('rule r { (hlo.reverse X) => X }')
        self.assertEqual(parts["legality"], 0)
        self.assertEqual(parts["attributes"], 0)

    def test_mapped_and_full_cohorts_differ_only_by_unmapped_dot_rule(self):
        data, _ = comparison.collect()
        excluded = [r for r in data["cases"] if r["case"] not in data["matched_cases"]]
        self.assertEqual([r["case"] for r in excluded], ['catalog_dot_scalar_multipliers'])
        self.assertEqual(len(data["matched_cases"]), 30)
        self.assertEqual(len(data["cases"]), 31)
        for variant in ["explicit", "inferred"]:
            for key in comparison.LABELS:
                self.assertEqual(data["all_tepl"][variant][key] - data["matched"][variant][key],
                                 excluded[0][variant]["parts"][key])
        self.assertEqual([sum(data['matched'][v].values()) for v in ['xla', 'explicit', 'inferred']],
                         [10022, 3205, 2374])
        self.assertEqual([sum(data['all_tepl'][v].values()) for v in ['explicit', 'inferred']],
                         [3396, 2520])
        self.assertEqual(data["all_tepl"]["inferred"]["legality"], 981)
        for row in data["cases"]:
            self.assertNotIn('auto', comparison.m.values(row['inferred']['source']))
            self.assertEqual(row['explicit']['parts']['legality'], row['inferred']['parts']['legality'])

    def test_axis_only_rule_omits_labels_fields_and_derive(self):
        original = (comparison.LAB / 'tepl/rules/semantic/dot_associate.tepl').read_text()
        canonical = (comparison.LAB / 'tepl/rules/axis/dot_associate.tepl').read_text()
        explicit, inferred, _ = comparison.representations(original, canonical, 'dot_associate')
        self.assertIn('lhs_contracting_dimensions: attr_outer.attrs.lhs_contracting_dimensions', explicit)
        self.assertIn('derive', explicit)
        for word in ['derive', 'auto', 'axis_eval_0', 'axis_eval_1']:
            self.assertNotIn(word, comparison.m.values(inferred))
        self.assertEqual(comparison.sections(inferred)['attributes'], 0)
        self.assertIn('attr_inner.attrs.lhs_contracting_dimensions', inferred)

    def test_non_axis_attributes_and_required_labels_remain(self):
        original = (comparison.LAB / 'tepl/rules/semantic/broadcast_add.tepl').read_text()
        canonical = (comparison.LAB / 'tepl/rules/axis/broadcast_add.tepl').read_text()
        explicit, inferred, _ = comparison.representations(original, canonical, 'broadcast_add')
        self.assertIn('new_extents: attr_left.attrs.new_extents', inferred)
        self.assertIn('derive', inferred)
        self.assertNotIn('dimensions: auto', inferred)
        self.assertEqual(comparison.sections(explicit)['legality'], comparison.sections(inferred)['legality'])


if __name__ == "__main__":
    unittest.main()
