"""Check counting, causal attribution, and paired-cohort accounting."""

import importlib.util
import copy
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("size_measure", HERE / "scripts/measure.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


class SizeTest(unittest.TestCase):
    def test_comments_strings_and_maximal_operators(self):
        source = 'x::y += "// not a comment"; /* ignored */ a && b != 0;'
        self.assertEqual(measure.values(source),
                         ['x', '::', 'y', '+=', '"// not a comment"', ';', 'a', '&&', 'b', '!=', '0', ';'])
        self.assertEqual(measure.values('R"tag(a // b\n/* c */)tag"'), ['R"tag(a // b\n/* c */)tag"'])

    def test_code_lines_ignore_blank_and_comment_lines(self):
        self.assertEqual(measure.sizes('// hi\n\nx /* x\n y */ + z;\n')["code_lines"], 2)
        self.assertEqual(measure.sizes('"a\nb"')["code_lines"], 2)

    def test_layout_ignores_input_formatting(self):
        a = 'rule r {(hlo.reverse X)=>(hlo.reverse X) derive { x.attrs={dimensions:auto}; }}'
        b = a.replace('{', '{\n').replace('}', '\n}').replace(' ', '  ')
        self.assertEqual(measure.sizes(a)["layout_lines"], measure.sizes(b)["layout_lines"])
        self.assertEqual(measure.sizes(a)["tokens"], measure.sizes(b)["tokens"])

    def test_matching_keeps_supplied_fields_and_accounts_exactly(self):
        source = (measure.LAB / 'tepl/rules/semantic/broadcast_multiply.tepl').read_text()
        pair = measure.matched_pair(source)
        before, after = pair['explicit_parts'], pair['inferred_parts']
        self.assertEqual(before['supplied_values'], after['supplied_values'])
        self.assertEqual(before['descriptor_syntax'], after['descriptor_syntax'])
        self.assertEqual(before['guard'], after['guard'])
        for v in ['explicit', 'inferred']:
            self.assertEqual(sum(pair[v+'_parts'].values()), measure.sizes(measure.rule_text(pair[v]))['tokens'])
        self.assertEqual(sum(before.values())-sum(after.values()), before['axis_values']-after['axis_values'])

    def test_only_needed_helpers_return_to_explicit_pair(self):
        source = '''rule helper_case {
          (s: hlo.broadcast X) => (r: hlo.broadcast X)
          derive { let unused = [99]; let a = s.attrs.dimensions;
                   let b = gather(a, range(len(a)));
                   r.attrs = {dimensions: b, new_extents: s.attrs.new_extents}; }
        }'''
        pair = measure.matched_pair(source)
        self.assertIn('let a', pair['explicit'])
        self.assertIn('let b', pair['explicit'])
        self.assertNotIn('let unused', pair['explicit'])
        self.assertNotIn('let a', pair['inferred'])
        self.assertGreater(pair['explicit_parts']['local_helpers'], 0)
        self.assertEqual(pair['inferred_parts'].get('local_helpers', 0), 0)

    def test_shared_supplied_helpers_are_kept(self):
        source = '''rule helper_case {
          (s: hlo.broadcast X) => (r: hlo.broadcast X)
          derive { let dims = s.attrs.dimensions; let ext = s.attrs.new_extents;
                   r.attrs = {dimensions: dims, new_extents: ext}; }
        }'''
        pair = measure.matched_pair(source)
        self.assertIn('let ext', pair['explicit'])
        self.assertIn('let ext', pair['inferred'])
        self.assertNotIn('let dims', pair['inferred'])

    def test_normalization_is_identical_in_pair(self):
        source = (measure.LAB / 'tepl/rules/semantic/catalog_indexing_reverse_inserted_unit_dims.tepl').read_text()
        pair = measure.matched_pair(source)
        self.assertEqual(len(pair['normalizations']), 2)
        self.assertEqual(pair['explicit_parts']['normalization_adapter'], pair['inferred_parts']['normalization_adapter'])
        self.assertEqual(pair['explicit_parts']['guard'], pair['inferred_parts']['guard'])

    def test_source_ranges_count_shared_lines_once(self):
        self.assertEqual(measure.union_ranges([[4, 8], [2, 5], [9, 10], [20, 21]]), [[2, 10], [20, 21]])
        with self.assertRaises(ValueError):
            measure.union_ranges([[0, 1]])

    def test_failure_evidence_cannot_enter_verified_cohort(self):
        result = dict(check=dict(returncode=0, stdout='status=proved\nadditional-guard: true\n'),
                      compile=dict(returncode=0), validation=dict(returncode=0),
                      artifact_sha256='a' * 64)
        self.assertTrue(measure.verified(result, 0))
        changed = copy.deepcopy(result)
        changed['validation']['returncode'] = 1
        self.assertFalse(measure.verified(changed, 0))
        changed = copy.deepcopy(result)
        changed['check']['stdout'] = 'status=proved\nadditional-guard: unresolved\n'
        self.assertFalse(measure.verified(changed, 0))
        self.assertFalse(measure.verified(result, 1))

    def test_operator_units_include_descriptor_spread(self):
        self.assertEqual(measure.values('{..p.attrs, x: []}'),
                         ['{', '..', 'p', '.', 'attrs', ',', 'x', ':', '[', ']', '}'])


if __name__ == '__main__':
    unittest.main()
