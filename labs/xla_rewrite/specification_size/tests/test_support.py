"""Exercise scanner, source preparation, XLA deduplication, and proof admission."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import support as s


class SupportTest(unittest.TestCase):
    def test_scanner_comments_strings_and_maximal_operators(self):
        self.assertEqual(s.values('x::y += "// literal"; /* ignored */ a && b != 0;'),
                         ['x', '::', 'y', '+=', '"// literal"', ';', 'a', '&&', 'b', '!=', '0', ';'])
        self.assertEqual(s.values('R"tag(a // b\n/* c */)tag"'), ['R"tag(a // b\n/* c */)tag"'])
        self.assertEqual(s.values('{..p.attrs}'), ['{', '..', 'p', '.', 'attrs', '}'])

    def test_explicit_restores_only_needed_helper_dependencies(self):
        source = '''rule helper_case {
          (s: hlo.broadcast X) => (r: hlo.broadcast X)
          derive { let unused = [99]; let a = s.attrs.dimensions;
                   let b = gather(a, range(len(a)));
                   r.attrs = {dimensions: b, new_extents: s.attrs.new_extents}; }
        }'''
        pair = s.matched_pair(source)
        self.assertIn('let a', pair['explicit'])
        self.assertIn('let b', pair['explicit'])
        self.assertNotIn('let unused', pair['explicit'])
        self.assertNotIn('let a', pair['inferred'])
        self.assertEqual(len(pair['fields']), 1)

    def test_supplied_helpers_and_fixed_adapter_assignments_are_retained(self):
        source = '''rule helper_case {
          (s: hlo.broadcast X) => (r: hlo.broadcast X)
          derive { let dims = s.attrs.dimensions; let ext = s.attrs.new_extents;
                   r.attrs = {dimensions: dims, new_extents: ext}; }
        }'''
        pair = s.matched_pair(source)
        compact = s.compact_source(pair['inferred'])
        self.assertIn('let ext', compact)
        self.assertNotIn('let dims', compact)
        source = (s.LAB / 'tepl/rules/axis/catalog_indexing_reverse_inserted_unit_dims.tepl').read_text()
        compact = s.compact_source(source)
        self.assertIn('new_extents: [1]', compact)
        self.assertIn('.attrs.dimensions == [1]', compact)

    def test_all_selected_tasks_match_recorded_sources_and_proofs(self):
        _, rows = s.load_successful_tasks()
        self.assertEqual(len(rows), 31)
        row = copy.deepcopy(rows[0])
        for field in ['check', 'compile', 'validation']:
            changed = copy.deepcopy(row)
            changed[field]['returncode'] = 1
            self.assertFalse(s.accepted_task(changed))
        row['check']['stdout'] = 'status=proved\nadditional-guard: unresolved\n'
        self.assertFalse(s.accepted_task(row))

    def test_xla_shared_source_is_counted_once_and_conflicts_fail(self):
        filename = 'algebraic_simplifier.cc'
        block = dict(id='a', file=filename, spans=[[10248, 10254, 'construction']])
        manifest = dict(source_hashes={filename: s.sha(s.LAB / 'catalog/upstream' / filename)},
                        blocks=[block], cases=[dict(case=n, status='reviewed', blocks=['a'], note='test')
                                               for n in ['one', 'two']])
        single, _ = s.xla_parts(manifest, {'one'})
        shared, _ = s.xla_parts(manifest, {'one', 'two'})
        self.assertEqual(single, shared)
        manifest['blocks'].append(dict(id='b', file=filename, spans=[[10248, 10254, 'matching']]))
        manifest['cases'][1]['blocks'] = ['b']
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            s.xla_parts(manifest, {'one', 'two'})


if __name__ == '__main__':
    unittest.main()
