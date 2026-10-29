"""Check source compaction and unique responsibility accounting."""

import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('size_responsibility', HERE / 'scripts/responsibility.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ResponsibilityTest(unittest.TestCase):
    def inferred(self, name):
        return (r.LAB / 'tepl/rules/axis' / (name + '.tepl')).read_text()

    def test_axis_only_rule_removes_descriptors_and_unused_labels(self):
        source = self.inferred('transpose_compose')
        variants = r.compact_sources(source)
        self.assertIn('.attrs = {};', variants['implicit'])
        for word in ['derive', 'attr_outer', 'attr_inner', 'attr_combined', 'auto']:
            self.assertNotIn(word, variants['compact'])
        self.assertEqual(r.m.values(r.m.rule_text(variants['compact'])),
                         ['rule','transpose_compose','{','(','transpose','(','transpose','X',')',')','=>','(','transpose','X',')','}'])

    def test_mixed_fields_and_guard_bindings_stay(self):
        variants = r.compact_sources(self.inferred('broadcast_multiply'))
        for source in variants.values():
            self.assertNotIn('dimensions: auto', source)
            self.assertIn('new_extents: attr_left.attrs.new_extents', source)
            self.assertIn('attr_left.attrs.dimensions == attr_right.attrs.dimensions', source)
            self.assertIn('derive', source)

    def test_fixed_adapter_assignments_stay_explicit(self):
        variants = r.compact_sources(self.inferred('catalog_indexing_reverse_inserted_unit_dims'))
        self.assertIn('new_extents: [1]', variants['compact'])
        self.assertIn('.attrs.dimensions == [1]', variants['compact'])

    def test_tepl_categories_cover_all_tokens_and_inline_copies(self):
        source = (r.LAB/'tepl/rules/semantic/dot_associate.tepl').read_text()
        parts = r.tepl_parts(source)
        self.assertEqual(sum(parts.values()), r.m.sizes(r.m.rule_text(source))['tokens'])
        self.assertGreater(parts['attributes'], 0)
        self.assertGreater(parts['legality'], 0)

    def test_xla_overlap_counts_once_and_disagreement_fails(self):
        name='algebraic_simplifier.cc'
        path=r.LAB/'catalog/upstream'/name
        block=dict(id='a',file=name,spans=[[10248,10254,'construction']])
        manifest=dict(source_hashes={name:r.m.sha(path)},blocks=[block],cases=[
            dict(case='one',status='reviewed',blocks=['a'],note='test'),
            dict(case='two',status='reviewed',blocks=['a'],note='test')])
        a,_=r.xla_parts(manifest,{'one'})
        b,_=r.xla_parts(manifest,{'one','two'})
        self.assertEqual(a,b)
        manifest['blocks'].append(dict(id='b',file=name,spans=[[10248,10254,'matching']]))
        manifest['cases'][1]['blocks']=['b']
        with self.assertRaisesRegex(ValueError,'conflicting'):
            r.xla_parts(manifest,{'one','two'})


if __name__ == '__main__':
    unittest.main()
