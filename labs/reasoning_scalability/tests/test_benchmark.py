"""Scope and integrity checks for the bounded in-process experiment."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark as p
import workloads as b


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.prepare()

    def test_only_required_bounded_workloads(self):
        manifest = json.loads(p.MANIFEST.read_text())
        self.assertEqual(manifest['sizes'], list(range(1, 7)))
        self.assertEqual(len(manifest['cases']), 46)
        self.assertEqual({c['family'] for c in manifest['cases']}, {'chain', 'rank'})
        self.assertTrue(all(1 <= c['size'] <= 6 for c in manifest['cases']))
        self.assertFalse(any(c['mode'] == 'partial' and c['size'] == 1 for c in manifest['cases']))
        for c in manifest['cases']:
            source = (b.ROOT / c['file']).read_text()
            self.assertNotIn('@', source)
            self.assertNotIn('[auto]', source)
            self.assertIn('rule ' + c['case_id'] + ' ', source)
            if c['family'] == 'rank':
                self.assertEqual(source.count(': transpose '), 3 + int(c['mode'] == 'explicit'))

    def test_partial_attrs_are_fixed_without_extra_conditions(self):
        for size in range(2, 7):
            _, explicit = b.dot_source(size, 'permuted', 'explicit')
            _, partial = b.dot_source(size, 'permuted', 'partial')
            self.assertEqual(explicit.split('where {')[1].split('}')[0],
                             partial.split('where {')[1].split('}')[0])
            rhs = partial.split('=>')[1].split('where')[0]
            self.assertEqual(rhs.count('(dot '), (size + 1) // 2)

    def test_exact_reference_oracles(self):
        self.assertEqual(p.validate_workloads(), 72)

    def test_saved_samples_when_present(self):
        if not p.RESULT.exists():
            self.skipTest('measurement has not been run')
        data = json.loads(p.RESULT.read_text())
        p.validate(data)
        self.assertEqual(len(p.summarize(data)), 46)
        for r in data['records']:
            self.assertLessEqual(r['construction_ms'] + r['verification_ms'], r['total_ms'] + 1e-6)


if __name__ == '__main__':
    unittest.main()
