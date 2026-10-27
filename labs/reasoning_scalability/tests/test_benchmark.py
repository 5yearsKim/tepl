"""Scope and integrity checks for the bounded in-process experiment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark as p
import workloads as b


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=b.HERE)
        cls.addClassCleanup(cls.directory.cleanup)
        output = Path(cls.directory.name)
        cls.paths = patch.multiple(p, OUTPUT=output, DIRECTORY=output / 'inputs',
                                   MANIFEST=output / 'inputs/manifest.json')
        cls.paths.start()
        cls.addClassCleanup(cls.paths.stop)
        cls.manifest = p.prepare()

    def test_only_required_bounded_workloads(self):
        manifest = json.loads(p.MANIFEST.read_text())
        self.assertEqual(manifest['chain_sizes'], list(range(1, 11)))
        self.assertEqual(manifest['ranks'], [2, 4, 8, 16, 32])
        self.assertEqual(len(manifest['cases']), 39)
        self.assertEqual({c['family'] for c in manifest['cases']}, {'chain', 'rank'})
        self.assertTrue(all(1 <= c['size'] <= 10 for c in manifest['cases'] if c['family'] == 'chain'))
        self.assertTrue(all('layout' not in c for c in manifest['cases']))
        self.assertFalse(any(c['mode'] == 'partial' and c['size'] == 1 for c in manifest['cases']))
        for c in manifest['cases']:
            source = (b.ROOT / c['file']).read_text()
            self.assertNotIn('@', source)
            self.assertNotIn('[auto]', source)
            self.assertIn('rule ' + c['case_id'] + ' ', source)
            if c['family'] == 'rank':
                self.assertEqual(source.count(': transpose '), 3 + int(c['mode'] == 'explicit'))
            else:
                self.assertEqual(source.count('dot '), 2 * c['size'])
                for node in b.chain(c['size'] + 1)[0]:
                    self.assertEqual(node.axes[0], 'b')

    def test_partial_attrs_are_fixed_without_extra_conditions(self):
        for size in range(2, 11):
            _, explicit = b.dot_source(size, 'explicit')
            _, partial = b.dot_source(size, 'partial')
            self.assertEqual(explicit.split('where {')[1].split('}')[0],
                             partial.split('where {')[1].split('}')[0])
            rhs = partial.split('=>')[1].split('where')[0]
            self.assertEqual(rhs.count('(dot '), (size + 1) // 2)

    def test_exact_reference_oracles(self):
        self.assertEqual(p.validate_workloads(), 60)

    def test_mixed_timeout_samples_are_preserved(self):
        case = self.manifest['cases'][0]
        samples = [dict(case_id=case['case_id'], stage='measured',
                        status=status, reason=status,
                        total_ms=total, construction_ms=total / 2,
                        verification_ms=total / 2, other_ms=0)
                   for status, total in [('proved', 1), ('time_limit', 10)]]
        summary = p.summarize(dict(manifest=dict(cases=[case]), records=samples))[0]
        self.assertEqual(summary['status'], 'mixed')
        self.assertEqual(summary['proved_repetitions'], 1)
        self.assertEqual(summary['median_total_ms'], 5.5)

    def test_plotting_changes_preserve_measurement_fingerprint(self):
        original = Path(p.__file__).read_text()
        expected = p.measurement_script_hash()
        restyled = original.replace('def render(data):',
                                    'def render(data):\n    """Updated presentation."""', 1)
        with patch.object(Path, 'read_text', return_value=restyled):
            self.assertEqual(p.measurement_script_hash(), expected)
        changed_workload = original.replace('CHAIN_SIZES = tuple(range(1, 11))',
                                            'CHAIN_SIZES = tuple(range(1, 12))', 1)
        with patch.object(Path, 'read_text', return_value=changed_workload):
            self.assertNotEqual(p.measurement_script_hash(), expected)

    def test_saved_samples_when_present(self):
        if not p.RESULT.exists():
            self.skipTest('measurement has not been run')
        data = json.loads(p.RESULT.read_text())
        p.validate(data)
        self.assertEqual(len(p.summarize(data)), 39)
        for r in data['records']:
            self.assertLessEqual(r['construction_ms'] + r['verification_ms'], r['total_ms'] + 1e-6)


if __name__ == '__main__':
    unittest.main()
