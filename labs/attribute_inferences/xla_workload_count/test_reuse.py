"""Checks reuse boundaries with saved data; never invokes XLA."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

LAB = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('workload_count', LAB / 'run.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
FIXTURE = LAB / 'tests/fixtures/dlrm-raw.json'


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.raw = api.read_json(FIXTURE)
        self.registry = api.read_json(LAB / 'support-registry.json')
        self.metadata = api.read_json(LAB / 'counter-provenance.json')

    def test_saved_dlrm_counts_and_classifications(self):
        rows, result = api.analyze(self.raw, self.registry, self.metadata)
        self.assertEqual((len(rows), result['total_applications']), (32, 201))
        self.assertEqual(result['semantic_model_coverage']['rule_variants'], 20)
        self.assertEqual(result['semantic_model_coverage']['associated_native_applications'], 103)
        self.assertEqual([result['semantic_categories'][s]['rule_variants'] for s in api.STATUSES], [14, 6, 6, 6, 0])

    def test_unseen_rule_is_counted_and_queued(self):
        self.raw['rules'].append(dict(original_line=910, function='HandleAbs', applications=3))
        self.raw['total_applications'] += 3
        self.raw['distinct_rule_sites'] += 1
        rows, result = api.analyze(self.raw, self.registry, self.metadata)
        self.assertEqual((len(rows), result['total_applications']), (33, 204))
        self.assertEqual(result['semantic_categories']['unreviewed'], dict(rule_variants=1, applications=3))
        self.assertEqual(result['semantic_model_coverage']['rule_variants'], 20)
        self.assertEqual(result['semantic_model_coverage']['total_rule_variants'], 33)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            api.write_report(out, self.raw, self.registry, self.metadata, LAB / 'support-registry.json', 'test')
            self.assertIn('AS-910', (out / 'review-needed.csv').read_text())
            self.assertTrue((out / 'review-context/AS-910.txt').exists())

    def test_source_change_blocks_reuse(self):
        self.metadata['original_algebraic_simplifier_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Counter/registry mismatch'):
            api.analyze(self.raw, self.registry, self.metadata)

    def test_profile_change_does_not_reuse_support(self):
        _, result = api.analyze(self.raw, self.registry, self.metadata, profiles_valid=False)
        self.assertEqual(result['semantic_categories']['unreviewed'], dict(rule_variants=32, applications=201))
        self.assertEqual(result['semantic_categories']['direct']['rule_variants'], 0)

    def test_bookkeeping_pair_mismatch_is_rejected(self):
        next(r for r in self.raw['rules'] if r['original_line'] == 9009)['applications'] -= 1
        self.raw['total_applications'] -= 1
        with self.assertRaisesRegex(ValueError, 'pairing differs'):
            api.analyze(self.raw, self.registry, self.metadata)

    def test_corrupt_total_is_rejected(self):
        self.raw['total_applications'] += 1
        with self.assertRaisesRegex(ValueError, 'Raw count total'):
            api.analyze(self.raw, self.registry, self.metadata)


if __name__ == '__main__':
    unittest.main()
