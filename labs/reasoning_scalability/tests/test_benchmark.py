"""Workload controls, successful-case filtering and phase accounting."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import os
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark as p
import workloads as b


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=b.HERE)
        self.addCleanup(directory.cleanup)
        output = Path(directory.name)
        self.paths = patch.multiple(p, OUTPUT=output, DIRECTORY=output / 'inputs',
                                    MANIFEST=output / 'inputs/manifest.json')
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.manifest = p.prepare()

    def test_automatic_only_and_fixed_additive_controls(self):
        cases = self.manifest['cases']
        self.assertEqual(len(cases), 14)
        self.assertEqual(self.manifest['chain_sizes'], [1,2,4,8,16,32,64])
        self.assertEqual([c['x'] for c in cases if c['panel']=='a'], [1,2,4,8,16,32,64])
        self.assertEqual(p.PHASES['a'], p.PHASES['b'])
        self.assertIn('feasibility', p.PHASES['c'])
        self.assertEqual({c['mode'] for c in cases}, {'automatic'})
        for c in cases:
            source = (b.ROOT / c['file']).read_text()
            lhs, rhs = source.split('=>')
            rhs = rhs.split('where')[0]
            self.assertNotIn('derive', source)
            self.assertNotIn(': dot', rhs)
            self.assertEqual(lhs.count('dot '), c['expected_constructors'])
            self.assertEqual(rhs.count('dot '), c['expected_constructors'])
            self.assertEqual(c['rank'], 'polymorphic')
            self.assertEqual(c['validation_ranks'], [2,3,5])
            self.assertIn('f32[Batch...,', source)
            self.assertIn('range(len(Batch))', source)
            self.assertIn('[len(Batch)+1]', source)
            self.assertIn('[len(Batch)]', source)
            self.assertNotIn('len(Batch)==',source.replace(' ',''))
            if c['panel'] == 'b':
                self.assertEqual(c['dots_per_term'], 2)
                self.assertEqual(c['expected_constructors'], 2*c['x'])
                self.assertEqual(lhs.count('add '), c['x']-1)
                self.assertEqual(rhs.count('add '), c['x']-1)
                self.assertIn('all([d>0 for d in Batch])', source)

    def test_runtime_uses_one_rank_independent_source(self):
        source = (b.ROOT / self.manifest['runtime_file']).read_text()
        self.assertNotIn('where', source)
        self.assertNotIn('[', source)
        self.assertEqual(source.count('dot '), 4)
        self.assertEqual(self.manifest['ranks'], [2,4,8,16,32,64,128,256])

    def test_independent_integer_oracles(self):
        self.assertEqual(self.manifest['dense_reference_checks'], 42)

    def test_successful_cases_only_and_arithmetic_mean_accounting(self):
        records = []
        for case, statuses in [('good', ['success','success']), ('mixed', ['success','unresolved'])]:
            for i, status in enumerate(statuses):
                records.append(dict(case_id=case, panel='a', x=1, stage='measured', round=i,
                                    status=status, total=10+10*i, preparation=2+2*i,
                                    checking_construction=3+3*i, lowering_validation=1+i, other=4+4*i))
        rows = p.summarize(dict(records=records))
        good, mixed = rows
        self.assertTrue(good['successful'])
        self.assertFalse(mixed['successful'])
        self.assertEqual(good['mean_total'], 15)
        self.assertEqual(sum(good['mean_'+k] for k in p.PHASES['a']), good['mean_total'])
        self.assertEqual((good['min_total'], good['max_total']), (10,20))
        records.append(dict(records[0], stage='warmup', round=-1, status='unresolved'))
        self.assertFalse(p.summarize(dict(records=records))[0]['successful'])

    def test_record_validation_rejects_missing_samples_or_wrong_phase_sum(self):
        case = self.manifest['cases'][0]
        manifest = copy.deepcopy(self.manifest)
        manifest['cases'] = [case]
        manifest['ranks'] = []
        samples = [dict(case_id=case['case_id'], panel='a', x=1, round=i, stage='measured',
                        status='success',total=10,preparation=2,checking_construction=3,
                        lowering_validation=1,other=4,validated_ranks=[2,3,5]) for i in range(2)]
        data = dict(schema_version=4,runtime_compilations=1,manifest=manifest,
                    warmups=0,repetitions=2,records=samples,compile_timeout_seconds=10,
                    case_outcomes=[dict(case_id=case['case_id'],panel='a',x=1,status='success')])
        p.validate(data)
        wrong = copy.deepcopy(data)
        wrong['records'][0]['other'] = 5
        with self.assertRaisesRegex(ValueError, 'phase sum'):
            p.validate(wrong)
        data['records'].pop()
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            p.validate(data)

    def test_compiled_rank_polymorphic_rules_match_multiple_ranks(self):
        binary = b.ROOT / 'bazel-bin/labs/reasoning_scalability/inference_benchmark'
        if not binary.exists():
            self.skipTest('build the optimized driver for process-level checks')
        manifest = copy.deepcopy(self.manifest)
        manifest['cases'] = [next(c for c in manifest['cases'] if c['panel']==panel and c['x']==2) for panel in 'ab']
        manifest['ranks'] = [2,4]
        custom = p.DIRECTORY / 'polymorphic_manifest.json'
        custom.write_text(json.dumps(manifest))
        result = p.DIRECTORY / 'polymorphic_result.json'
        subprocess.run([str(binary),str(custom),str(result),'0','2'],cwd=b.ROOT,
                       check=True,capture_output=True,timeout=45)
        data = json.loads(result.read_text())
        self.assertTrue(all(r['status']=='success' for r in data['case_outcomes']), data['case_outcomes'])
        self.assertEqual(data['runtime_compilations'],1)
        for sample in data['records']:
            if sample['panel'] in 'ab':
                self.assertEqual(sample['validated_ranks'],[2,3,5])
                self.assertLess(sample['total'],10000)

    def test_driver_stops_series_after_first_compilation_failure(self):
        binary = b.ROOT / 'bazel-bin/labs/reasoning_scalability/inference_benchmark'
        if not binary.exists():
            self.skipTest('build the optimized driver for process-level checks')
        manifest = copy.deepcopy(self.manifest)
        manifest['cases'] = manifest['cases'][:2]
        manifest['ranks'] = []
        broken = p.DIRECTORY / 'broken.tepl'
        broken.write_text('rule broken')
        manifest['cases'][0]['file'] = broken.relative_to(b.ROOT).as_posix()
        custom = p.DIRECTORY / 'stop_manifest.json'
        custom.write_text(json.dumps(manifest))
        result = p.DIRECTORY / 'stop_result.json'
        subprocess.run([str(binary),str(custom),str(result),'0','2'],cwd=b.ROOT,
                       check=True,capture_output=True,timeout=30)
        data = json.loads(result.read_text())
        self.assertEqual(data['compile_timeout_seconds'],10)
        self.assertEqual([r['status'] for r in data['case_outcomes']],['unresolved','not_run'])
        self.assertEqual(len(data['preflight']),1)
        self.assertEqual(data['records'],[])

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'hard-timeout control needs a FIFO')
    def test_hard_timeout_kills_blocked_import_and_stops_scaling(self):
        binary = b.ROOT / 'bazel-bin/labs/reasoning_scalability/inference_benchmark'
        if not binary.exists():
            self.skipTest('build the optimized driver for process-level checks')
        manifest = copy.deepcopy(self.manifest)
        manifest['cases'] = manifest['cases'][:2]
        manifest['ranks'] = []
        fifo = p.DIRECTORY / 'blocked_import.tepl'
        os.mkfifo(fifo)
        blocked = p.DIRECTORY / 'blocked_rule.tepl'
        blocked.write_text(f'from "{fifo.resolve()}" import TensorLang as t;\nrule blocked {{(t.dot A B)=>(t.dot A B)}}')
        manifest['cases'][0]['file'] = blocked.relative_to(b.ROOT).as_posix()
        custom = p.DIRECTORY / 'timeout_manifest.json'
        custom.write_text(json.dumps(manifest))
        result = p.DIRECTORY / 'timeout_result.json'
        subprocess.run([str(binary),str(custom),str(result),'0','2'],cwd=b.ROOT,
                       check=True,capture_output=True,timeout=25)
        data = json.loads(result.read_text())
        self.assertEqual([r['status'] for r in data['case_outcomes']],['time_limit','not_run'])
        self.assertEqual(len(data['preflight']),1)
        self.assertFalse(data['preflight'][0]['timing_complete'])
        self.assertGreaterEqual(data['preflight'][0]['total'],10000)
        self.assertEqual(data['records'],[])


if __name__ == '__main__':
    unittest.main()
