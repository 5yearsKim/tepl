"""Semantic controls plus a real CLI smoke test; no tensor-value allocation."""
import copy
import importlib.util
import json
import gzip
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

LAB = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('einsum_lab', LAB / 'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
SAMPLES = json.loads((LAB / 'tests/fixtures/smoke.json').read_text())
LOCAL16 = json.loads((LAB / 'fixtures/local16.json').read_text())


class Semantics(unittest.TestCase):
    def test_runner_rejects_non_16_input_cases_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'inputs.json'
            source.write_text(json.dumps([SAMPLES[0]]))
            out = Path(folder) / 'out'
            result = subprocess.run([sys.executable, str(LAB / 'benchmark.py'), 'run',
                                     '--input', str(source), '--out', str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires exactly 16 inputs', result.stderr)
            self.assertFalse(out.exists())

    def test_lowered_paths_preserve_equation(self):
        for record in SAMPLES[:3]:
            for path in record['paths']:
                with self.subTest(record=record['id'], path=path):
                    self.assertTrue(b.verify(record, b.lower(record, path)))

    def test_hyperedge_survives_until_last_use(self):
        tree = b.lower(SAMPLES[1], 'opt_flops')
        dots = [n for n in tree['nodes'] if n['op'] == 'dot_general']
        self.assertEqual(dots[0]['attrs']['lhs_batching_dimensions'], [1])
        self.assertEqual(dots[0]['attrs']['lhs_contracting_dimensions'], [])
        self.assertEqual(dots[1]['attrs']['lhs_contracting_dimensions'], [0])

    def test_equal_extents_do_not_hide_wrong_axis_pairing(self):
        record = copy.deepcopy(SAMPLES[0])
        record['shapes'] = [[3,3]] * 3
        tree = b.lower(record, 'opt_size')
        tree['nodes'][3]['attrs']['lhs_contracting_dimensions'] = [1]
        with self.assertRaises(ValueError):
            b.verify(record, tree)

    def test_wrong_output_order_rejected_even_with_equal_extents(self):
        record = copy.deepcopy(SAMPLES[0])
        record['shapes'] = [[3,3]] * 3
        tree = b.lower(record, 'opt_size')
        tree['nodes'][-1]['attrs']['permutation'] = [1,0]
        with self.assertRaises(ValueError):
            b.verify(record, tree)

    def test_dropped_reduction_rejected(self):
        record = SAMPLES[2]
        tree = b.lower(record, 'opt_flops')
        node = next(n for n in tree['nodes'] if n['op'] == 'reduce_sum')
        node['attrs']['dimensions'] = []
        with self.assertRaises(ValueError):
            b.verify(record, tree)

    def test_hadamard_outer_product_scalar_and_singleton(self):
        for equation, shapes in [('i,i,i->i', [[3],[3],[3]]),
                                  ('i,j,k->ijk', [[2],[3],[4]]),
                                  (',i,i->', [[],[3],[3]]),
                                  ('i,i,i->', [[1],[1],[1]])]:
            record = dict(id='test', equation=equation, shapes=shapes, dtype='float64',
                          paths=dict(opt_flops=[[0,1],[0,1]], opt_size=[[1,2],[0,1]]))
            for name in record['paths']:
                self.assertTrue(b.verify(record, b.lower(record, name)))

    def test_surrogate_index_labels_survive_portable_json(self):
        # Adjacent high/low surrogate labels must not merge into one index.
        equation = '\ud801\udc00,\ud801x,\udc00y->xy'
        record = dict(id='surrogate', equation_codepoints=list(map(ord, equation)),
                      shapes=[[3,4],[3,2],[4,5]], dtype='float64',
                      paths=dict(opt_flops=[[0,1],[0,1]], opt_size=[[1,2],[0,1]]))
        record = json.loads(json.dumps(record))
        terms, out = b.validate(record)
        self.assertEqual([len(t) for t in terms], [2,2,2])
        self.assertEqual(out, 'xy')
        self.assertTrue(b.verify(record, b.lower(record, 'opt_size')))

    def test_invalid_path_and_extent_mismatch_rejected(self):
        record = copy.deepcopy(SAMPLES[0])
        record['paths']['opt_size'][0] = [1,1]
        with self.assertRaises(ValueError):
            b.validate(record)
        record = copy.deepcopy(SAMPLES[0])
        record['shapes'][0][1] = 7
        with self.assertRaises(ValueError):
            b.validate(record)

    def test_unreachable_inputs_cannot_satisfy_oracle(self):
        tree = b.lower(SAMPLES[0], 'opt_size')
        tree['root'] = 0
        with self.assertRaises(ValueError):
            b.verify(SAMPLES[0], tree)

    def test_domain_guards_keep_domain_without_original_value_pinning(self):
        record=SAMPLES[0]
        source,target=b.lower(record,'opt_flops'),b.lower(record,'opt_size')
        rule=b.rule_source(record,source,target).split('graph before')[0]
        self.assertIn('where',rule)
        self.assertIn('all([d >= 2',rule)
        self.assertNotIn('.attrs.',rule)
        self.assertNotIn('== 100',rule)

    def test_report_keeps_failures_in_denominator(self):
        rows = [dict(category='test', representation='ok', directions=[dict(status='compile_timeout', compile_ms=100)])]
        self.assertIn('| test | 1 | 1 | 1 | 0 | 0 | 0 |', b.report(rows))


class NumericalLowering(unittest.TestCase):
    def test_numpy_reference(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy optional: install requirements.txt for numeric controls')
        rng = np.random.default_rng(12)
        for record in SAMPLES[:3]:
            record = copy.deepcopy(record)
            record['shapes'] = [[2] * len(shape) for shape in record['shapes']]
            arrays = [rng.integers(-2, 3, size=shape) for shape in record['shapes']]
            expected = np.einsum(record['equation'], *arrays)
            for name in record['paths']:
                tree = b.lower(record, name)
                values = []
                for n in tree['nodes']:
                    op = n['op']
                    if op == 'input':
                        value = arrays[int(n['name'][1:])]
                    elif op == 'transpose':
                        value = values[n['inputs'][0]].transpose(n['attrs']['permutation'])
                    elif op == 'reduce_sum':
                        value = values[n['inputs'][0]].sum(axis=tuple(n['attrs']['dimensions']))
                    else:
                        left, right = (values[j] for j in n['inputs'])
                        lc, rc, lb, rb = (n['attrs'][f] for f in b.FIELDS)
                        la = list(range(left.ndim))
                        ra = list(range(left.ndim, left.ndim + right.ndim))
                        for l, r in zip(lc + lb, rc + rb):
                            ra[r] = la[l]
                        out = ([la[j] for j in lb] + [a for j, a in enumerate(la) if j not in lc + lb]
                               + [a for j, a in enumerate(ra) if j not in rc + rb])
                        value = np.einsum(left, la, right, ra, out)
                    values.append(value)
                np.testing.assert_array_equal(values[tree['root']], expected)


@unittest.skipUnless((b.ROOT / 'bazel-bin/tepl').exists(), 'build //:tepl for integration tests')
class CLI(unittest.TestCase):
    def test_inference_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, record in enumerate(SAMPLES[:3]):
                row = b.evaluate(record, Path(folder) / str(i), b.ROOT / 'bazel-bin/tepl', 20)
                self.assertEqual(row['representation'], 'ok', row)
                self.assertEqual([d['status'] for d in row['directions']], ['verified', 'verified'], row)

    def test_same_domain_rule_handles_new_extents_and_source_axis_attributes(self):
        original=copy.deepcopy(SAMPLES[0])
        source,target=b.lower(original,'opt_flops'),b.lower(original,'opt_size')
        rule=b.rule_source(original,source,target).split('graph before')[0]
        changed_sizes=copy.deepcopy(original)
        changed_sizes['shapes']=[[3,5],[7,5],[7,11]]
        changed_axes=copy.deepcopy(original)
        changed_axes['equation']='ki,jk,jl->il'
        changed_axes['shapes']=[[3,3],[3,3],[3,3]]
        singleton=copy.deepcopy(original)
        singleton['shapes']=[[1,1],[1,1],[1,1]]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i,record in enumerate([changed_sizes,changed_axes,singleton]):
                folder=root/str(i);folder.mkdir()
                inp,artifact=folder/'rule.tepl',folder/'program.json'
                inp.write_text(rule+b.graph_source(record,b.lower(record,'opt_flops'),'before'))
                compiled=b.command(b.ROOT/'bazel-bin/tepl',['compile',inp,'--core','--semantic-policy','algebraic','-o',artifact],folder,'compile',20)
                self.assertEqual(compiled['status'],'ok',compiled)
                exported=folder/'after.tepl'
                rewritten=b.command(b.ROOT/'bazel-bin/tepl',['rewrite',artifact,'--graph','before','--rule','convert','--max-steps','1','--trace','-o',exported],folder,'rewrite',20)
                if i == 2:
                    self.assertNotIn("Successfully applied rule 'convert'.",rewritten['stderr'])
                    continue
                self.assertIn("Successfully applied rule 'convert'.",rewritten['stderr'])
                concrete=folder/'after.json'
                check=b.command(b.ROOT/'bazel-bin/tepl',['compile',exported,'-o',concrete],folder,'check',20)
                self.assertEqual(check['status'],'ok',check)
                actual=b.compiled_graph(json.loads(concrete.read_text()),'before')
                self.assertTrue(b.verify(record,actual))

    def test_one_committed_rewrite_at_step_limit_is_verified(self):
        record=copy.deepcopy(SAMPLES[0])
        record['shapes']=[[3,3],[3,3],[3,3]]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            row=b.evaluate(record,folder,b.ROOT/'bazel-bin/tepl',20)
            self.assertEqual([d['status'] for d in row['directions']],['verified','verified'],row)
            logs=[json.loads(p.read_text()) for p in folder.glob('*/rewrite.log.json')]
            self.assertTrue(any(log.get('returncode')==3 and 'Rewrites: 1; stop: step_limit;' in log['stderr'] for log in logs))

    def test_dtype_and_trace_exclusions_have_distinct_reasons(self):
        with tempfile.TemporaryDirectory() as folder:
            for i, record in enumerate(SAMPLES[3:]):
                row = b.evaluate(record, Path(folder) / str(i), b.ROOT / 'bazel-bin/tepl', 20)
                self.assertEqual(row['representation'], 'unsupported')
                self.assertEqual(row['reason'], ['unsupported_dtype', 'trace_or_diagonal'][i])


@unittest.skipUnless((b.ROOT / 'bazel-bin/tepl').exists(), 'build //:tepl for integration tests')
class BatchCLI(unittest.TestCase):
    def test_parallel_compressed_run_keeps_results_and_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'inputs.json'
            supported = next(r for r in LOCAL16 if r['id'] == 'str_ctg_lattice_01__local_16')
            unsupported = next(r for r in LOCAL16 if r['dtype'] == 'complex64')
            source.write_text(json.dumps([supported, unsupported]))
            result = subprocess.run([sys.executable, str(LAB / 'benchmark.py'), 'run',
                                     '--input', str(source), '--out', str(folder / 'out'),
                                     '--jobs', '2', '--compress-artifacts'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads((folder / 'out/results.json').read_text())
            self.assertEqual([r['id'] for r in rows], [supported['id'], unsupported['id']])
            self.assertEqual(rows[0]['directions'][0]['status'], 'verified')
            self.assertTrue((folder / 'out/0000/result.json').exists())
            log = folder / 'out/0000/representation.log.json.gz'
            with gzip.open(log, 'rt') as f:
                self.assertEqual(json.load(f)['status'], 'ok')
            result = subprocess.run([sys.executable, str(LAB / 'axis_report.py'),
                                     '--run', str(folder / 'out'), '--input', str(source),
                                     '--out', str(folder / 'report')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = json.loads((folder / 'report/summary.json').read_text())['total']
            self.assertEqual(stats['cases'], 2)
            self.assertEqual(stats['verified'], 2)
            self.assertGreater(stats['axis_positions'], 0)
            self.assertEqual(stats['representation_reasons'], {'unsupported_dtype': 1})


    def test_zip_import_preserves_official_path_order(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy required to read official pickles')
        import pickle
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive = folder / 'instances.zip'
            size_path, flops_path = [[1, 2], [0, 1]], [[0, 1], [0, 1]]
            data = ('ik,jk,jl->il', [np.zeros((2,3)), np.zeros((4,3)), np.zeros((4,5))],
                    ((size_path, 0, 0, 0, 0), (flops_path, 0, 0, 0, 0)), 0)
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('instances/test.pkl', pickle.dumps(data))
            result = subprocess.run([sys.executable, str(LAB / 'benchmark.py'), 'import-zip',
                                     str(archive), '--out', str(folder / 'export.json')],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((folder / 'export.json').read_text())[0]
            self.assertEqual(record['paths']['opt_size'], size_path)
            self.assertEqual(record['paths']['opt_flops'], flops_path)
            self.assertTrue(b.verify(record, b.lower(record, 'opt_size')))


if __name__ == '__main__':
    unittest.main()
