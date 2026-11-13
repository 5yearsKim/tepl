"""Main scaling experiment: chain compilation, additive compilation, runtime rank."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

import numpy as np
import workloads as b

CHAIN_SIZES = (1, 2, 4, 8, 16, 32, 64)
RANKS = (2, 4, 8, 16, 32, 64, 128, 256)
TERMS = (1, 2, 4, 8, 16, 32, 64)
DOTS_PER_TERM = 2
SEED = 20261104
REFERENCE_BATCH_SHAPES = ((), (2,), (2, 3, 2))
VALIDATION_RANKS = (2, 3, 5)
OUTPUT = b.HERE / 'output'
DIRECTORY = OUTPUT / 'inputs'
MANIFEST = DIRECTORY / 'manifest.json'
RESULT = OUTPUT / 'performance.json'
PHASES = {'a': ('preparation', 'checking_construction', 'lowering_validation', 'other'),
          'c': ('feasibility', 'attribute_construction', 'other'),
          'b': ('preparation', 'checking_construction', 'lowering_validation', 'other')}


def validate_workloads():
    """Dense integer oracle for chains/sums, independent of compiler reasoning."""
    rng = np.random.default_rng(SEED)
    checks = 0
    def check_chain(size, batch_shape):
        leaves, lhs, rhs = b.chain(size + 1)
        dims = {f'd{i}': 2 + i % 2 for i in range(size + 2)}
        batch_rank = len(batch_shape)
        values = {v.name: rng.integers(-2, 3, batch_shape + tuple(dims[a] for a in v.axes if a != 'b')).astype(object) for v in leaves}
        def evaluate(node):
            if not node.children:
                return values[node.name]
            a, c = map(evaluate, node.children)
            lc, rc, lb, rb = (node.attrs[f][0] for f in b.FIELDS)
            lf = next(i for i in range(3) if i not in (lc, lb))
            rf = next(i for i in range(3) if i not in (rc, rb))
            def axes(index):
                return tuple(range(batch_rank)) if index == 0 else (batch_rank + index - 1,)
            return np.matmul(a.transpose(axes(lb)+axes(lf)+axes(lc)),
                             c.transpose(axes(rb)+axes(rc)+axes(rf)))
        oracle = values[leaves[0].name]
        for leaf in leaves[1:]:
            oracle = np.matmul(oracle, values[leaf.name])
        np.testing.assert_array_equal(evaluate(lhs), oracle)
        np.testing.assert_array_equal(evaluate(rhs), oracle)
        return evaluate(lhs), evaluate(rhs)
    for batch_shape in REFERENCE_BATCH_SHAPES:
        for size in CHAIN_SIZES:
            check_chain(size, batch_shape)
            checks += 1
        for terms in TERMS:
            pairs = [check_chain(DOTS_PER_TERM, batch_shape) for _ in range(terms)]
            np.testing.assert_array_equal(sum(x[0] for x in pairs), sum(x[1] for x in pairs))
            checks += 1
    return checks


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / 'tepl.toml').write_text('tepl-version = "0.2.0"\n')
    relative = Path(os.path.relpath(b.DIALECT, DIRECTORY)).as_posix()
    header = f'from "{relative}" import TensorLang as t\nuse t::{{dot,add}}\n'
    cases = []
    def add(name, source, panel, size, expected):
        path = DIRECTORY / (name + '.tepl')
        path.write_text(header + source)
        cases.append(dict(case_id=name, panel=panel, x=size, mode='automatic', rank='polymorphic', validation_ranks=list(VALIDATION_RANKS),
                          dots_per_term=DOTS_PER_TERM if panel == 'b' else size,
                          expected_constructors=expected, file=Path(os.path.relpath(path, b.ROOT)).as_posix(),
                          input_hash=b.digest(path)))
    for size in CHAIN_SIZES:
        name, source = b.dot_source(size)
        add(name, source, 'a', size, size)
    for terms in TERMS:
        name, source = b.additive_source(terms, DOTS_PER_TERM)
        add(name, source, 'b', terms, terms * DOTS_PER_TERM)
    runtime = DIRECTORY / 'runtime_rank_independent.tepl'
    runtime.write_text(header + 'rule runtime {(dot (dot A B) C)=>(dot A (dot B C))}\n')
    manifest = dict(schema_version=4, seed=SEED, cases=cases, ranks=list(RANKS),
                    chain_sizes=list(CHAIN_SIZES), terms=list(TERMS), dots_per_term=DOTS_PER_TERM,
                    runtime_file=Path(os.path.relpath(runtime, b.ROOT)).as_posix(), runtime_input_hash=b.digest(runtime),
                    dense_reference_checks=validate_workloads(), rank_polymorphic=True,
                    reference_batch_shapes=[list(x) for x in REFERENCE_BATCH_SHAPES], validation_ranks=list(VALIDATION_RANKS))
    b.save_json(MANIFEST, manifest)
    return manifest


def fingerprint(binary):
    paths = [p for p in (b.ROOT / 'src').rglob('*') if p.is_file()]
    paths += [b.HERE / 'inference_benchmark.cc', b.HERE / 'BUILD.bazel',
              b.HERE / 'scripts/workloads.py', b.HERE / 'scripts/benchmark.py', b.DIALECT,
              b.ROOT / '.bazelrc', b.ROOT / 'MODULE.bazel', MANIFEST, DIRECTORY / 'tepl.toml']
    def name(path):
        if path.is_relative_to(b.ROOT):
            return str(path.relative_to(b.ROOT))
        return 'run-inputs/' + str(path.relative_to(DIRECTORY))
    return dict(binary_hash=b.digest(binary),
                dependencies={name(p): b.digest(p) for p in paths})


def validate(data, binary=None):
    if data['schema_version'] != 4 or data['runtime_compilations'] != 1:
        raise ValueError('incompatible experiment format or repeated runtime compilation')
    if binary and data['fingerprint'] != fingerprint(binary):
        raise ValueError('stale experiment inputs or binary; rerun measurements')
    manifest = data['manifest']
    for case in manifest['cases']:
        if case['rank'] != 'polymorphic' or case['validation_ranks'] != manifest['validation_ranks']:
            raise ValueError('compile workloads must retain symbolic rank')
        if b.digest(b.ROOT / case['file']) != case['input_hash']:
            raise ValueError('changed workload: ' + case['case_id'])
    if b.digest(b.ROOT / manifest['runtime_file']) != manifest['runtime_input_hash']:
        raise ValueError('changed runtime workload')
    expected = {c['case_id'] for c in manifest['cases']} | {f'rank_{r}' for r in manifest['ranks']}
    groups = {name: [] for name in expected}
    outcomes = {r['case_id']: r for r in data['case_outcomes']}
    if set(outcomes) != expected or data['compile_timeout_seconds'] != 10:
        raise ValueError('missing outcomes or wrong compilation timeout')
    for sample in data['records']:
        groups[sample['case_id']].append(sample)
        if sample['panel'] in ('a','b') and sample['status'] == 'success' and sample.get('validated_ranks') != manifest['validation_ranks']:
            raise ValueError('missing polymorphic execution controls')
        if not sample.get('timing_complete', True):
            if sample['status'] == 'success':
                raise ValueError('successful sample lacks timings')
            continue
        times = [sample['total']] + [sample[k] for k in PHASES[sample['panel']]]
        if not all(math.isfinite(t) and t >= -1e-8 for t in times):
            raise ValueError('invalid phase timing')
        if not math.isclose(sum(times[1:]), times[0], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('phase sum does not equal measured total')
    for panel in 'abc':
        stopped = False
        for outcome in sorted((r for r in outcomes.values() if r['panel'] == panel), key=lambda r:r['x']):
            if stopped and outcome['status'] not in ('not_run','stopped'):
                raise ValueError('scaled beyond a failed case')
            stopped |= outcome['status'] != 'success'
    for name, samples in groups.items():
        rounds = [s['round'] for s in samples]
        if len(rounds) != len(set(rounds)):
            raise ValueError('missing or duplicate repetitions: ' + name)
        if outcomes[name]['status'] == 'success' and sorted(rounds) != list(range(-data['warmups'], data['repetitions'])):
            raise ValueError('missing or duplicate repetitions: ' + name)


def summarize(data):
    groups = {}
    for sample in data['records']:
        groups.setdefault(sample['case_id'], []).append(sample)
    definitions = {r['case_id']: r for r in data.get('case_outcomes', [])}
    if not definitions:
        definitions = {k: dict(case_id=k,panel=v[0]['panel'],x=v[0]['x'],status='success') for k,v in groups.items()}
    rows = []
    for name, definition in definitions.items():
        samples = groups.get(name, [])
        measured = [s for s in samples if s['stage'] == 'measured']
        successful = bool(measured) and definition['status'] == 'success' and all(s['status'] == 'success' for s in samples)
        row = dict(case_id=name, panel=definition['panel'], x=definition['x'], successful=successful,
                   samples=len(measured), reason=definition.get('reason',''),
                   outcomes=dict(Counter(s['status'] for s in samples)) or {definition['status']:1})
        for field in ('total',) + PHASES[row['panel']]:
            row['mean_' + field] = statistics.mean(s[field] for s in measured) if successful else None
        row.update(min_total=min(s['total'] for s in measured) if successful else None,
                   max_total=max(s['total'] for s in measured) if successful else None)
        if row['panel'] == 'c' and successful:
            row['mean_unprofiled_total'] = statistics.mean(s['unprofiled_total'] for s in measured)
            row['iterations'] = measured[0]['iterations']
        rows.append(row)
    return sorted(rows, key=lambda r: (r['panel'], r['x']))


def render(data):
    from figures import render_figures
    rows = summarize(data)
    figures = OUTPUT / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    render_figures(rows, figures)
    b.save_json(OUTPUT / 'performance_summary.json', rows)
    text = ['# Implicit-feasibility scaling experiment', '',
            f"Revision: `{data['revision']}` (dirty: {data['dirty']}). Backend: automatic metadata DAG (`--core`), algebraic policy.",
            f"Warmups: {data['warmups']}; measured repetitions: {data['repetitions']}. Runtime program compilations: 1.",
            'Bars and black line use arithmetic means of exactly the same samples. Whiskers show min–max repeated totals.',
            'Only cases with successful warmups and every measured repetition enter the figure. All outcomes remain in JSON.', '',
            'Every compilation has a hard 10 s wall-clock timeout in an isolated worker. '
            'Ascending preflight stops each panel at its first failure; larger sizes are not attempted. '
            'Failures in any repetition also stop and invalidate that case and larger sizes in its series. '
            'Runtime scaling stops at its first resource/feasibility failure. Fork, IPC and runtime setup serialization/loading are outside phase timings.', '',
            'Compilation: source text is preloaded; preparation includes parsing, import reads, resolution and type checking. '
            'Checking/construction includes semantic admission, constructor emission and materialization. Lowering/validation uses the production factory. '
            'Other includes stage dispatch and intermediate teardown. Build time, workload-file reads and post-timing checks are excluded.', '',
            'Runtime: executeSound on one cached, compiled, rank-independent two-dot rule (C=true). '
            'Feasibility and attribute construction are directly instrumented, with shared interpreter caches; they are not separate executions. '
            'Other includes source legality replay, interpreter setup, allocation, result teardown and clock overhead. '
            'Per-call values are averages of calibrated ~10 ms batches. There is no graph matching, tensor-value execution, loading, synthesis or proof search inside runtime measurements. '
            'Unprofiled matched batches quantify observer overhead; no clock overhead is subtracted.', '',
            'Panel (a): 1, 2, 4, 8, 16, 32, 64 dots per side. Panel (b): 1–64 independent additive terms, two dots per term. Both compile rank-polymorphic rules with shapes [Batch...,Dleft,Dright]; Batch has arbitrary length, including zero. '
            'All RHS descriptors are automatic. Source batching lists are range(len(Batch)); contraction positions are len(Batch)+1 and len(Batch). Every batch/matrix extent remains positive. This is an unbounded-rank batched-matrix family, not unrestricted tensor layouts. '
            'Panel (c): ranks 2–256, all input extents 2, canonical batch prefixes and one contracted dimension.', '',
            f"Independent dense integer reference checks over batch prefixes [], [2], and [2,3,2]: {data['manifest']['dense_reference_checks']}. Runtime profile/plain attributes, layouts and work counts agree; independent positional RHS replay validates every successful rank outside timing.", '',
            'Outside phase timing, every successful compilation is checked at ranks 2, 3, and 5 with anisotropic matrix dimensions; each reuses the same compiled q/f and independently replays the RHS. These finite instances supplement the rank-independent constructor contract, not a proof of arbitrary equivalence.', '',
            '| Panel | X | Outcome | Mean total | Min–max |', '|---|---:|---|---:|---:|']
    for r in rows:
        if not r['successful']:
            text.append(f"| {r['panel']} | {r['x']} | excluded: {r['reason']} | — | — |")
            continue
        text.append(f"| {r['panel']} | {r['x']} | {'success' if r['successful'] else 'excluded: '+str(r['outcomes'])} | {r['mean_total']:.4f} | {r['min_total']:.4f}–{r['max_total']:.4f} |")
    text += ['', 'Compilation totals are ms; runtime totals are µs.', '',
             '| Rank | Instrumented mean (µs) | Unprofiled mean (µs) | Batch calls |', '|---:|---:|---:|---:|']
    for r in rows:
        if r['panel'] == 'c' and r['successful']:
            text.append(f"| {r['x']} | {r['mean_total']:.4f} | {r['mean_unprofiled_total']:.4f} | {r['iterations']} |")
    (OUTPUT / 'performance.md').write_text('\n'.join(text) + '\n')


def main():
    global OUTPUT, DIRECTORY, MANIFEST, RESULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=b.ROOT / 'bazel-bin/labs/attribute_inferences/reasoning_scalability/inference_benchmark')
    parser.add_argument('--out', type=Path, help='Fresh output directory; preserves the default recorded run')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.out:
        OUTPUT = args.out.resolve()
        DIRECTORY = OUTPUT / 'inputs'
        MANIFEST = DIRECTORY / 'manifest.json'
        RESULT = OUTPUT / 'performance.json'
        if not (args.check or args.report) and OUTPUT.exists() and any(OUTPUT.iterdir()):
            parser.error('choose a new or empty --out directory')
    if args.warmups < 0 or args.repetitions < 2:
        parser.error('need nonnegative warmups and at least two repetitions')
    binary = args.binary.resolve()
    if args.prepare:
        prepare()
        return
    if args.report or args.check:
        data = json.loads(RESULT.read_text())
        validate(data, binary)
        if args.report:
            render(data)
        return
    manifest = prepare()
    before = fingerprint(binary)
    with tempfile.TemporaryDirectory(prefix='tepl-scaling-') as directory:
        raw = Path(directory) / 'samples.json'
        subprocess.run([str(binary), str(MANIFEST), str(raw), str(args.warmups), str(args.repetitions)],
                       cwd=b.ROOT, check=True, timeout=1800)
        data = json.loads(raw.read_text())
    if before != fingerprint(binary):
        raise ValueError('measurement dependencies changed during run')
    data.update(manifest=manifest, warmups=args.warmups, repetitions=args.repetitions,
                fingerprint=before, revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=b.ROOT,text=True)),
                build_command='bazel build -c opt //labs/attribute_inferences/reasoning_scalability:inference_benchmark',
                runtime_order='alternating profiled/unprofiled batches',
                allowed_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                run_at_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),platform.processor()))
    validate(data, binary)
    b.save_json(RESULT, data)
    render(data)
    print(OUTPUT / 'figures/scalability.png')


if __name__ == '__main__':
    main()
