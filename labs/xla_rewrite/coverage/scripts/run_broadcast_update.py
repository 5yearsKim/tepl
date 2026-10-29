#!/usr/bin/env python3
"""Rerun the frozen in-scope tasks with a broadcast/transpose companion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import HERE, ROOT
from scripts.run_synthesis import dependencies, measure_case, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, default=ROOT / 'bazel-bin/tepl')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout <= 0:
        parser.error('jobs and timeout must be positive')
    compiler = args.compiler.resolve(strict=True)
    classification = json.loads((HERE / 'data/synthesis_classification.json').read_text())
    rows = [dict(r) for r in classification['cases'] if r['in_scope'] and not r['control']]
    controls = [dict(r) for r in classification['cases'] if r['in_scope'] and r['control']]
    companion = ROOT / 'labs/xla_rewrite/tepl/rules/experiments/broadcast_transpose_broadcast.tepl'
    chosen = [r for r in rows if r['case'] == 'catalog_broadcast_transpose_broadcast']
    if len(chosen) != 1 or len(rows) != 112 or len(controls) != 1:
        raise ValueError('unexpected frozen case inventory')
    row = chosen[0]
    row.update(file=str(companion.relative_to(ROOT)), file_sha256=sha(companion),
               rule_name='broadcast_transpose_broadcast',
               axis_fields=[dict(row['axis_fields'][0], node='out')],
               variant='equivalent metadata construction')
    files = dependencies()
    files[str(companion.relative_to(ROOT))] = sha(companion)
    files[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__).resolve())
    compiler_sha = sha(compiler)
    with tempfile.TemporaryDirectory(prefix='tepl-broadcast-evaluation-') as directory:
        def worker(row):
            result = measure_case(row, compiler, args.timeout, Path(directory))
            print(row['case'] + ': ' + result['outcome'], flush=True)
            return result
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(worker, rows))
            control_results = list(pool.map(worker, controls))
        if any(r['outcome'] in {'verified', 'inferred'} for r in control_results):
            raise ValueError('negative control accepted')
    after = dependencies()
    after[str(companion.relative_to(ROOT))] = sha(companion)
    after[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__).resolve())
    if compiler_sha != sha(compiler) or files != after:
        raise ValueError('inputs changed during evaluation')
    output = dict(protocol='axis_only_broadcast_companion_v1',
                  run_at_utc=datetime.now(timezone.utc).isoformat(),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  compiler_sha256=compiler_sha, timeout_seconds_per_command=args.timeout,
                  jobs=args.jobs, dependencies=files,
                  note='Replaces the frozen broadcast-transpose case with an equivalent metadata-construction variant. Original frozen results remain unchanged.',
                  results=results, controls=control_results)
    (HERE / 'results/broadcast_update.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
