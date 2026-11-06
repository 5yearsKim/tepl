#!/usr/bin/env python3
"""Measure the eleven additional semantically declared cases separately."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.axis_fields import AxisTask
from scripts.paths import HERE, ROOT, TEPL
from scripts.run_synthesis import measure_case, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, default=ROOT / 'bazel-bin/tepl')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout <= 0:
        parser.error('jobs and timeout must be positive')
    compiler = args.compiler.resolve(strict=True)
    records = json.loads((HERE / 'data/synthesis_classification.json').read_text())['cases']
    operations = json.loads((HERE / 'data/hlo_support.json').read_text())['operations']
    selected = [r for r in records if not r['control'] and not r['in_scope']
                and all(operations[op]['semantic'] for op in r['required_operations'])]
    if len(selected) != 11:
        raise ValueError('unexpected candidate inventory; audit before running')
    paths = [HERE / r['source_file'] for r in selected]
    inputs = paths + [HERE / 'data/synthesis_classification.json', HERE / 'data/hlo_support.json',
                      HERE / 'scripts/axis_fields.py', HERE / 'scripts/synthesis_cases.py',
                      HERE / 'scripts/run_synthesis.py', Path(__file__).resolve(),
                      TEPL / 'dialects/hlo.tepl', TEPL / 'tepl.toml', compiler]
    hashes = {str(p): sha(p) for p in inputs}
    for row, path in zip(selected, paths):
        if sha(path) != row['source_sha256']:
            raise ValueError('source differs from frozen classification: ' + row['case'])
    # Tasks live inside the project so normal imports and project discovery work.
    with tempfile.TemporaryDirectory(prefix='expanded-', dir=TEPL / 'rules') as task_dir, \
         tempfile.TemporaryDirectory(prefix='tepl-expanded-artifacts-') as artifact_dir:
        tasks = []
        for original, path in zip(selected, paths):
            task = AxisTask(path.read_text())
            generated = task.generate()
            target = Path(task_dir) / (original['case'] + '.tepl')
            target.write_text(generated)
            row = dict(original, in_scope=True, file=str(target.relative_to(ROOT)),
                       file_sha256=sha(target), axis_fields=task.targets,
                       task=('axis_inference' if task.targets else
                             'supplied_non_axis' if task.rule.attribute_sites() else 'attribute_free'))
            row.pop('exclusion_reason', None)
            tasks.append(row)
        def worker(row):
            result = measure_case(row, compiler, args.timeout, Path(artifact_dir))
            result['generated_source'] = (ROOT / row['file']).read_text()
            print(row['case'] + ': ' + result['outcome'], flush=True)
            return result
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            measured = list(pool.map(worker, tasks))
    if hashes != {str(p): sha(p) for p in inputs}:
        raise ValueError('inputs changed during run')
    result = {'protocol': 'expanded_semantic_candidates_v1',
              'run_at_utc': datetime.now(timezone.utc).isoformat(),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'compiler_sha256': hashes[str(compiler)], 'timeout_seconds_per_command': args.timeout,
              'jobs': args.jobs, 'input_sha256': hashes,
              'note': 'Eleven newly included cases only; uses the axis-only runner acceptance criteria. Temporary paths in commands are historical; generated_source and hashes record each task.',
              'results': measured}
    (HERE / 'results/expanded_semantics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
