#!/usr/bin/env python3
"""16-input Einsum → TEPL representation and axis-inference experiment."""
import argparse
import collections
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import shutil
import zipfile
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import re
import subprocess
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIALECT = ROOT / 'tests/fixtures/semantic/dialects/tensor.tepl'
DTYPES = {'float32': 'f32', 'float64': 'f64', 'int16': 'i16',
          'int32': 'i32', 'int64': 'i64'}
FIELDS = ('lhs_contracting_dimensions', 'rhs_contracting_dimensions',
          'lhs_batching_dimensions', 'rhs_batching_dimensions')
CATEGORIES = {'gm': 'Graphical models', 'lm': 'Tensor network language models',
              'mc': 'Model counting', 'qc': 'Quantum computing',
              'rnd': 'Random', 'str': 'Structural', 'wmc': 'Weighted model counting'}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(record):
    """Validate portable input; avoid silently changing equation or dimensions."""
    equation = (''.join(map(chr, record['equation_codepoints'])) if 'equation_codepoints' in record
                else record['equation']).replace(' ', '')
    if equation.count('->') != 1 or '.' in equation:
        raise ValueError('explicit output required; ellipses unsupported')
    left, output = equation.split('->')
    terms = left.split(',')
    if len(terms) != len(record['shapes']) or not terms:
        raise ValueError('operand count mismatch')
    extents = {}
    for term, shape in zip(terms, record['shapes']):
        if len(term) != len(shape):
            raise ValueError('input rank mismatch')
        for axis, dim in zip(term, shape):
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise ValueError('invalid extent')
            if axis in extents and extents[axis] != dim:
                raise ValueError('inconsistent index extents')
            extents[axis] = dim
    if len(set(output)) != len(output) or not set(output) <= set(extents):
        raise ValueError('invalid output indices')
    for name in ('opt_flops', 'opt_size'):
        live = len(terms)
        path = record['paths'][name]
        if len(path) != live - 1:
            raise ValueError('path must contain n-1 binary contractions')
        for pair in path:
            if (len(pair) != 2 or any(type(i) is not int for i in pair)
                    or pair[0] == pair[1] or not all(0 <= i < live for i in pair)):
                raise ValueError('invalid linear path step')
            live -= 1
    return terms, output


def lower(record, path_name):
    """Linear path → dot_general + unary sum + final transpose witness.

    Retain a shared index as batch while another operand/output still needs it.
    Unique dead indices are reduced before the binary contraction.
    """
    terms, output = validate(record)
    nodes = [dict(op='input', name=f'A{i}', axes=list(t), inputs=[])
             for i, t in enumerate(terms)]
    live = list(range(len(terms)))

    def add(op, inputs, axes, attrs):
        nodes.append(dict(op=op, inputs=inputs, axes=list(axes), attrs=attrs))
        return len(nodes) - 1

    def reduce(node, needed):
        axes = nodes[node]['axes']
        dims = [i for i, a in enumerate(axes) if a not in needed]
        return (add('reduce_sum', [node], [a for a in axes if a in needed],
                    {'dimensions': dims}) if dims else node)

    for li, ri in record['paths'][path_name]:
        lhs, rhs = live[li], live[ri]
        other = set(output)
        for i, node in enumerate(live):
            if i not in (li, ri):
                other.update(nodes[node]['axes'])
        lhs = reduce(lhs, other | set(nodes[rhs]['axes']))
        rhs = reduce(rhs, other | set(nodes[lhs]['axes']))
        la, ra = nodes[lhs]['axes'], nodes[rhs]['axes']
        shared = [a for a in la if a in ra]
        batch = [a for a in shared if a in other]
        contract = [a for a in shared if a not in other]
        attrs = dict(zip(FIELDS, ([la.index(a) for a in contract],
                                  [ra.index(a) for a in contract],
                                  [la.index(a) for a in batch],
                                  [ra.index(a) for a in batch])))
        axes = batch + [a for a in la if a not in shared] + [a for a in ra if a not in shared]
        result = add('dot_general', [lhs, rhs], axes, attrs)
        live = [n for i, n in enumerate(live) if i not in (li, ri)] + [result]
    root = reduce(live[0], set(output))
    axes = nodes[root]['axes']
    root = add('transpose', [root], output,
               {'permutation': [axes.index(a) for a in output]})
    return dict(nodes=nodes, root=root)


def attrs_text(attrs):
    return ', '.join(f'{k} = {json.dumps(v)}' for k, v in attrs.items())


def expression(tree, automatic=False):
    # Iterative traversal avoids Python recursion limits on skewed trees.
    parts, stack = [], [tree['root']]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parts.append(item)
            continue
        n = tree['nodes'][item]
        if n['op'] == 'input':
            parts.append(n['name'])
            continue
        label = '' if automatic else f'n{item}: '
        parts.append(f'({label}t.{n["op"]} ')
        stack.append(')')
        for child in reversed(n['inputs']):
            stack.extend([' ', child])
    return ''.join(parts)


def header():
    return f'from "{DIALECT}" import TensorLang as t\n\n'


def graph_source(record, tree, name):
    lines = [f'graph {name} {{']
    for i, shape in enumerate(record['shapes']):
        lines.append(f'  input A{i}: {DTYPES[record["dtype"]]}{json.dumps(shape)}')
    for i, n in enumerate(tree['nodes']):
        if n['op'] == 'input':
            continue
        operands = ' '.join(tree['nodes'][j]['name'] if tree['nodes'][j]['op'] == 'input'
                            else f'v{j}' for j in n['inputs'])
        lines.append(f'  let v{i} = (t.{n["op"]}[{attrs_text(n["attrs"])}] {operands})')
    lines.extend([f'  yield v{tree["root"]}', '}'])
    return '\n'.join(lines) + '\n'


def rule_source(record, source, target):
    lines = ['rule convert {']
    terms, _ = validate(record)
    dimensions = {a: f'D{i}' for i, a in enumerate(dict.fromkeys(''.join(terms)))}
    for i, term in enumerate(terms):
        lines.append(f'  A{i}: {DTYPES[record["dtype"]]}[' + ', '.join(dimensions[a] for a in term) + ']')
    lines.extend(['  ' + expression(source), '  => ' + expression(target, True)])
    # Keep rank/dtype/shared-extent declarations and captured LHS descriptors.
    # Core feasibility checks applicability to the replacement skeleton.
    if dimensions:
        values = ', '.join(dimensions.values())
        lines.extend(['  where', f'    all([d >= 2 for d in [{values}]])'])
    lines.append('}')
    return header() + '\n'.join(lines) + '\n\n' + graph_source(record, source, 'before')


def skeleton(tree):
    # A flat prefix encoding keeps comparisons nonrecursive for large trees.
    result, stack = [], [tree['root']]
    while stack:
        n = tree['nodes'][stack.pop()]
        result.append((n['op'], n.get('name') if n['op'] == 'input' else len(n['inputs'])))
        stack.extend(reversed(n['inputs']))
    return result


def compiled_graph(program, name):
    """Decode concrete graph attributes, never constructor/proof internals."""
    p = program['program']
    graph = next(g for g in p['graphs'] if g['name'] == name)
    operations = {o['id']: o for o in p['operations']}
    schemas = {s['id']: s for s in p['attribute_schemas']}
    nodes = []
    for n in graph['nodes']:
        if n['kind'] == 'input':
            nodes.append(dict(op='input', name=n['text'], inputs=[],
                              shape=[int(d) for d in n['shape']], dtype=n['dtype']))
            continue
        op = operations[n['operation']]
        fields = schemas[op['attributes']]['fields']
        attrs = {}
        for field, val in zip(fields, n['attributes'], strict=True):
            if val['kind'] != 'list' or any(v['kind'] != 'integer' for v in val['elements']):
                raise ValueError('expected concrete axis-list attribute')
            attrs[field['name']] = [int(v['text']) for v in val['elements']]
        nodes.append(dict(op=op['name'], inputs=n['operands'], attrs=attrs))
    return dict(nodes=nodes, root=graph['root'])


def axis_inference_metrics(tree):
    """Count concrete RHS axis-list fields and their position occurrences.

    Count reachable replacement operations only. An empty inferred list is one
    completed field but contributes zero axis positions. Positions on different
    operands/nodes count separately, even if they name the same logical index.
    """
    counts = dict(axis_fields=0, nonempty_axis_fields=0, empty_axis_fields=0,
                  axis_positions=0, contracting_positions=0, batching_positions=0,
                  reduction_positions=0, permutation_positions=0,
                  rhs_operations=0, rhs_contractions=0)
    seen, stack = set(), [tree['root']]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        node = tree['nodes'][i]
        stack.extend(node['inputs'])
        if node['op'] == 'input':
            continue
        counts['rhs_operations'] += 1
        if node['op'] == 'dot_general':
            counts['rhs_contractions'] += 1
            roles = dict(zip(FIELDS, ['contracting_positions', 'contracting_positions',
                                     'batching_positions', 'batching_positions']))
        elif node['op'] == 'reduce_sum':
            roles = {'dimensions': 'reduction_positions'}
        elif node['op'] == 'transpose':
            roles = {'permutation': 'permutation_positions'}
        else:
            raise ValueError('unsupported axis metric operation: ' + node['op'])
        for field, role in roles.items():
            values = node['attrs'][field]
            if not isinstance(values, list) or any(type(v) is not int for v in values):
                raise ValueError('axis metrics require concrete integer lists')
            counts['axis_fields'] += 1
            counts['nonempty_axis_fields' if values else 'empty_axis_fields'] += 1
            counts['axis_positions'] += len(values)
            counts[role] += len(values)
    return counts


def verify(record, tree):
    """Independent port-partition oracle for read-once dot/sum/permutation.

    Compare all input-port equivalence classes, reduced classes and ordered output
    ports against the equation. Equal extents alone never identify two axes.
    """
    terms, output = validate(record)
    parent, labels, values, seen, reductions = [], [], {}, set(), []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    def positions(seq, rank):
        if len(seq) != len(set(seq)) or any(type(i) is not int or not 0 <= i < rank for i in seq):
            raise ValueError('illegal axis selector')

    # Traverse only reachable nodes; dead graph nodes cannot hide missing inputs.
    order, stack, visits = [], [(tree['root'], False)], collections.Counter()
    while stack:
        i, done = stack.pop()
        if not 0 <= i < len(tree['nodes']):
            raise ValueError('invalid node reference')
        n = tree['nodes'][i]
        if done:
            order.append(i)
            continue
        visits[i] += 1
        if visits[i] != 1:
            raise ValueError('graph is not read-once')
        stack.append((i, True))
        for child in reversed(n['inputs']):
            if child >= i:
                raise ValueError('graph is not producer-first')
            stack.append((child, False))
    for i in order:
        n = tree['nodes'][i]
        if n['op'] == 'input':
            match = re.fullmatch(r'A(\d+)', n['name'])
            if not match or int(match[1]) >= len(terms) or n['name'] in seen:
                raise ValueError('unexpected or repeated input')
            k = int(match[1]); seen.add(n['name'])
            if n.get('shape', record['shapes'][k]) != record['shapes'][k]:
                raise ValueError('input shape changed')
            if n.get('dtype', DTYPES[record['dtype']]) != DTYPES[record['dtype']]:
                raise ValueError('input dtype changed')
            ports = []
            for a in terms[k]:
                ports.append(len(parent)); parent.append(len(parent)); labels.append(a)
            values[i] = ports
            continue
        xs = [values[j] for j in n['inputs']]
        at = n['attrs']
        if n['op'] == 'transpose' and len(xs) == 1:
            perm = at['permutation']; positions(perm, len(xs[0]))
            if len(perm) != len(xs[0]):
                raise ValueError('incomplete permutation')
            values[i] = [xs[0][j] for j in perm]
        elif n['op'] == 'reduce_sum' and len(xs) == 1:
            dims = at['dimensions']; positions(dims, len(xs[0]))
            reductions.extend(xs[0][j] for j in dims)
            values[i] = [a for j, a in enumerate(xs[0]) if j not in dims]
        elif n['op'] == 'dot_general' and len(xs) == 2:
            lc, rc, lb, rb = (at[f] for f in FIELDS)
            for seq, rank in [(lc + lb, len(xs[0])), (rc + rb, len(xs[1]))]:
                positions(seq, rank)
            if len(lc) != len(rc) or len(lb) != len(rb):
                raise ValueError('unpaired axes')
            for l, r in zip(lc + lb, rc + rb):
                union(xs[0][l], xs[1][r])
            reductions.extend(xs[0][j] for j in lc)
            values[i] = ([xs[0][j] for j in lb]
                         + [a for j, a in enumerate(xs[0]) if j not in lc + lb]
                         + [a for j, a in enumerate(xs[1]) if j not in rc + rb])
        else:
            raise ValueError('unsupported oracle operation')
    if len(seen) != len(terms):
        raise ValueError('missing input')
    by_label, by_class = {}, {}
    for i, a in enumerate(labels):
        root = find(i)
        if a in by_label and by_label[a] != root:
            raise ValueError('original index split into different classes')
        if root in by_class and by_class[root] != a:
            raise ValueError('different original indices merged')
        by_label[a], by_class[root] = root, a
    actual = [find(i) for i in values[tree['root']]]
    if actual != [by_label[a] for a in output]:
        raise ValueError('output interface mismatch')
    reduced = [find(i) for i in reductions]
    if len(set(reduced)) != len(reduced) or set(reduced) != set(by_class) - set(actual):
        raise ValueError('incorrect reduction binding')
    return True


def command(binary, args, directory, label, timeout):
    start = time.perf_counter()
    try:
        p = subprocess.run([str(binary), *map(str, args)], text=True,
                           capture_output=True, timeout=timeout)
        result = dict(status='ok' if p.returncode == 0 else 'error', returncode=p.returncode,
                      stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as exc:
        result = dict(status='timeout', stdout=str(exc.stdout or ''), stderr=str(exc.stderr or ''))
    result['wall_ms'] = (time.perf_counter() - start) * 1000
    save(directory / (label + '.log.json'), result)
    return result


def evaluate(record, directory, binary, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    save(directory / 'input.json', record)
    row = dict(id=record['id'], category=record.get('category', CATEGORIES.get(record['id'].split('_')[0], 'Other')),
               tensors=len(record['shapes']), representation='not_attempted', directions=[])
    try:
        terms, _ = validate(record)
        if record['dtype'] not in DTYPES:
            row.update(representation='unsupported', reason='unsupported_dtype'); return row
        if any(len(t) != len(set(t)) for t in terms):
            row.update(representation='unsupported', reason='trace_or_diagonal'); return row
        trees = {p: lower(record, p) for p in ('opt_flops', 'opt_size')}
        save(directory / 'witnesses.json', trees)
        for tree in trees.values():
            verify(record, tree)
        path = directory / 'explicit.tepl'
        path.write_text(header() + '\n'.join(graph_source(record, t, p) for p, t in trees.items()))
        compiled = directory / 'explicit.json'
        result = command(binary, ['compile', path, '--semantic-policy', 'algebraic', '-o', compiled], directory, 'representation', timeout)
        row['representation_ms'] = result['wall_ms']
        if result['status'] != 'ok':
            row.update(representation=result['status'], reason='explicit_compile'); return row
        program = json.loads(compiled.read_text())
        for name in trees:
            verify(record, compiled_graph(program, name))
        row['representation'] = 'ok'
        for source, target in [('opt_flops', 'opt_size'), ('opt_size', 'opt_flops')]:
            d = dict(direction=f'{source}_to_{target}', status='not_attempted')
            row['directions'].append(d)
            if skeleton(trees[source]) == skeleton(trees[target]):
                d['status'] = 'identical_structure'; continue
            if any(dim < 2 for shape in record['shapes'] for dim in shape):
                d['status'] = 'outside_core_extent_domain'; continue
            folder = directory / d['direction']; folder.mkdir(exist_ok=True)
            src, artifact = folder / 'automatic.tepl', folder / 'automatic.json'
            src.write_text(rule_source(record, trees[source], trees[target]))
            result = command(binary, ['compile', src, '--semantic-policy', 'algebraic', '--core', '-o', artifact], folder, 'inference_compile', timeout)
            d['compile_ms'] = result['wall_ms']
            if result['status'] != 'ok':
                d['status'] = 'compile_' + result['status']; continue
            d['compiled'] = True
            exported = folder / 'rewritten.tepl'
            result = command(binary, ['rewrite', artifact, '--graph', 'before', '--rule', 'convert', '--max-steps', '1', '--trace', '-o', exported], folder, 'rewrite', timeout)
            d['rewrite_process_ms'] = result['wall_ms']
            # --max-steps 1 intentionally bounds the experiment. CLI exit 3
            # at step_limit can still contain exactly one committed rewrite;
            # broader rules may match again. Validate that concrete output.
            committed = "Successfully applied rule 'convert'." in result['stderr']
            bounded_success = (result.get('returncode') == 3 and committed and
                               'Rewrites: 1; stop: step_limit;' in result['stderr'])
            if result['status'] != 'ok' and not bounded_success:
                d['status'] = 'rewrite_' + result['status']; continue
            if not committed:
                d['status'] = 'not_applied'; continue
            d['applied'] = True
            concrete = folder / 'rewritten.json'
            result = command(binary, ['compile', exported, '--semantic-policy', 'algebraic', '-o', concrete], folder, 'export_compile', timeout)
            if result['status'] != 'ok':
                d['status'] = 'export_' + result['status']; continue
            try:
                actual = compiled_graph(json.loads(concrete.read_text()), 'before')
                if skeleton(actual) != skeleton(trees[target]):
                    raise ValueError('replacement skeleton mismatch')
                verify(record, actual)
                d['axis_inference'] = axis_inference_metrics(actual)
                d['status'] = 'verified'
            except (ValueError, KeyError, StopIteration) as exc:
                d.update(status='verification_failed', reason=str(exc))
        return row
    except (ValueError, KeyError, StopIteration, TypeError, IndexError, OverflowError) as exc:
        row.update(representation='invalid_or_oracle_failed', reason=str(exc)); return row


def report(rows):
    lines = ['# Einsum → TEPL evaluation', '',
             'Counts are measured, not metadata estimates. Representation requires both explicit graphs to compile and pass the independent port oracle. Inference requires a committed rewrite and independent verification.', '',
             '| Category | Problems | Representable | Nonidentical pairs | Inference compiled | Applied | Verified | Compile median / p95 (ms) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def summary(group, name):
        ds = [d for r in group for d in r['directions'] if d['status'] != 'identical_structure']
        times = sorted(d['compile_ms'] for d in ds if 'compile_ms' in d)
        timing = f'{statistics.median(times):.2f} / {times[math.ceil(.95*len(times))-1]:.2f}' if times else '—'
        return f'| {name} | {len(group)} | {sum(r["representation"] == "ok" for r in group)} | {len(ds)} | {sum(d.get("compiled", False) for d in ds)} | {sum(d.get("applied", False) for d in ds)} | {sum(d["status"] == "verified" for d in ds)} | {timing} |'
    for cat in sorted({r['category'] for r in rows}):
        lines.append(summary([r for r in rows if r['category'] == cat], cat))
    lines.extend([summary(rows, 'Total'), '',
                  'Compile timing includes CLI startup, parsing, synthesis, conditional semantic verification, lowering and serialization; includes failed/timed-out attempts. Rewrite process time in raw data includes loading and graph matching; it is not isolated attribute-construction latency.', '',
                  'Identical operation/capture skeletons are excluded from the pair denominator. Extent-domain exclusions remain in the pair denominator. Representation failures remain in the problem denominator. No filtered result is extrapolated to 168 problems.', '', '## Outcomes', '', '```json',
                  json.dumps(dict(representation=dict(collections.Counter(r['representation'] for r in rows)),
                                  inference=dict(collections.Counter(d['status'] for r in rows for d in r['directions']))), indent=2), '```', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    impzip = sub.add_parser('import-zip', help='export trusted official ZIP directly without extracting tensors')
    impzip.add_argument('archive', type=Path)
    impzip.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--input', type=Path, default=HERE / 'fixtures/local16.json')
    run.add_argument('--out', type=Path, default=HERE / 'results/local16')
    run.add_argument('--binary', type=Path, default=ROOT / 'bazel-bin/tepl')
    run.add_argument('--timeout', type=float, default=10)
    run.add_argument('--jobs', type=int, default=4, help='parallel independent cases; timing under contention')
    run.add_argument('--compress-artifacts', action='store_true', help='gzip completed case artifacts to conserve disk')
    args = parser.parse_args()
    if args.command == 'import-zip':
        import pickle  # Only explicitly supplied, trusted official files.
        import io
        records = []
        archive = zipfile.ZipFile(args.archive)
        paths = sorted(n for n in archive.namelist() if Path(n).suffix in ('.pkl', '.p', '.pickle'))
        if not paths:
            parser.error('no pickle instances found')
        for i, path in enumerate(paths):
            data = archive.read(path)
            equation, tensors, metas, _ = pickle.load(io.BytesIO(data))
            dtypes = {str(t.dtype) for t in tensors}
            if len(dtypes) != 1:
                raise ValueError(f'{path}: mixed dtypes')
            records.append(dict(id=Path(path).stem, equation=equation,
                                shapes=[list(t.shape) for t in tensors], dtype=next(iter(dtypes)),
                                paths=dict(opt_size=metas[0][0], opt_flops=metas[1][0]),
                                provenance=dict(file=Path(path).name, sha256=hashlib.sha256(data).hexdigest(),
                                                dataset_doi='10.5281/zenodo.11477304', license='CC-BY-4.0')))
            # Some official large equations use surrogate code points as index
            # labels. JSON surrogate-pair decoding can merge adjacent labels;
            # preserve these equations as explicit code points instead.
            if any(0xD800 <= ord(c) <= 0xDFFF for c in equation):
                records[-1]['equation_codepoints'] = list(map(ord, equation))
                del records[-1]['equation']
            del tensors, data, metas
            print(f'{i+1}/{len(paths)} exported {Path(path).stem}', flush=True)
        archive.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, ensure_ascii=True, separators=(',', ':')) + '\n')
        print(f'Exported {len(records)} instances to {args.out}')
        return
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    if not 1 <= args.jobs <= 8:
        parser.error('jobs must be between 1 and 8')
    records = json.loads(args.input.read_text())
    if not isinstance(records, list) or not records:
        parser.error('input must be a nonempty JSON array')
    if len({r['id'] for r in records}) != len(records):
        parser.error('duplicate instance ids')
    if any(len(r['shapes']) != 16 for r in records):
        parser.error('experiment requires exactly 16 inputs per case; use local_cases.py to extract them')
    binary, out = args.binary.resolve(), args.out.resolve()
    if not binary.is_file():
        parser.error(f'build TEPL first: bazel build //:tepl ({binary})')
    if out.exists() and any(out.iterdir()):
        parser.error('output directory must be empty; use a fresh run directory')
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'provenance.json', dict(utc=datetime.now(timezone.utc).isoformat(),
         git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
         binary=str(binary), binary_sha256=digest(binary), input_sha256=digest(args.input),
         harness_sha256=digest(__file__), dialect_sha256=digest(DIALECT),
         platform=platform.platform(), strategy='core', timeout_s=args.timeout,
         max_tensors=16, guard_policy='domain', jobs=args.jobs, compress_artifacts=args.compress_artifacts, policy='algebraic', synthetic=False))
    rows_by_index = {}

    def work(i, record):
        folder = out / f'{i:04d}'
        row = evaluate(record, folder, binary, args.timeout)
        save(folder / 'result.json', row)
        if args.compress_artifacts:
            for path in folder.rglob('*'):
                if path.suffix in ('.json', '.tepl') and path.name != 'result.json':
                    with path.open('rb') as src, gzip.open(str(path) + '.gz', 'wb', compresslevel=1) as dst:
                        shutil.copyfileobj(src, dst)
                    path.unlink()
        return i, row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(work, i, record) for i, record in enumerate(records)]
        for future in as_completed(futures):
            i, row = future.result()
            rows_by_index[i] = row
            rows = [rows_by_index[k] for k in sorted(rows_by_index)]
            save(out / 'results.json', rows)
            print(f'{len(rows)}/{len(records)} {row["id"]}: {row["representation"]} {[d["status"] for d in row["directions"]]}', flush=True)
    rows = [rows_by_index[k] for k in sorted(rows_by_index)]
    with (out / 'results.csv').open('w', newline='') as f:
        fields = ['id', 'category', 'tensors', 'representation', 'direction', 'status', 'applied', 'compile_ms', 'rewrite_process_ms', 'inferred_axis_fields', 'inferred_axis_positions', 'reason']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for r in rows:
            for d in r['directions'] or [{}]:
                writer.writerow({**{k: v for k, v in r.items() if k != 'directions'}, **d,
                                 'inferred_axis_fields': d.get('axis_inference', {}).get('axis_fields'),
                                 'inferred_axis_positions': d.get('axis_inference', {}).get('axis_positions')})
    (out / 'report.md').write_text(report(rows))


if __name__ == '__main__':
    main()
