#!/usr/bin/env python3
"""Extract connected 16-input subnetworks, preserving boundary indices and paths."""
import argparse
from array import array
import collections
import hashlib
import json
from pathlib import Path
import random

import benchmark as b


def equation(record):
    return ''.join(map(chr, record['equation_codepoints'])) if 'equation_codepoints' in record else record['equation']


def neighborhood(terms, maximum, seed):
    """Deterministic BFS prefix; each added tensor shares an index with the set."""
    incidence = collections.defaultdict(list)
    for i, term in enumerate(terms):
        for axis in dict.fromkeys(term):
            incidence[axis].append(i)
    rng = random.Random(seed)
    candidates = rng.sample(range(len(terms)), min(len(terms), 64))
    best = []
    for start in candidates:
        selected, seen, seen_axes = [start], {start}, set()
        cursor = 0
        while cursor < len(selected) and len(selected) < maximum:
            for axis in dict.fromkeys(terms[selected[cursor]]):
                if axis in seen_axes:
                    continue
                seen_axes.add(axis)
                for other in incidence[axis]:
                    if other not in seen:
                        seen.add(other); selected.append(other)
                        if len(selected) == maximum:
                            return selected
            cursor += 1
        if len(selected) > len(best):
            best = selected
    return best


class LiveSlots:
    """Order-statistic slots: official linear paths → SSA in O(n log n)."""
    def __init__(self, n):
        self.size = 2 * n - 1
        self.bits = array('i', [0]) * (self.size + 1)
        # Initially slots 0..n-1 are live; later intermediates append in order.
        for i in range(1, self.size + 1):
            self.bits[i] = max(0, min(i, n) - (i - (i & -i)))
        self.top = 1 << (self.size.bit_length() - 1)

    def add(self, slot, value):
        i = slot + 1
        while i <= self.size:
            self.bits[i] += value
            i += i & -i

    def kth(self, index):
        rank, pos, step = index + 1, 0, self.top
        while step:
            nxt = pos + step
            if nxt <= self.size and self.bits[nxt] < rank:
                rank -= self.bits[nxt]; pos = nxt
            step >>= 1
        return pos


def ssa_path(path, n):
    slots = LiveSlots(n)
    left, right = array('i'), array('i')
    for step, (li, ri) in enumerate(path):
        l, r = slots.kth(li), slots.kth(ri)
        left.append(l); right.append(r)
        slots.add(l, -1); slots.add(r, -1); slots.add(n + step, 1)
    return left, right


def restrict_path(ssa, n, selected):
    selected = sorted(selected)
    states = {original: local for local, original in enumerate(selected)}
    compact, k = [], len(selected)
    for step, (l, r) in enumerate(zip(*ssa)):
        a, c = states.pop(l, None), states.pop(r, None)
        if a is not None and c is not None:
            compact.append((a, c))
            states[n + step] = k + len(compact) - 1
        elif a is not None or c is not None:
            states[n + step] = a if a is not None else c
    if len(compact) != k - 1:
        raise ValueError('restricted tree lost leaves')
    live, linear = list(range(k)), []
    for i, (a, c) in enumerate(compact):
        li, ri = live.index(a), live.index(c)
        linear.append([li, ri])
        live = [v for j, v in enumerate(live) if j not in (li, ri)] + [k + i]
    return linear


def local_record(parent, terms, output, selected, size, paths, seed):
    selected = sorted(selected)
    selected_set = set(selected)
    inside_counts = collections.Counter(a for i in selected for a in terms[i])
    all_counts = collections.Counter(''.join(terms))
    first = list(dict.fromkeys(a for i in selected for a in terms[i]))
    boundary = set(output) | {a for a, count in inside_counts.items() if all_counts[a] > count}
    # Original output order first, followed by extra cut-boundary axes.
    local_output = [a for a in output if a in inside_counts]
    local_output += [a for a in first if a in boundary and a not in local_output]
    expr = ','.join(terms[i] for i in selected) + '->' + ''.join(local_output)
    result = dict(id=f'{parent["id"]}__local_{size}',
                  category=parent.get('category', b.CATEGORIES.get(parent['id'].split('_')[0], 'Other')),
                  shapes=[parent['shapes'][i] for i in selected], dtype=parent['dtype'], paths=paths,
                  provenance=dict(parent_id=parent['id'], parent_file_sha256=parent.get('provenance', {}).get('sha256'),
                                  input_tensor_ids=selected, boundary_codepoints=list(map(ord, local_output)),
                                  selection_seed=seed, size=size,
                                  path_origin='both supplied linear paths restricted to selected leaves',
                                  dataset_doi='10.5281/zenodo.11477304', license='CC-BY-4.0'))
    if any(0xD800 <= ord(c) <= 0xDFFF for c in expr):
        result['equation_codepoints'] = list(map(ord, expr))
    else:
        result['equation'] = expr
    b.validate(result)
    return result


def extract(parent, sizes, seed):
    terms, output = b.validate(parent)
    # Stable per-parent seed, independent of Python hash randomization/order.
    case_seed = int.from_bytes(hashlib.sha256(f'{seed}:{parent["id"]}'.encode()).digest()[:8], 'big')
    neighborhood_ids = neighborhood(terms, max(sizes), case_seed)
    selections = {size: sorted(neighborhood_ids[:size]) for size in sizes if len(neighborhood_ids) >= size}
    restricted = {size: {} for size in selections}
    for name, path in parent['paths'].items():
        ssa = ssa_path(path, len(terms))
        for size, selected in selections.items():
            restricted[size][name] = restrict_path(ssa, len(terms), selected)
    cases = [local_record(parent, terms, output, selections[size], size, restricted[size], case_seed)
             for size in sorted(selections)]
    omitted = [dict(parent_id=parent['id'], size=size, reason='no_connected_neighborhood_found',
                    maximum_found=len(neighborhood_ids), seed_candidates=min(len(terms), 64))
               for size in sizes if size not in selections]
    return cases, omitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    parents = json.loads(args.input.read_text())
    records, omitted = [], []
    for i, parent in enumerate(parents):
        cases, missing = extract(parent, [16], args.seed)
        records.extend(cases); omitted.extend(missing)
        print(f'{i+1}/{len(parents)} {parent["id"]}: {len(cases)} cases', flush=True)
    b.save(args.out, records)
    b.save(args.out.with_suffix('.manifest.json'), dict(parent_count=len(parents), case_count=len(records),
           sizes=[16], seed=args.seed, input_sha256=b.digest(args.input),
           generator_sha256=b.digest(__file__), sampling='up to 64 seeded starting tensors; first largest BFS neighborhood',
           omitted=omitted, boundary_policy='retain every index in original output or outside selected tensor set',
           dtype_and_extents='unchanged', target_policy='restrict both original paths; no synthetic alternative'))


if __name__ == '__main__':
    main()
