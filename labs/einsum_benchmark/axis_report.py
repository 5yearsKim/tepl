#!/usr/bin/env python3
"""Primary single-size coverage report with counts of actually inferred RHS axes."""
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics

import benchmark as b


def read_json(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as f:
        return json.load(f)


def features(record):
    terms,_=b.validate(record)
    incidence=collections.Counter(a for term in terms for a in set(term))
    return dict(input_ports=sum(map(len,terms)), logical_indices=len(incidence),
                indices_shared_by_3_plus_inputs=sum(count>=3 for count in incidence.values()),
                max_input_rank=max(map(len,terms)),
                mean_input_rank=statistics.mean(map(len,terms)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=b.HERE/'results/local16')
    parser.add_argument('--input',type=Path,default=b.HERE/'fixtures/local16.json')
    parser.add_argument('--out',type=Path,default=b.HERE/'reports/local16')
    args=parser.parse_args()
    rows=read_json(args.run/'results.json')
    inputs=read_json(args.input)
    by_id={r['id']:r for r in inputs}
    if {r['id'] for r in rows} != set(by_id) or any(r['tensors']!=16 for r in rows):
        raise ValueError('expected a complete 16-input-only run')
    provenance=read_json(args.run/'provenance.json')
    if provenance['input_sha256']!=b.digest(args.input):
        raise ValueError('input digest mismatch')
    measured=[]
    for p in sorted(args.run.glob('*/result.json')):
        row=read_json(p);record=by_id[row['id']]
        for d in row['directions']:
            if d['status']!='verified':
                continue
            program=read_json(p.parent/d['direction']/'rewritten.json')
            tree=b.compiled_graph(program,'before')
            b.verify(record,tree)
            metrics=b.axis_inference_metrics(tree)
            if metrics!=d['axis_inference']:
                raise ValueError('recorded/recomputed axis metric mismatch')
            depths=[]
            for node in tree['nodes']:
                depths.append(0 if node['op']=='input' else 1+max(depths[j] for j in node['inputs']))
            measured.append(dict(id=row['id'],category=row['category'],direction=d['direction'],
                                 rhs_depth=depths[tree['root']],**metrics,**features(record)))
    def stats(group):
        ids={r['id'] for r in group}
        ms=[m for m in measured if m['id'] in ids]
        ds=[d for r in group for d in r['directions']]
        sums={key:sum(m[key] for m in ms) for key in ['axis_fields','nonempty_axis_fields','empty_axis_fields',
               'axis_positions','contracting_positions','batching_positions','reduction_positions',
               'permutation_positions','rhs_operations','rhs_contractions']}
        values=sorted(m['axis_positions'] for m in ms)
        return dict(cases=len(group),represented=sum(r['representation']=='ok' for r in group),
                    verified=sum(d['status']=='verified' for d in ds),
                    nonidentical=sum(d['status']!='identical_structure' for d in ds),
                    identical=sum(d['status']=='identical_structure' for d in ds),
                    extent_excluded=sum(d['status']=='outside_core_extent_domain' for d in ds),
                    inference_status=dict(collections.Counter(d['status'] for d in ds)),
                    representation_reasons=dict(collections.Counter(r.get('reason','none') for r in group if r['representation']!='ok')),
                    positions_per_rule_median=statistics.median(values) if values else None,
                    positions_per_rule_min=min(values) if values else None,
                    positions_per_rule_max=max(values) if values else None,
                    positions_per_rule_p95=values[math.ceil(.95*len(values))-1] if values else None,
                    **sums)
    categories={category:stats([r for r in rows if r['category']==category])
                for category in b.CATEGORIES.values() if any(r['category']==category for r in rows)}
    total=stats(rows)
    assert len(measured)==total['verified']
    assert total['axis_positions']==sum(total[k] for k in ['contracting_positions','batching_positions','reduction_positions','permutation_positions'])
    summary=dict(size=16,guard_policy=provenance['guard_policy'],total=total,by_category=categories,
                 metric_definition=dict(axis_fields='RHS axis-list fields completed, including empty lists',
                    axis_positions='integer entries in those lists, counted per occurrence; both sides of a paired axis count',
                    scope='reachable replacement operations of independently verified directed rewrites only',
                    exclusions='no credit for reference witnesses, identical pairs, skipped or failed transformations'),
                 input_sha256=b.digest(args.input),measurement_script_sha256=b.digest(__file__),
                 benchmark_sha256=b.digest(b.__file__),source_provenance_sha256=b.digest(args.run/'provenance.json'))
    text='# 16-input contraction rewrites: coverage and inferred axes\n\n'
    text+=('One connected 16-input subnetwork is selected per original benchmark instance (seed 42). '
           'Original shapes and dtypes and cut-boundary indices are preserved. Both supplied paths are '
           'restricted to the selection. Each path has 15 binary contractions, plus any reductions '
           'and a final permutation.\n\n')
    text+='| Category | Sampled | Represented | Verified / nonidentical tasks | Inferred axis-list fields | Inferred axis positions | Positions/rule median [min, max] |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for category,s in [*categories.items(),('Total',total)]:
        fraction=f'{s["verified"]}/{s["nonidentical"]}' if s['nonidentical'] else '—'
        spread=f'{s["positions_per_rule_median"]:g} [{s["positions_per_rule_min"]}, {s["positions_per_rule_max"]}]' if s['verified'] else '—'
        text+=f'| {category} | {s["cases"]} | {s["represented"]} | {fraction} | {s["axis_fields"]:,} | {s["axis_positions"]:,} | {spread} |\n'
    text+='\n## Metric definitions\n\n'
    text+=('An **inferred axis-list field** is one omitted RHS attribute completed by TEPL: each '
           '`dot_general` has four fields, each `reduce_sum` one, and each `transpose` one. '
           'An empty inferred list counts as one completed field and zero axis positions.\n\n'
           'An **inferred axis position** is one integer entry in such a field. For example, '
           '`lhs_contracting_dimensions=[1,3]` and `rhs_contracting_dimensions=[0,2]` contribute '
           'two fields and four positions. A rank-4 permutation contributes one field and four '
           'positions. The same logical axis appearing in multiple fields/nodes counts multiple '
           'times. These are attribute-entry occurrences, not unique logical axes or tensor elements.\n\n'
           'Counts are measured from the actual rewritten concrete graph after independent '
           'verification, not from the hidden target witness. Each direction is counted separately. '
           'The per-rule CSV also includes maximum/mean input rank, input-port count, distinct logical '
           'indices, indices shared by at least three inputs (including output indices), and RHS depth.\n\n')
    text+='## Inferred positions by role\n\n| Category | Contracting | Batching | Reduction | Permutation |\n|---|---:|---:|---:|---:|\n'
    for category,s in [*categories.items(),('Total',total)]:
        text+=f'| {category} | {s["contracting_positions"]:,} | {s["batching_positions"]:,} | {s["reduction_positions"]:,} | {s["permutation_positions"]:,} |\n'
    text+='\n## Coverage and protocol\n\n'
    text+=(f'{total["represented"]}/{total["cases"]} local cases are represented. '
           f'{total["identical"]} structurally identical directed pairs are excluded. '
           f'{total["extent_excluded"]} nonidentical directions are excluded for singleton extents and remain '
           'in the task denominator. Unsupported dtypes and trace/diagonal handling are adapter '
           'exclusions, not proofs of impossibility in arbitrary TEPL dialects.\n\n'
           f'Guard policy `{summary["guard_policy"]}`, algebraic arithmetic, {provenance["timeout_s"]:g} s '
           f'per subprocess, and {provenance["jobs"]} parallel cases. Rule declarations retain rank, dtype '
           'and shared extents; written guards require extents >=2. Actual LHS descriptors and '
           'generated feasibility determine applicability. No work budget was increased. '
           'Any valid once-applied result at the intentional step limit must still pass independent checks.\n\n'
           'CLI compile/rewrite times are diagnostic process times, not isolated synthesis or '
           'constructor latency. This experiment does not claim full-network coverage, optimal local '
           'paths, numerical execution speedups, or arbitrary-rank generalization.\n\n'
           '[Per-rule axis counts](axis_counts.csv), [category totals](categories.csv), '
           '[summary](summary.json), [raw results](results.json), [run provenance](provenance.json). '
           f'Full generated artifacts: `{args.run}`.\n')
    args.out.mkdir(parents=True,exist_ok=True)
    b.save(args.out/'summary.json',summary)
    (args.out/'report.md').write_text(text)
    with (args.out/'axis_counts.csv').open('w',newline='') as f:
        if measured:
            writer=csv.DictWriter(f,fieldnames=list(measured[0]));writer.writeheader();writer.writerows(measured)
    with (args.out/'categories.csv').open('w',newline='') as f:
        fields=['category','cases','represented','nonidentical','verified','axis_fields','axis_positions',
                'positions_per_rule_median','positions_per_rule_p95','positions_per_rule_min','positions_per_rule_max',
                'contracting_positions','batching_positions','reduction_positions','permutation_positions',
                'nonempty_axis_fields','empty_axis_fields','rhs_operations','rhs_contractions']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for category,s in [*categories.items(),('Total',total)]:writer.writerow(dict(category=category,**s))
    for name in ['results.json','results.csv','provenance.json']:
        shutil.copyfile(args.run/name,args.out/name)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
