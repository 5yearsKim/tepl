#!/usr/bin/env python3
"""Count any HLO workload once and join a version-pinned semantic review registry."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

LAB = Path(__file__).resolve().parent
STATUSES = ('direct', 'unit_view_candidate', 'conditional', 'outside', 'unreviewed')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def validate_counter(registry, metadata):
    for expected, actual in (
        ('xla_revision', 'xla_revision'),
        ('original_source_sha256', 'original_algebraic_simplifier_sha256'),
        ('instrumented_source_sha256', 'instrumented_algebraic_simplifier_sha256'),
    ):
        if registry[expected] != metadata.get(actual):
            raise ValueError(f'Counter/registry mismatch: {actual}. Rule IDs and normalization must be reviewed for this source version.')


def profile_mismatches(registry, registry_path):
    problems = []
    for name, entry in registry['semantic_profiles'].items():
        path = (Path(registry_path).parent / entry['path']).resolve()
        if not path.exists() or sha(path) != entry['sha256']:
            problems.append(name)
    return problems


def analyze(raw, registry, metadata, profiles_valid=True):
    validate_counter(registry, metadata)
    raw_rows = raw['rules']
    if any(type(r['applications']) is not int or r['applications'] <= 0 for r in raw_rows):
        raise ValueError('Expected positive integer application counts.')
    lines = [r['original_line'] for r in raw_rows]
    if len(set(lines)) != len(lines):
        raise ValueError('Duplicate source sites in raw data; aggregate before analysis.')
    if sum(r['applications'] for r in raw_rows) != raw['total_applications']:
        raise ValueError('Raw count total does not match per-site counts.')
    if len(raw_rows) != raw['distinct_rule_sites']:
        raise ValueError('Raw distinct-site count does not match rows.')
    by_line = {r['original_line']: r for r in raw_rows}
    excluded, normalization = set(), []
    for rule in registry['normalization']:
        a, b = rule['exclude_line'], rule['paired_line']
        count = by_line.get(a, {}).get('applications', 0)
        if count != by_line.get(b, {}).get('applications', 0):
            raise ValueError(f'Bookkeeping/replacement pairing differs at {a}/{b}; inspect before reporting applications.')
        excluded.add(a)
        if count:
            normalization.append(dict(line=a, notifications=count, reason=rule['reason']))
    rows = []
    for event in raw_rows:
        line = event['original_line']
        if line in excluded:
            continue
        rid = f'AS-{line}'
        review = registry['rules'].get(rid)
        can_reuse = bool(review and profiles_valid and review['function'] == event['function'])
        row = dict(rule_id=rid, function=event['function'], original_line=line,
                   applications=event['applications'], family=review['family'] if review else 'unreviewed',
                   description=review['description'] if review else 'Unreviewed transformation branch')
        if can_reuse:
            row.update({key: review[key] for key in ('semantic_status', 'evidence', 'reason')})
        else:
            reason = ('Semantic profile changed; review is stale.' if not profiles_valid else
                      'Previously unseen source site.' if not review else 'Function identity differs from reviewed entry.')
            row.update(semantic_status='unreviewed', evidence='', reason=reason)
        if row['semantic_status'] not in STATUSES:
            raise ValueError(f'Invalid registry status for {rid}.')
        rows.append(row)
    rows.sort(key=lambda r: (-r['applications'], r['original_line']))
    categories = {s: dict(rule_variants=0, applications=0) for s in STATUSES}
    families = {}
    for row in rows:
        for group in (categories[row['semantic_status']], families.setdefault(row['family'], dict(rule_variants=0, applications=0))):
            group['rule_variants'] += 1
            group['applications'] += row['applications']
    covered = [r for r in rows if r['semantic_status'] in ('direct', 'unit_view_candidate')]
    coverage = dict(rule_variants=len(covered), total_rule_variants=len(rows),
                    associated_native_applications=sum(r['applications'] for r in covered),
                    total_native_applications=sum(r['applications'] for r in rows),
                    unit_view_candidates=categories['unit_view_candidate']['rule_variants'],
                    definition='Direct semantic representations plus restricted unit-axis normalization candidates; not measured TEPL acceptance or delegation')
    summary = dict(semantic_model_coverage=coverage, scope=raw['scope'], rule_identity=raw['rule_identity'],
                   model_compilations=raw['model_compilations'], pass_invocations=raw['pass_invocations'],
                   distinct_rule_variants=len(rows), total_applications=sum(r['applications'] for r in rows),
                   semantic_categories=categories, families=families, normalization=normalization,
                   semantic_assessment=registry['support_meaning'],
                   semantic_weight_meaning='Application counts are native activity associated with reviewed source variants, not verified supported TEPL applications',
                   tepl_verification_measured=False, tepl_delegation_measured=False, actual_application_eligibility_measured=False)
    return rows, summary


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def original_source(metadata, registry):
    for path in [Path(metadata.get('original_source', '/nonexistent')), LAB / 'counter/algebraic_simplifier.original.cc']:
        if path.is_file() and sha(path) == registry['original_source_sha256']:
            return path
    return None


def write_report(out, raw, registry, metadata, registry_path, workload):
    mismatches = profile_mismatches(registry, registry_path)
    rows, summary = analyze(raw, registry, metadata, profiles_valid=not mismatches)
    summary.update(workload=workload, semantic_profile_mismatches=mismatches)
    save_json(out / 'summary.json', summary)
    fields = ['rule_id', 'family', 'description', 'function', 'original_line', 'applications', 'semantic_status', 'evidence', 'reason']
    write_csv(out / 'rules.csv', rows, fields)
    unknown = [r for r in rows if r['semantic_status'] == 'unreviewed']
    write_csv(out / 'review-needed.csv', unknown, fields)
    write_csv(out / 'conditional.csv', [r for r in rows if r['semantic_status'] == 'conditional'], fields)
    family_rows = [dict(family=k, **v) for k, v in sorted(summary['families'].items())]
    write_csv(out / 'families.csv', family_rows, ['family', 'rule_variants', 'applications'])
    source = original_source(metadata, registry)
    if source and unknown:
        snippets = out / 'review-context'
        snippets.mkdir()
        source_lines = source.read_text().splitlines()
        for row in unknown:
            line = row['original_line']
            snippet = '\n'.join(f'{i + 1}: {source_lines[i]}' for i in range(max(0, line - 16), min(len(source_lines), line + 16)))
            (snippets / (row['rule_id'] + '.txt')).write_text(snippet + '\n')
    (out / 'README.md').write_text(
        f"# {workload}\n\nScope: {summary['scope']}.\n\n"
        f"Distinct rewrite variants: **{summary['distinct_rule_variants']}**. "
        f"Applications: **{summary['total_applications']}**.\n\n"
        f"Semantic model coverage (direct + unit-view candidates): **{summary['semantic_model_coverage']['rule_variants']}/{summary['distinct_rule_variants']}**. "
        "This includes normalization candidates, not measured TEPL acceptance.\n\n"
        "Known semantic classifications are reused from the reviewed registry. "
        "Unreviewed sites are not treated as supported or unsupported. "
        "Unit-view and conditional entries remain candidates, not accepted TEPL rewrites. "
        "Per-application types, guards and reducers are not checked by this report. "
        "No TEPL proof or delegation is measured.\n\n"
        "See `rules.csv`, `families.csv`, `review-needed.csv`, and `conditional.csv`. "
        "New source sites have context excerpts when the pinned original source is available.\n")
    print(f"{workload}: {summary['distinct_rule_variants']} distinct variants; {summary['total_applications']} applications")
    coverage = summary['semantic_model_coverage']
    print(f"Semantic model coverage (direct + unit-view candidates): {coverage['rule_variants']}/{coverage['total_rule_variants']} variants")
    print('Semantic variants: ' + ', '.join(f"{k}={v['rule_variants']}" for k, v in summary['semantic_categories'].items()))
    print(f"Review queue: {len(unknown)} unseen/stale variants. These are not auto-classified as supported.")
    return summary


def ensure_empty(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory is not empty: {path}. Choose a new directory to preserve previous results.')
    path.mkdir(parents=True, exist_ok=True)


def prepare(args, registry):
    xla = args.xla_root.resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=xla, text=True).strip()
    if revision != registry['xla_revision']:
        raise ValueError('XLA revision differs from the reviewed registry. Remap rule sites before preparing this version.')
    source = xla / 'xla/hlo/transforms/simplifiers/algebraic_simplifier.cc'
    if sha(source) not in (registry['original_source_sha256'], registry['instrumented_source_sha256']):
        raise ValueError('AlgebraicSimplifier contains other changes; preserve/review them before installing the counter.')
    target = xla / 'research/rewrite_count'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('instrument.py', 'driver.cc', 'BUILD.bazel'):
        source_file = LAB / 'counter' / name
        dest = target / name
        if dest.exists() and dest.read_bytes() != source_file.read_bytes():
            raise ValueError(f'Research driver has different contents: {dest}; preserve/review before replacing.')
        if not dest.exists():
            shutil.copy2(source_file, dest)
    archive = target / 'algebraic_simplifier.original.cc'
    if archive.exists() and sha(archive) != registry['original_source_sha256']:
        raise ValueError('Existing original-source archive differs from the reviewed version.')
    if not archive.exists():
        shutil.copy2(LAB / 'counter/algebraic_simplifier.original.cc', archive)
    subprocess.run([sys.executable, str(target / 'instrument.py')], check=True, cwd=xla)
    if sha(source) != registry['instrumented_source_sha256']:
        raise ValueError('Generated instrumentation differs from the registry contract.')
    if not args.build:
        print('Prepared source only; no build or model compilation was run. Use prepare --build for the one-time build.')
        return
    subprocess.run(['bazel', 'build', '--config=joint_shard', '-c', 'opt', '--jobs=4', '//research/rewrite_count:rewrite_count'], cwd=xla, check=True)
    binary = xla / 'bazel-bin/research/rewrite_count/rewrite_count'
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    controls = subprocess.run([str(binary), '--self-test'], env=env, capture_output=True, text=True, check=True)
    result = json.loads(controls.stdout)
    if result != dict(controls=2, repeated_rule_applications=2, distinct_rule_sites=1, no_match_applications=0):
        raise ValueError('Counter controls did not match the contract.')
    save_json(args.counter_metadata, dict(xla_revision=revision, driver_binary=str(binary), driver_sha256=sha(binary),
              original_source=str(target / 'algebraic_simplifier.original.cc'),
              original_algebraic_simplifier_sha256=registry['original_source_sha256'],
              instrumented_algebraic_simplifier_sha256=sha(source), controls_passed=True))
    print('Counter built and controls passed; no model was compiled.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['count', 'replay', 'prepare'])
    parser.add_argument('--registry', type=Path, default=LAB / 'support-registry.json')
    parser.add_argument('--counter-metadata', type=Path, default=LAB / 'counter-provenance.json')
    parser.add_argument('--hlo', type=Path)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--provenance', type=Path, help='Counter metadata accompanying saved raw data (defaults to adjacent provenance.json).')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--driver', type=Path)
    parser.add_argument('--xla-root', type=Path, default=Path(os.environ.get('XLA_ROOT', str(LAB.parents[1].parent / 'xla'))))
    parser.add_argument('--build', action='store_true', help='Build the counter once; no model run.')
    args = parser.parse_args(argv)
    registry = read_json(args.registry)
    if args.action == 'prepare':
        prepare(args, registry)
        return
    if not args.out:
        raise ValueError('--out is required.')
    if args.action == 'replay':
        if not args.raw:
            raise ValueError('replay requires --raw.')
        metadata_path = args.provenance or next((p for p in [args.raw.parent / 'counter-provenance.json', args.raw.parent / 'provenance.json'] if p.exists()), args.raw.parent / 'provenance.json')
        metadata = read_json(metadata_path)
        raw = read_json(args.raw)
        validate_counter(registry, metadata)
        ensure_empty(args.out)
        shutil.copy2(args.raw, args.out / 'raw-results.json')
    else:
        if not args.hlo or not args.hlo.is_file():
            raise ValueError('count requires an existing --hlo file.')
        metadata = read_json(args.counter_metadata)
        validate_counter(registry, metadata)
        binary = args.driver or Path(metadata['driver_binary'])
        if not binary.is_file() or sha(binary) != metadata['driver_sha256']:
            raise ValueError('Counter binary is missing or changed. Run prepare --build once; no automatic build is attempted.')
        if not metadata.get('controls_passed'):
            raise ValueError('Counter controls have not been validated.')
        ensure_empty(args.out)
        shutil.copy2(args.hlo, args.out / 'input.hlo')
        save_json(args.out / 'plan.json', dict(input_sha256=sha(args.hlo), model_compilations=1, numerical_executions=0,
                  scope='All AlgebraicSimplifier invocations in one normal XLA CPU compilation', driver_sha256=sha(binary)))
        env = dict(os.environ)
        env['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        result = subprocess.run([str(binary), str(args.out / 'input.hlo'), str(args.out / 'hlo')], env=env, capture_output=True, text=True)
        (args.out / 'driver.stdout').write_text(result.stdout)
        (args.out / 'driver.stderr').write_text(result.stderr)
        if result.returncode:
            raise ValueError(f'XLA compilation failed ({result.returncode}); logs saved in {args.out}.')
        raw = json.loads(result.stdout)
        save_json(args.out / 'raw-results.json', raw)
    save_json(args.out / 'counter-provenance.json', metadata)
    frozen_registry = json.loads(json.dumps(registry))
    (args.out / 'semantic-profile').mkdir()
    for name, entry in registry['semantic_profiles'].items():
        profile = (args.registry.parent / entry['path']).resolve()
        if profile.is_file():
            shutil.copy2(profile, args.out / 'semantic-profile' / name)
        frozen_registry['semantic_profiles'][name]['path'] = 'semantic-profile/' + name
    save_json(args.out / 'support-registry.json', frozen_registry)
    write_report(args.out, raw, registry, metadata, args.registry, args.name or (args.hlo.stem if args.hlo else args.raw.parent.name))
    files = {str(p.relative_to(args.out)): sha(p) for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    save_json(args.out / 'manifest.json', files)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
