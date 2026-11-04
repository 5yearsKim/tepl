# XLA Axis-Only Evaluation

The active attribute-inference evaluation uses one canonical migrated rule per
in-scope XLA case. Every eligible original RHS `axis` field is `auto`.
Extent, coordinate, and general fields remain supplied. The runner does not use
whole-descriptor inference, partial variants, an explicit-baseline comparison,
or legacy non-axis synthesis.

- [Current report](reports/attribute_synthesis.md): measured scope, inference,
  verification-only outcomes, and unresolved cases.
- [Semantic and theorem breakdown](reports/semantic_theorem_breakdown.md):
  broader semantic inventory, theorem candidates, and untested cases.
- [Expanded semantic run](reports/expanded_semantics.md): separate results for
  the eleven additional slice and restricted convolution cases.
- [Reciprocal normalization](reports/reciprocal_normalization.md): a guarded
  compiler improvement and a separate nonzero-literal companion case.
- [Broadcast–transpose update](reports/broadcast_transpose_update.md): a
  separate rerun with a general-rank metadata construction.
- [Unit-view cost check](reports/unit_view_cost.md): indicative timings for unchanged successful tasks.
- [Token-count experiment](../specification_size/README.md): the main five-section XLA/explicit/inferred comparison over 30 mapped rules, plus all 31 successful TEPL cases.
- [Case table](reports/synthesis_cases.md): one outcome per original case,
  including exclusions and the separate negative control.
- [Operation-family coverage](reports/operation_families.md): overlapping family
  memberships across the 209 rewrite cases, with separate transpose and reshape
  families, representative operations, and counts with percentages.
- [Protocol](reports/axis_only_scope.md): migration, normalization, admission,
  and success criteria.
- [TEPL project](../tepl/README.md): source fixtures and canonical migrated rules.

The checked-in reports and JSON are frozen measurements. Integrity checks,
including the snapshot-dependent tests in `tests/test_synthesis.py`, require
the recorded source/tool/compiler revision. The sequence below creates a new
classification and run for a changed checkout. Documentation relocation alone
can change recorded proof/source hashes; keep old outcomes and hashes intact
rather than rewriting them to match the current tree.

## Reproduce

From the repository root:

```sh
bazel build //:tepl
python3 labs/xla_rewrite/coverage/scripts/synthesis_cases.py
python3 labs/xla_rewrite/coverage/scripts/run_synthesis.py --jobs 4 --timeout 30
python3 labs/xla_rewrite/coverage/scripts/synthesis_report.py
python3 labs/xla_rewrite/coverage/scripts/synthesis_report.py --check
python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_synthesis.py'
```

Classification and migration run before measurement. The classifier replaces
only its own generated files in `../tepl/rules/axis/`. The runner checks every
in-scope canonical task once, including cases without axis targets as explicit
verification-only tasks. It records original/task hashes, requested fields,
normalization metadata, compiler provenance, search paths, timings, proof
output, and compile/JSON validation evidence. `--compiler PATH`,
`--timeout SECONDS`, and `--jobs N` configure the runner. Do not change measured
sources or scripts while it runs; regenerate results after changing inputs.

The public grouping has one out-of-scope category. Per-case exclusion reasons
remain in the classification for auditing. Theorem admission is independent
of measured success. Structural candidates must still satisfy the stated
positive-extent, supplied-program, arithmetic-domain, and tree assumptions.
A failed proof is unresolved, not an infeasibility result. The general complete
mathematical procedure is not fully implemented by the production search.

## Files

| Location | Purpose |
|---|---|
| [../tepl/rules/axis/](../tepl/rules/axis/) | Canonical field-level axis tasks; the only active inference inputs |
| [../tepl/dialects/axis_views.tepl](../tepl/dialects/axis_views.tepl) | Type-preserving singleton-axis removal adapter |
| [data/synthesis_classification.json](data/synthesis_classification.json) | Frozen pre-run scope, roles, targets, normalization, and theorem candidates |
| [results/synthesis.json](results/synthesis.json) | Current axis-only outcomes and evidence |
| [scripts/axis_fields.py](scripts/axis_fields.py) | Role-directed migration and conservative unit-view normalization |
| [scripts/run_synthesis.py](scripts/run_synthesis.py) | One-task runner; no legacy inference modes |
| [scripts/synthesis_report.py](scripts/synthesis_report.py) | Integrity validation and report generation |
| [tests/test_synthesis.py](tests/test_synthesis.py) | Migration, domain, and success-accounting regressions |
| [data/operation_family_mapping.json](data/operation_family_mapping.json) | Complete operation-to-family mapping for corpus reporting |
| [data/case_families.json](data/case_families.json) | Generated per-case memberships and family counts, excluding the control from summaries |
| [scripts/operation_families.py](scripts/operation_families.py) | Generate or check the family classification and report without rerunning measurements |

## Corpus and structural compilation

The original [semantic](../tepl/rules/semantic/) and
[structural](../tepl/rules/structural/) rule files are retained as corpus fixtures
and provenance, not paired baseline variants in the active inference run.
[data/verification_cases.json](data/verification_cases.json) supplies original
case IDs, graphs, conditions, and catalog mappings. The migrated task removes
unused RHS axis-construction formulas and keeps required supplied-field helpers.

The [structural compilation report](reports/compilation.md) measures the
independent question of whether the full corpus can be represented and compiled,
without asserting value equivalence. Its designated target set is recorded in
[data/compilation_cases.json](data/compilation_cases.json). Reproduce it with:

```sh
python3 labs/xla_rewrite/coverage/scripts/expression_coverage.py --run
python3 labs/xla_rewrite/coverage/scripts/expression_coverage.py --check
```

The pinned [catalog](../catalog/README.md), source corpus, and dense rewrite
execution tests remain available for provenance and independent checks.
