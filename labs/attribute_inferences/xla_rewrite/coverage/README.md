# XLA Axis-Only Evaluation

[Current core classification](reports/core_classification.md) records 21 forms
in the main-theorem scope. Measured native dispatch uses constructive metadata
DAGs for 18 forms and specialized schemas for three existing reversal forms.
[Results and hashes](results/core_classification.json) identify the measured
compiler, inputs, and implementation paths.

## Reproduce current core coverage

From the repository root:

```sh
bazel build //:tepl
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/core_classification.py \
  --paper /path/to/TEPL-paper
bazel test //tests:insertion_test --test_output=all
```

The runner reads the paper audit and its rule inputs without modifying them.
It compiles and portable-validates the 21 main-theorem forms using `--core`.
The insertion regression checks the eight generic reductions, including the
recorded normalized unit insertion. [Implementation scope](../../../../docs/references/restricted-axis-insertion.md)
describes the normalization, constructive builder, assumptions, and limits.

## Corpus and full-corpus measurement tools

The pinned [catalog](../catalog/README.md), original
[semantic](../tepl/rules/semantic/) and [structural](../tepl/rules/structural/)
rules, and dense execution tests remain available as inputs and provenance.
[data/verification_cases.json](data/verification_cases.json) supplies original
case IDs, graphs, conditions, and catalog mappings.

Full-corpus aggregate outputs are generated on demand. The classifier creates
one canonical migrated rule per in-scope case in `../tepl/rules/axis/`. Every
eligible original RHS axis field is `auto`; extent, coordinate, and other
fields stay supplied. Original source conditions are retained. Regenerate
classification before running measurements against a changed checkout:

```sh
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/synthesis_cases.py
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/run_synthesis.py --jobs 4 --timeout 30
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/synthesis_report.py
```

These commands generate fresh result JSON and aggregate reports. The
[axis-only protocol](reports/axis_only_scope.md) defines migration and acceptance.
The runner records source/tool/compiler hashes, requested fields, diagnostics,
proof evidence, timing, and executable validation. A failed proof is unresolved,
not an infeasibility certificate. Measurement validator tests use synthetic
fixtures and do not depend on a checked-in aggregate snapshot:

```sh
python3 -m unittest discover -s labs/attribute_inferences/xla_rewrite/coverage/tests -p 'test_synthesis.py'
python3 -m unittest discover -s labs/attribute_inferences/xla_rewrite/coverage/tests -p 'test_engineering_dependencies.py'
```

## Other reports

- [Semantic support inventory](reports/semantic_support.md): modeled operation
  declarations and task categories.
- [Expanded semantic run](reports/expanded_semantics.md): the separate slice and
  restricted convolution experiment.
- [Reciprocal normalization](reports/reciprocal_normalization.md): the nonzero
  scalar companion case.
- [Broadcast–transpose companion](reports/broadcast_transpose_update.md): the
  supplied general-rank metadata construction and its separate experiment.
- [Unit-view cost check](reports/unit_view_cost.md): paired proof-cost observations.
- [Operation-family coverage](reports/operation_families.md): overlapping family
  memberships across the 209 corpus cases.
- [TEPL project](../tepl/README.md): dialects, fixtures, and migrated rules.

## Files

| Location | Purpose |
|---|---|
| [reports/core_classification.md](reports/core_classification.md) | Current theorem scope and native dispatch |
| [results/core_classification.json](results/core_classification.json) | Current core compilation and portable-validation evidence |
| [scripts/core_classification.py](scripts/core_classification.py) | Reproduce current core coverage |
| [scripts/restricted_insertion.py](scripts/restricted_insertion.py) | Verify the eight recorded insertion inputs and their generic dispatch |
| [../tepl/rules/axis/](../tepl/rules/axis/) | Canonical field-level axis tasks |
| [data/synthesis_classification.json](data/synthesis_classification.json) | Task roles, normalization, and source metadata; regenerate before a new full-corpus run |
| [scripts/axis_fields.py](scripts/axis_fields.py) | Role-directed migration and unit-view normalization |
| [scripts/run_synthesis.py](scripts/run_synthesis.py) | Full-corpus measurement runner |
| [scripts/synthesis_report.py](scripts/synthesis_report.py) | Validate and render a fresh full-corpus run |
| [tests/test_synthesis.py](tests/test_synthesis.py) | Migration, domain, and success-accounting regressions |
| [data/operation_family_mapping.json](data/operation_family_mapping.json) | Operation-to-family mapping |
| [data/case_families.json](data/case_families.json) | Per-case memberships and family counts |

## Structural compilation

The [structural compilation report](reports/compilation.md) measures whether
the corpus can be represented and compiled, without asserting value equivalence.
Its target set is in [data/compilation_cases.json](data/compilation_cases.json).
Reproduce it with:

```sh
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/expression_coverage.py --run
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/expression_coverage.py --check
```
