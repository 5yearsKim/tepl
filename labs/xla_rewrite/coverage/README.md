# HLO Rewrite Coverage

[Attribute synthesis](reports/attribute_synthesis.md) reports the refreshed
theorem-based experiment: syntax migration, classification before measurement,
and explicit, full-auto, and partial-auto outcomes. The generated
[case table](reports/synthesis_cases.md) accounts for all 209 original cases.

[Compilation](reports/compilation.md) measures rule expressibility and executable
compilation. [Verification](reports/verification.md) reports semantic acceptance,
missing operation semantics and unresolved proofs for the same rule bodies.

The self-contained [TEPL project](../tepl/README.md) lives in `../tepl/`.
Its configuration, dialects, rules and input graphs are kept together. This
directory contains the evaluation data, tools, tests and results.

| Directory | Purpose |
|---|---|
| [../tepl/dialects/](../tepl/dialects/) | Local semantic HLO and matching declarations without value semantics |
| [../tepl/rules/semantic/](../tepl/rules/semantic/) | Explicit rules checked for semantic equivalence, plus a false control |
| [../tepl/rules/structural/](../tepl/rules/structural/) | The same rule bodies using the structural HLO profile for compilation |
| [data/](data/) | Case manifests, manual assessments and the operation-support inventory |
| [baselines/](baselines/) | Frozen reasoning baseline, original manifests and the path-migration record |
| [results/](results/) | Current measurements, diagnostics, proof evidence and dependency hashes |
| [reports/](reports/) | Generated compilation/verification reports and the host-dependency audit |
| [scripts/](scripts/) | Experiment runners, report generation and shared validators |
| [tests/](tests/) | Integrity checks and execution tests using the graphs in `../tepl/graphs/` |

[data/compilation_cases.json](data/compilation_cases.json) and
[data/verification_cases.json](data/verification_cases.json) identify every case,
its catalog schema and its scope. Case file paths are relative to `coverage/`.
Counterpart baseline files differ only in their dialect import and have explicit
RHS attributes. Inference variants live separately in `rules/full_auto/` and
`rules/partial_auto/`; omitted RHS descriptors request inference.

Semantic acceptance requires explicit `status=proved` and
`additional-guard: true`. An ordinary type check does not count as verification.
The deliberately incorrect control must be rejected. Fixed ranks, literal
representations and other adaptations remain explicit in the manifests. The
local HLO dialect is an algebraic evaluation model, not a complete HLO importer
or an exact IEEE/backend equivalence model.

## Reproduce

Run these commands from the repository root:

```sh
bazel build //:tepl
python3 labs/xla_rewrite/coverage/scripts/migrate_syntax.py --check
python3 labs/xla_rewrite/coverage/scripts/synthesis_cases.py
python3 labs/xla_rewrite/coverage/scripts/run_checks.py
python3 labs/xla_rewrite/coverage/scripts/expression_coverage.py --run
python3 labs/xla_rewrite/coverage/scripts/run_synthesis.py
python3 labs/xla_rewrite/coverage/scripts/synthesis_report.py
python3 labs/xla_rewrite/coverage/scripts/synthesis_report.py --check
python3 labs/xla_rewrite/coverage/scripts/summarize_coverage.py
python3 labs/xla_rewrite/coverage/scripts/summarize_coverage.py --check
python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_*.py'
```

Both runners accept `--compiler PATH` and `--timeout SECONDS`. After changing a
measured input, rerun its experiment before regenerating reports. The compilation
runner emits and validates executable JSON, then removes temporary artifacts.
Measurements record compiler provenance and input hashes.

Run individual TEPL cases rather than checking the whole directory: some rules
intentionally lack a proof. Use the TEPL project's `rules/structural/` for
ordinary compilation and `rules/semantic/` for checking with `--semantic-policy=algebraic`.

[test_rule_execution.py](tests/test_rule_execution.py) executes eight
representative structural rewrites and three verified scalar rewrites, comparing
independent dense results. These concrete checks supplement universal semantic
verification.

## Historical inputs

[reasoning_baseline.json](baselines/reasoning_baseline.json) preserves the
pre-extension measurement and original corpus hashes. The directory reorganization
did not edit this file or the rule bodies.
[layout_migration.json](baselines/layout_migration.json) maps historical paths to
the new layout. [pre_layout/](baselines/pre_layout/) preserves the original two
manifests. Validators check their original hashes and require current manifests
to differ only in case file paths.

The syntax migration preserves the changed files separately in
[pre_syntax/](baselines/pre_syntax/). Historical hashes continue to validate
against those snapshots; current measurements use the migrated sources.

The manual semantic assessment in [data/assessments.json](data/assessments.json)
describes an older compiler and is separate from current measurements. Host calls
and host `fn` parameters remain unsupported; see the
[host-dependency audit](reports/host_dependency_audit.md). Current operation
support is recorded in [data/hlo_support.json](data/hlo_support.json).
