# XLA specification size

This experiment compares matched explicit and inferred RHS axis specifications
with the current compiler. It records where lexical token savings come from,
along with per-rule proof, compilation, and JSON-validation evidence.

- [Responsibility report](reports/responsibility.md): reviewed XLA excerpts,
  original explicit TEPL, compact inferred TEPL, and the five-row contribution table.
- [Responsibility LaTeX table](reports/responsibility.tex): measured table for a paper.
- [Omission protocol](responsibility_protocol.md) and
  [source annotations](reports/xla_excerpts.txt): compaction and XLA boundaries.
- [Matched-field report](reports/specification_size.md): paired totals, component and family
  breakdowns, all attempted cases, and limitations.
- [LaTeX table](reports/specification_size.tex): generated table for a paper.
- [Protocol](protocol.md): cohort, counting, attribution, and XLA excerpt review.
- [Measurements](results/measurement.json): source snapshots, counts, diagnostics,
  hashes, and verification results.
- [XLA excerpt manifest](data/xla_excerpts.json): source contexts and reviewed
  excerpt boundaries. Context ranges do not establish a compression ratio.

Run from the repository root:

```sh
bazel build //:tepl
python3 labs/xla_rewrite/specification_size/scripts/measure.py --run --jobs 4 --timeout 30
python3 labs/xla_rewrite/specification_size/scripts/responsibility.py --run --jobs 4 --timeout 30
python3 labs/xla_rewrite/specification_size/scripts/measure.py --check
python3 labs/xla_rewrite/specification_size/scripts/responsibility.py --check
python3 -m unittest discover -s labs/xla_rewrite/specification_size/tests -p 'test_*.py'
```

Without flags, the script regenerates the Markdown and LaTeX reports from an
unchanged result. `--check` checks input hashes, accounting, and report contents
without a new compiler run. `--compiler PATH` selects a compiler. `--run` replaces
this experiment's current measurements and reports; archive them before a new
published measurement. It does not regenerate the frozen coverage experiment.

The selected 32-case cohort is fixed before observing new outcomes. The primary
size comparison includes only pairs that both verify, compile, and validate;
all failed cases remain visible. Extent and other non-axis fields stay supplied.
Original files are also measured so migration changes are not credited to
inference. The separate [axis-only evaluation](../coverage/README.md) reports
inference coverage over its wider scope.

The historical [full-auto size proxy](../../attribute_inference/reports/corpus.md)
uses a different protocol. Its numbers should not be combined with this run.

The responsibility experiment additionally omits inferable constructor fields,
empty assignments/blocks, and unused node labels. Every representation is checked
with the same compiler; its main table uses a common verified cohort and reviewed
XLA source mappings. Source policy and logging are recorded separately, shared
XLA helpers count once, and broader XLA behavior is identified in the audit notes.
