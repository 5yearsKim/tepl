# XLA rewrite token-count experiment

This is the main source-size experiment for XLA-derived TEPL rewrites. It
compares **XLA**, **explicit TEPL**, and **axis inferred TEPL** by five sections:
matching (LHS), legality checks (`where`), attribute reasoning (`derive`),
replacement construction (RHS), and other syntax. Whole-rule counts include
complete `where` and `derive` blocks; comments and whitespace cost zero.

The graph uses **30 matched rules** with XLA source mappings. A separate table
covers **all 31 successful synthesis cases** in TEPL; the remaining dot scalar
multiplier case has no corresponding XLA dot implementation in the snapshot.

| Cohort | XLA tokens | Explicit TEPL tokens | Axis inferred TEPL tokens |
|---|---:|---:|---:|
| 30 matched rules | 10,022 | 3,205 | 2,374 |
| All 31 successful TEPL cases | — | 3,396 | 2,520 |

Explicit TEPL spells out axis transformations in `derive`. Axis inferred TEPL
omits inferable axis fields, unused labels, and empty assignments/blocks. Both
use the same normalized graphs and conditions; supplied non-axis attributes,
required labels/helpers, and fixed adapters remain counted. All 31 compact
inferred variants pass proof, compilation, and JSON validation. Three explicit
variants remain unproved and are retained for source counting with diagnostics.

XLA excerpts are annotated by responsibility and shared code is counted once.
Policy/logging and shared library implementations are outside the comparison.
XLA branches can cover broader behavior than the TEPL specializations. These
lexical counts do not measure runtime speed or authoring effort.

## Results and inputs

- [Main report and both tables](reports/section_comparison.md)
- Graph: [PNG](reports/section_comparison.png) / [SVG](reports/section_comparison.svg)
- [LaTeX tables](reports/section_comparison.tex)
- [Measurements and complete source snapshots](results/section_comparison.json)
- [Proof, compilation, and validation evidence](results/section_verification.json)
- [Counting protocol](protocol.md)
- [XLA source annotation ledger](data/xla_spans.json) and [annotated excerpts](reports/section_xla_excerpts.txt)
- [Synthesis coverage and frozen evaluation](../coverage/README.md)

## Reproduce

Run from the repository root. Checking measurements needs only Python's standard
library; graph generation additionally requires `matplotlib`.

```sh
python3 labs/xla_rewrite/specification_size/scripts/section_comparison.py --check
python3 -m unittest discover -s labs/xla_rewrite/specification_size/tests -p 'test_*.py'
```

Regenerate reports and figures from hash-matched sources and existing proof
evidence:

```sh
python3 labs/xla_rewrite/specification_size/scripts/section_comparison.py
```

To refresh proof, compilation, and JSON validation of both representations:

```sh
bazel build //:tepl
python3 labs/xla_rewrite/specification_size/scripts/section_comparison.py --run --jobs 4 --timeout 30
```

`--compiler PATH` selects a compiler (default: `bazel-bin/tepl`). `--run` replaces
this experiment's verification and generated outputs. Without it, the script
requires recorded verification to match the exact measured source corpus.
Frozen synthesis coverage results are inputs and are never rewritten here.

There is one runner (`scripts/section_comparison.py`) and one shared helper
module (`scripts/support.py`); the helper has no separate experiment/report mode.
