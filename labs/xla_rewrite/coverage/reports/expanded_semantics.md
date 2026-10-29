# Expanded semantic candidates: separate run

The eleven cases added by the [semantic inventory](semantic_theorem_breakdown.md)
were checked separately on 2026-10-29 using compiler SHA-256
`96258055f9087add6e7857b0767c69e6c5191a303feddd54d7c011ddbbc57912`
from checkout revision `a65c09e2a1bbf5fed73809c3dc0ee946ddc17b0c`.
The earlier frozen axis-only measurement used a different compiler hash; its
results have not been refreshed. Do not interpret combined counts as a single
run under one compiler revision.

The test uses the existing axis-only migration and acceptance criteria:
original RHS axis fields are `auto`, other fields and source conditions remain
supplied, and no extra guards are added. A success requires proof with
`additional-guard: true`, compilation, and JSON validation. The declared slice
and convolution semantics are restricted models; this is not a claim about
full XLA or strict IEEE semantics.

| Case | Task | Result | Diagnostic or result |
|---|---|---|---|
| `slice_identity` | no RHS attributes | verified | Proof, compilation and validation passed |
| `slice_compose` | supplied non-axis attributes | unresolved | Cannot establish integer bounds for summed slice starts |
| `slice_reverse` | axis inference | unresolved | Derived metadata not established within its attribute type |
| `slice_compose_zero_starts` | supplied non-axis attributes | verified | Proof, compilation and validation passed |
| `expr_slice_compose_general` | supplied non-axis attributes | unresolved | Cannot establish integer bounds for summed slice starts |
| `expr_slice_reverse_general` | axis inference | unresolved | Comprehension outside supported pointwise metadata fragment |
| `expr_convolution_scale_filter_bias` | axis inference | unresolved | Cannot establish an intermediate shape equality |
| `catalog_broadcast_slice_broadcast_only` | axis inference | unresolved | Cannot establish an extent inequality in supplied metadata |
| `catalog_broadcast_slice_broadcast` | axis inference | unresolved | Cannot establish sliced input bounds for attribute construction |
| `catalog_indexing_slice_singleton_stride` | supplied non-axis attributes | unresolved | Comprehension outside supported pointwise metadata fragment |
| `catalog_dot_slice_output` | axis inference | unresolved | Cannot establish positive extents for an attribute |

Thus **0/6 axis tasks** and **2/5 verification-only tasks** passed. All nine
failures have status `unknown`/`not_proved`; they are not proofs that a rewrite
or an axis assignment is impossible. The recorded diagnostic is the first
obligation at which each check stopped; resolving it does not guarantee that
subsequent proof obligations will pass.

[expanded_semantics.json](../results/expanded_semantics.json) includes exact
generated tasks, target fields, check output, successful compile/validation
output, input hashes, timings, and compiler provenance. To repeat the separate
run from the repository root:

```sh
bazel build //:tepl
python3 labs/xla_rewrite/coverage/scripts/run_expanded_semantics.py
```

The runner creates temporary tasks inside the TEPL project for import
resolution, applies the same `measure_case` checks as the frozen evaluation,
and writes only the separate expanded results file. It requires the original
case sources to match the frozen classification's hashes.
