# General reasoning fixes: XLA attribute-inference recheck

2026-10-20T04:04:05.287547+00:00

The compiler now propagates uniform-vector values through arithmetic and selections, eliminates length-compatible unit scaling/zero offsets, and normalizes zero-cardinality coordinate groups before preserving registered views. No operation names or benchmark names participate in these algorithms.

Uses the original frozen 42-case manifest and identical generated explicit/full-auto inputs: 38 primary axis/extent cases and four supplementary scalar-setting cases. Original rules, guards and dialect semantics are unchanged. Assistance is not rerun in this comparison. The original results/corpus.json is preserved, with its hash recorded in the new evidence.

| Measurement | Before | After |
|---|---:|---:|
| Primary explicit proofs | 20/38 | 20/38 |
| Primary full-auto proofs | 17/38 | 19/38 |
| Full auto within the original explicit-proved cohort | 15/20 | 17/20 |
| Full-auto proofs excluding detected identity witnesses | 16/38 | 18/38 |

## Changed outcomes

| Case | Variant | Before | After |
|---|---|---|---|
| scalar_broadcast_transpose | full_auto | not_proved | proved |
| slice_compose | full_auto | not_proved | proved |

Previously proved explicit/full-auto variants lost: **0**. Incorrect permutation-order control: **not_proved**.

## All explicit/full-auto outcomes

| Case | Cohort | Explicit | Full auto |
|---|---|---|---|
| transpose_compose | axis_extent | proved | proved |
| broadcast_add | axis_extent | proved | proved |
| broadcast_multiply | axis_extent | proved | proved |
| scalar_broadcast_reverse | axis_extent | proved | proved |
| scalar_broadcast_transpose | axis_extent | proved | proved |
| reduce_negate | axis_extent | proved | proved |
| reduce_compose | axis_extent | not_proved | not_proved |
| reduce_transpose | axis_extent | proved | proved |
| reverse_compose_disjoint | axis_extent | proved | not_proved |
| reverse_add_scalar | axis_extent | not_proved | not_proved |
| slice_compose | axis_extent | proved | proved |
| slice_reverse | axis_extent | not_proved | not_proved |
| dot_associate | axis_extent | proved | proved |
| dot_transpose_output | axis_extent | proved | proved |
| dot_move_reverse | axis_extent | not_proved | not_proved |
| reduce_compose_leading | axis_extent | not_proved | not_proved |
| slice_compose_zero_starts | axis_extent | proved | not_proved |
| broadcast_compose_append | axis_extent | not_proved | not_proved |
| dot_swap_operands | axis_extent | proved | proved |
| dot_no_contraction | axis_extent | not_proved | not_proved |
| dot_reduce_product | axis_extent | proved | proved |
| dot_reduce_batch | axis_extent | proved | proved |
| expr_reverse_compose_general | axis_extent | proved | not_proved |
| expr_slice_compose_general | axis_extent | proved | proved |
| expr_slice_reverse_general | axis_extent | not_proved | not_proved |
| expr_convolution_scale_filter_bias | axis_extent | not_proved | not_proved |
| catalog_elementwise_log_power | other_attributes | not_proved | not_proved |
| catalog_elementwise_compare_selected_mask | other_attributes | proved | not_proved |
| catalog_elementwise_compare_max | other_attributes | proved | not_proved |
| catalog_elementwise_compare_min | other_attributes | proved | not_proved |
| catalog_broadcast_divide_broadcast | axis_extent | not_proved | not_proved |
| catalog_broadcast_multiply_constant_into_broadcast | axis_extent | not_proved | not_proved |
| catalog_broadcast_transpose | axis_extent | not_proved | proved |
| catalog_broadcast_slice_broadcast_only | axis_extent | not_proved | not_proved |
| catalog_broadcast_slice_broadcast | axis_extent | not_proved | not_proved |
| catalog_indexing_slice_singleton_stride | axis_extent | not_proved | proved |
| catalog_dot_transpose_both_operands | axis_extent | not_proved | not_proved |
| catalog_dot_move_broadcast | axis_extent | proved | proved |
| catalog_dot_slice_output | axis_extent | not_proved | not_proved |
| catalog_dot_reduce_output_free | axis_extent | proved | proved |
| catalog_dot_scalar_multipliers | axis_extent | not_proved | not_proved |
| catalog_reduction_broadcast_original_axes | axis_extent | proved | proved |

Raw stdout, diagnostics, constructors, source/input hashes and compiler hash are in [reasoning_fixes.json](../results/reasoning_fixes.json). An unproved case is not an impossibility result. The detected singleton-stride identity witness is still reported separately from optimization coverage. These are proof-coverage measurements, not timing or complete XLA/IEEE equivalence results.
