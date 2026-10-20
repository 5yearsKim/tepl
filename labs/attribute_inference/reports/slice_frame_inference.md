# Shared-frame slice inference: XLA attribute-inference recheck

2026-10-20T04:41:14.455830+00:00

A single non-reduced output domain now retains its coordinate frame when that frame overlaps an input. This lets existing affine index normalization match a composed zero-start, unit-stride slice to its inferred replacement. Completed rules are verified in a fresh context.

Uses the original frozen 42-case manifest and identical generated explicit/full-auto inputs: 38 primary axis/extent cases and four supplementary scalar-setting cases. Original rules, guards and dialect semantics are unchanged. Assistance is not rerun in this comparison. The previous results/reverse_inference.json is preserved, with its hash recorded in the new evidence.

| Measurement | Before | After |
|---|---:|---:|
| Primary explicit proofs | 20/38 | 20/38 |
| Primary full-auto proofs | 21/38 | 22/38 |
| Full auto within the original explicit-proved cohort | 19/20 | 20/20 |
| Full-auto proofs excluding detected identity witnesses | 20/38 | 21/38 |

## Changed outcomes

| Case | Variant | Before | After |
|---|---|---|---|
| slice_compose_zero_starts | full_auto | not_proved | proved |

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
| reverse_compose_disjoint | axis_extent | proved | proved |
| reverse_add_scalar | axis_extent | not_proved | not_proved |
| slice_compose | axis_extent | proved | proved |
| slice_reverse | axis_extent | not_proved | not_proved |
| dot_associate | axis_extent | proved | proved |
| dot_transpose_output | axis_extent | proved | proved |
| dot_move_reverse | axis_extent | not_proved | not_proved |
| reduce_compose_leading | axis_extent | not_proved | not_proved |
| slice_compose_zero_starts | axis_extent | proved | proved |
| broadcast_compose_append | axis_extent | not_proved | not_proved |
| dot_swap_operands | axis_extent | proved | proved |
| dot_no_contraction | axis_extent | not_proved | not_proved |
| dot_reduce_product | axis_extent | proved | proved |
| dot_reduce_batch | axis_extent | proved | proved |
| expr_reverse_compose_general | axis_extent | proved | proved |
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

Raw stdout, diagnostics, constructors, source/input hashes and compiler hash are in [slice_frame_inference.json](../results/slice_frame_inference.json). An unproved case is not an impossibility result. The detected singleton-stride identity witness is still reported separately from optimization coverage. These are proof-coverage measurements, not timing or complete XLA/IEEE equivalence results.
