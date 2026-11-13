# Current Constructive Core Coverage

The paper audit classifies 21 forms within the main theorem. In this measurement, **13 use generic construction** and **8 use normalization followed by generic construction**. Compilation and portable validation pass for all 21. The three direct reversal forms now use the generic orientation predicate and reversal-axis builder.

## Populations and interpretation

All 38 audited axis forms are measured with the same compiler binary and algebraic policy using the native `--core` backend. The table records current core results separately from historical inference. Unresolved means this backend did not accept the input; it does not invalidate a historical success using another implementation path.

| Current path | All audited forms | Main-theorem forms |
|---|---:|---:|
| Generic construction | 13 | 13 |
| Normalization then generic construction | 8 | 8 |
| Specialized schema | 10 | 0 |
| Unresolved | 7 | 0 |

The historical success population is 31/38. The mathematical theorem classification is 21/38, or 21/31 historically successful forms. Current constructive coverage of that theorem population is 21/21; these are updated artifact measurements, not a reclassification of historical dispatch.

## Case-level native dispatch

| Case | Paper classification | Historical outcome | Current path | Compile / portable validation |
|---|---|---|---|---|
| transpose_compose | main_theorem_form | inferred | Generic construction | Passed |
| broadcast_add | additional_structural_candidate | inferred | Specialized schema | Passed |
| broadcast_multiply | additional_structural_candidate | inferred | Specialized schema | Passed |
| scalar_broadcast_reverse | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| scalar_broadcast_transpose | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| reduce_negate | additional_structural_candidate | inferred | Specialized schema | Passed |
| reduce_compose | main_theorem_form | inferred | Generic construction | Passed |
| reduce_transpose | main_theorem_form | inferred | Generic construction | Passed |
| reverse_compose_disjoint | main_theorem_form | inferred | Generic construction | Passed |
| reverse_add_scalar | additional_structural_candidate | inferred | Specialized schema | Passed |
| dot_associate | main_theorem_form | inferred | Generic construction | Passed |
| dot_transpose_output | main_theorem_form | inferred | Generic construction | Passed |
| dot_move_reverse | main_theorem_form | inferred | Generic construction | Passed |
| reduce_compose_leading | main_theorem_form | inferred | Generic construction | Passed |
| broadcast_compose_append | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| dot_swap_operands | main_theorem_form | inferred | Generic construction | Passed |
| dot_no_contraction | additional_structural_candidate | inferred | Specialized schema | Passed |
| dot_reduce_product | additional_structural_candidate | inferred | Specialized schema | Passed |
| dot_reduce_batch | main_theorem_form | inferred | Generic construction | Passed |
| expr_reverse_compose_general | main_theorem_form | inferred | Generic construction | Passed |
| catalog_broadcast_divide_broadcast | reciprocal_domain_restricted_candidate | not_proved | Unresolved | Unresolved |
| catalog_broadcast_multiply_constant_into_broadcast | additional_structural_candidate | inferred | Specialized schema | Passed |
| catalog_broadcast_transpose | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| catalog_broadcast_inserted_unit_dims | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| catalog_broadcast_drop_unit_dims | additional_structural_candidate | inferred | Specialized schema | Passed |
| catalog_broadcast_transpose_broadcast | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| catalog_indexing_reverse_inserted_unit_dims | additional_structural_candidate | inferred | Specialized schema | Passed |
| catalog_dot_transpose_both_operands | main_theorem_form | inferred | Generic construction | Passed |
| catalog_dot_move_broadcast | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| catalog_dot_reduce_output_free | main_theorem_form | inferred | Generic construction | Passed |
| catalog_dot_scalar_multipliers | additional_structural_candidate | inferred | Specialized schema | Passed |
| catalog_reduction_broadcast_original_axes | main_theorem_reduction | inferred | Normalization then generic construction | Passed |
| slice_reverse | outside_both_stated_contracts | not_proved | Unresolved | Unresolved |
| expr_slice_reverse_general | outside_both_stated_contracts | not_proved | Unresolved | Unresolved |
| expr_convolution_scale_filter_bias | outside_both_stated_contracts | not_proved | Unresolved | Unresolved |
| catalog_broadcast_slice_broadcast_only | outside_both_stated_contracts | not_proved | Unresolved | Unresolved |
| catalog_broadcast_slice_broadcast | outside_both_stated_contracts | not_proved | Unresolved | Unresolved |
| catalog_dot_slice_output | outside_both_stated_contracts | not_proved | Unresolved | Unresolved |

## Original and companion inputs

The 38-form audit, the paper's 31-success population, and the 21-form theorem population use the recorded broadcast–transpose companion for `catalog_broadcast_transpose_broadcast`. Its supplied shape expression differs from the original frozen input. Each input's guards and supplied expressions are preserved during syntax migration. Acceptance of the companion is not acceptance of the frozen input. The latter is measured separately below and is excluded from the 38-form denominator.

- Original frozen `catalog_broadcast_transpose_broadcast_original`: unresolved under the same native core compiler; source `labs/attribute_inferences/xla_rewrite/tepl/rules/axis/catalog_broadcast_transpose_broadcast.tepl`.

## Evidence and limits

[Machine-readable results](../results/core_classification.json) record compiler and source hashes, Git revision, implementation hashes, per-case diagnostics, instruction kinds and DAG operators. The source hashes identify working-tree changes beyond the Git revision. Historical runs and paper files are preserved.

`//tests:insertion_test` independently compares dense tensor results for the eight insertion reductions and pure signed core programs, including reversal parity, contraction transfer, eliminated axes and impossible orientations. Tests are bounded; compilation does not establish full-domain completeness.

Completeness remains conditional on exact-real semantics, input extents at least two, positive supplied fresh extents, independent unknowns, and the restricted insertion boundary. Native `--core` measurements do not establish generated C++/Rust adapter coverage.

```sh
bazel build //:tepl
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/core_classification.py
bazel test //tests:insertion_test //tests:ordered_synthesis_test --test_output=errors
```

Measured UTC: `2026-11-13T16:29:50.029833+00:00`.
