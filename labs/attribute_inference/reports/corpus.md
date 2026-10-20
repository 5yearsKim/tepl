# Automatic attribute inference: complete XLA cohort

All non-control attribute-bearing cases with definitions for every required operation. Primary cohort: only index[] RHS fields (axis/extent); other fields are supplementary. Run explicit and full-auto variants for every case. On full-auto failure, test every nonempty proper subset of RHS descriptors for operation-level assistance. For single-descriptor cases, also test every nonempty proper subset of available literal-list field values from the original constructor, its spreads, or direct conjunctive guard equalities. At least one field must remain inferred. Do not invent literals, change guards, or combine the two assistance modes. Assisted variants of full-auto successes are prepared but not executed.

## Corpus and primary results

The source evaluation has 209 non-control cases: 98 have no RHS attributes and 111 have attribute work. Of the latter, 69 lack required operation semantics. All remaining 42 are checked here: 38 axis/extent cases in the primary cohort and four scalar-setting cases in a supplementary cohort. All numbers count concrete cases, not entire catalog schemas.

| Measurement | Result |
|---|---:|
| Primary cases attempted | 38 |
| Explicit baselines proved | 20/38 |
| Full auto, all primary cases | 17/38 |
| Full auto, conditioned on explicit proof | 15/20 |
| Additional cases recovered by assistance | 2 |
| Full or assisted, all primary cases | 19/38 |
| Full or assisted, conditioned on explicit proof | 17/20 |
| Inference successes with unproved explicit baseline | 2 |

The primary cohort spans 31 catalog schemas; full-auto successes cover 16 schemas. A concrete specialization does not prove its whole catalog schema.

Assistance is tested only after full-auto failure. Its fraction is additional coverage under the stated bounded policy, not completeness over every possible user hint.

## Primary results by family

| Family | Cases | Explicit proved | Full auto | Additional assisted | Full or assisted |
|---|---:|---:|---:|---:|---:|
| broadcast | 10 | 4 | 4 | 1 | 5 |
| convolution | 1 | 0 | 0 | 0 | 0 |
| dot | 12 | 7 | 7 | 0 | 7 |
| indexing | 9 | 5 | 2 | 1 | 3 |
| reduction | 5 | 3 | 3 | 0 | 3 |
| transpose-reshape | 1 | 1 | 1 | 0 | 1 |

## What was constructed

This syntactic classification describes the printed constructor programs. Constants do not demonstrate arbitrary-rank reconstruction; reuse may contain constant fields; computed programs contain operations beyond field copies and literal lists.

| Full-auto constructor class | Cases |
|---|---:|
| reuse | 6 |
| constant | 8 |
| computed | 3 |

For the **15 pairs where both explicit and full-auto versions verify**, RHS attribute/derive tokens decrease from **341 to 60** (82.4% reduction).
Across all 17 full-auto successes, the written explicit-reference token count decreases from 446 to 66 (85.2%). That broader comparison includes 2 unproved explicit references.
These counts exclude unchanged guards, shared operation semantics and failed inferred cases; they are code-size proxies, not usability results.

## Witness quality

Inference constructs a semantically valid witness, which need not equal the handwritten descriptor or implement its optimization policy.
A conservative unchanged-unary-operation audit identifies **1 identity witness** among the full-auto successes: catalog_indexing_slice_singleton_stride.
Thus 16 primary full-auto successes remain after excluding those detected identity witnesses. This audit is not a general profitability analysis.
In `catalog_indexing_slice_singleton_stride`, inference copies `@s.starts`, `@s.sizes` and `@s.strides`; the intended explicit rule replaces singleton-axis strides with one. The inferred version proves equivalence but leaves the original slice unchanged. It is included in semantic-witness coverage and identified separately from optimization realization.
The other new full-auto success, `catalog_broadcast_transpose`, constructs a replacement transpose. Both written explicit versions remain unproved because their comprehensions exceed the checker's supported metadata fragment.

## Per-case primary results

| Case | Explicit | Full auto | Best successful assistance | Attribute tokens explicit → selected proved variant |
|---|---|---|---|---|
| transpose_compose | proved | proved | not needed | 26 → 3 |
| broadcast_add | proved | proved | not needed | 3 → 3 |
| broadcast_multiply | proved | proved | not needed | 3 → 3 |
| scalar_broadcast_reverse | proved | proved | not needed | 3 → 3 |
| scalar_broadcast_transpose | proved | not_proved | assisted_field__dimensions | 28 → 8 |
| reduce_negate | proved | proved | not needed | 3 → 3 |
| reduce_compose | not_proved | not_proved | none proved | unproved |
| reduce_transpose | proved | proved | not needed | 16 → 3 |
| reverse_compose_disjoint | proved | not_proved | none proved | unproved |
| reverse_add_scalar | not_proved | not_proved | none proved | unproved |
| slice_compose | proved | not_proved | assisted_field__strides | 40 → 9 |
| slice_reverse | not_proved | not_proved | none proved | unproved |
| dot_associate | proved | proved | not needed | 6 → 6 |
| dot_transpose_output | proved | proved | not needed | 32 → 3 |
| dot_move_reverse | not_proved | not_proved | none proved | unproved |
| reduce_compose_leading | not_proved | not_proved | none proved | unproved |
| slice_compose_zero_starts | proved | not_proved | none proved | unproved |
| broadcast_compose_append | not_proved | not_proved | none proved | unproved |
| dot_swap_operands | proved | proved | not needed | 47 → 6 |
| dot_no_contraction | not_proved | not_proved | none proved | unproved |
| dot_reduce_product | proved | proved | not needed | 16 → 3 |
| dot_reduce_batch | proved | proved | not needed | 32 → 3 |
| expr_reverse_compose_general | proved | not_proved | none proved | unproved |
| expr_slice_compose_general | proved | proved | not needed | 52 → 3 |
| expr_slice_reverse_general | not_proved | not_proved | none proved | unproved |
| expr_convolution_scale_filter_bias | not_proved | not_proved | none proved | unproved |
| catalog_broadcast_divide_broadcast | not_proved | not_proved | none proved | unproved |
| catalog_broadcast_multiply_constant_into_broadcast | not_proved | not_proved | none proved | unproved |
| catalog_broadcast_transpose | not_proved | proved | not needed | 57 → 3 |
| catalog_broadcast_slice_broadcast_only | not_proved | not_proved | none proved | unproved |
| catalog_broadcast_slice_broadcast | not_proved | not_proved | none proved | unproved |
| catalog_indexing_slice_singleton_stride | not_proved | proved | not needed | 48 → 3 |
| catalog_dot_transpose_both_operands | not_proved | not_proved | none proved | unproved |
| catalog_dot_move_broadcast | proved | proved | not needed | 43 → 6 |
| catalog_dot_slice_output | not_proved | not_proved | none proved | unproved |
| catalog_dot_reduce_output_free | proved | proved | not needed | 35 → 6 |
| catalog_dot_scalar_multipliers | not_proved | not_proved | none proved | unproved |
| catalog_reduction_broadcast_original_axes | proved | proved | not needed | 24 → 6 |

## Supplementary scalar-setting cohort

These four comparison-setting cases are measured and disclosed separately; they do not change the primary axis/extent denominator.

| Case | Explicit | Full auto |
|---|---|---|
| catalog_elementwise_log_power | not_proved | not_proved |
| catalog_elementwise_compare_selected_mask | proved | not_proved |
| catalog_elementwise_compare_max | proved | not_proved |
| catalog_elementwise_compare_min | proved | not_proved |

## Assistance evidence

Executed 37 predeclared assisted variants; 2 proved. Multiple variants of one case count only once toward additional case coverage.
Best means fewest remaining attribute tokens, then most requested fields, then stable variant name. All alternatives, failures and skipped variants are in results/corpus.json.

### scalar_broadcast_transpose: assisted_field__dimensions

```text
User pins: {"out": {"dimensions": "[]"}}
dimensions = []
new_extents = gather(@b.new_extents, [1, 0])
```

### slice_compose: assisted_field__strides

```text
User pins: {"out": {"strides": "[1]"}}
starts = extent_add(@outer.starts, @inner.starts)
sizes = @outer.sizes
strides = [1]
```

## Unproved full-auto cases

An unproved result is not an impossibility claim. Exact diagnostics are retained below and in raw evidence.

### scalar_broadcast_transpose

```text
labs/attribute_inference/tepl/corpus/scalar_broadcast_transpose/full_auto.tepl:5:1: cannot verify rule 'scalar_broadcast_transpose': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(gather(@#1.new_extents, gather([1, 0], ordered_complement([0, 1], gather([1, 0], positions(n1.N, n1.S))))), gather(@#1.new_extents, [1, 0])). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reduce_compose

```text
labs/attribute_inference/tepl/corpus/reduce_compose/full_auto.tepl:5:1: cannot verify rule 'reduce_compose': attribute construction: declared conditions and LHS legality do not establish legality obligation: same_set(gather(n2.F, [0]), difference_axes(gather(concat(n1.K, gather(concat(n2.K, n2.F), [1, 0])), [1, 0, 2]), gather(concat(n1.K, gather(concat(n2.K, n2.F), [1, 0])), [2, 0]))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reverse_compose_disjoint

```text
labs/attribute_inference/tepl/corpus/reverse_compose_disjoint/full_auto.tepl:5:1: cannot verify rule 'reverse_compose_disjoint': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reverse_add_scalar

```text
labs/attribute_inference/tepl/corpus/reverse_add_scalar/full_auto.tepl:5:1: cannot verify rule 'reverse_add_scalar': coordinate correspondence: declared conditions and LHS legality do not establish legality obligation: same_set(n4.S, gather(n4.S, positions(empty_axes(), n2.S))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### slice_compose

```text
labs/attribute_inference/tepl/corpus/slice_compose/full_auto.tepl:5:1: cannot verify rule 'slice_compose': attribute construction: declared conditions and LHS legality do not establish legality obligation: is_positive_extents(offset_extents(extent_sub(offset_extents(@#0.sizes, -1), offset_extents(@#0.sizes, -1)), 1)). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### slice_reverse

```text
labs/attribute_inference/tepl/corpus/slice_reverse/full_auto.tepl:5:1: cannot verify rule 'slice_reverse': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### dot_move_reverse

```text
labs/attribute_inference/tepl/corpus/dot_move_reverse/full_auto.tepl:5:1: cannot verify rule 'dot_move_reverse': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reduce_compose_leading

```text
labs/attribute_inference/tepl/corpus/reduce_compose_leading/full_auto.tepl:4:1: cannot verify rule 'reduce_compose_leading': attribute construction: declared conditions and LHS legality do not establish legality obligation: same_set(filter_axes(gather(concat(n2.K, n2.F), [0, 1]), n2.F), difference_axes(concat(n1.K, gather(concat(n2.K, n2.F), [0, 1])), gather(concat(n1.K, gather(concat(n2.K, n2.F), [0, 1])), [1, 0]))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### slice_compose_zero_starts

```text
labs/attribute_inference/tepl/corpus/slice_compose_zero_starts/full_auto.tepl:4:1: cannot verify rule 'slice_compose_zero_starts': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### broadcast_compose_append

```text
labs/attribute_inference/tepl/corpus/broadcast_compose_append/full_auto.tepl:4:1: cannot verify rule 'broadcast_compose_append': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(gather(concat(gather(concat(shape(Input(#0)), @#1.new_extents), inverse(concat(range(len(shape(Input(#0)))), offset_positions(range(len(@#1.new_extents)), len(shape(Input(#0))))))), @#0.new_extents), gather(inverse(concat(range(+(len(shape(Input(#0))), len(@#1.new_extents))), offset_positions(range(len(@#0.new_extents)), +(len(shape(Input(#0))), len(@#1.new_extents))))), offset_positions(range(+(len(@#1.new_extents), len(@#0.new_extents))), len(shape(Input(#0)))))), gather(concat(@#1.new_extents, @#0.new_extents), order_positions(concat(offset_positions(range(len(@#1.new_extents)), len(shape(Input(#0)))), offset_positions(range(len(@#0.new_extents)), +(len(shape(Input(#0))), len(@#1.new_extents))))))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### dot_no_contraction

```text
labs/attribute_inference/tepl/corpus/dot_no_contraction/full_auto.tepl:4:1: cannot verify rule 'dot_no_contraction': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(shape(Input(#0)), shape(Input(#1))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### expr_reverse_compose_general

```text
labs/attribute_inference/tepl/corpus/expr_reverse_compose_general/full_auto.tepl:4:1: cannot verify rule 'expr_reverse_compose_general': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### expr_slice_reverse_general

```text
labs/attribute_inference/tepl/corpus/expr_slice_reverse_general/full_auto.tepl:4:1: cannot verify rule 'expr_slice_reverse_general': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### expr_convolution_scale_filter_bias

```text
labs/attribute_inference/tepl/corpus/expr_convolution_scale_filter_bias/full_auto.tepl:4:1: cannot verify rule 'expr_convolution_scale_filter_bias': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(shape(Input(#1)), shape(Input(#3))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_elementwise_log_power

```text
labs/attribute_inference/tepl/corpus/catalog_elementwise_log_power/full_auto.tepl:6:1: cannot verify rule 'catalog_elementwise_log_power': attribute construction: legality obligation contains an unresolved auto field. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_elementwise_compare_selected_mask

```text
labs/attribute_inference/tepl/corpus/catalog_elementwise_compare_selected_mask/full_auto.tepl:6:1: cannot verify rule 'catalog_elementwise_compare_selected_mask': attribute construction: legality obligation contains an unresolved auto field. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_elementwise_compare_max

```text
labs/attribute_inference/tepl/corpus/catalog_elementwise_compare_max/full_auto.tepl:6:1: cannot verify rule 'catalog_elementwise_compare_max': attribute construction: legality obligation contains an unresolved auto field. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_elementwise_compare_min

```text
labs/attribute_inference/tepl/corpus/catalog_elementwise_compare_min/full_auto.tepl:6:1: cannot verify rule 'catalog_elementwise_compare_min': attribute construction: legality obligation contains an unresolved auto field. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_broadcast_divide_broadcast

```text
labs/attribute_inference/tepl/corpus/catalog_broadcast_divide_broadcast/full_auto.tepl:6:1: cannot verify rule 'catalog_broadcast_divide_broadcast': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(shape(Input(#0)), []). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_broadcast_multiply_constant_into_broadcast

```text
labs/attribute_inference/tepl/corpus/catalog_broadcast_multiply_constant_into_broadcast/full_auto.tepl:6:1: cannot verify rule 'catalog_broadcast_multiply_constant_into_broadcast': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(shape(Input(#2)), inverse(concat(positions(empty_axes(), n2.S), ordered_complement([], positions(empty_axes(), n2.S))))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_broadcast_slice_broadcast_only

```text
labs/attribute_inference/tepl/corpus/catalog_broadcast_slice_broadcast_only/full_auto.tepl:6:1: cannot verify rule 'catalog_broadcast_slice_broadcast_only': attribute construction: declared conditions and LHS legality do not establish legality obligation: same_set(n1.S, empty_axes()). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_broadcast_slice_broadcast

```text
labs/attribute_inference/tepl/corpus/catalog_broadcast_slice_broadcast/full_auto.tepl:6:1: cannot verify rule 'catalog_broadcast_slice_broadcast': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_dot_transpose_both_operands

```text
labs/attribute_inference/tepl/corpus/catalog_dot_transpose_both_operands/full_auto.tepl:6:1: cannot verify rule 'catalog_dot_transpose_both_operands': attribute construction: declared conditions and LHS legality do not establish legality obligation: same_set(gather(concat(gather(gather(input2, [0]), [0]), gather(gather(gather(input0, [1]), [0]), [0])), [1, 0]), difference_axes(union_axes(input0, gather(concat(gather(gather(input0, [0]), [0]), gather(input2, [0])), [1, 0])), filter_axes(gather(concat(gather(gather(input0, [0]), [0]), gather(input2, [0])), [1, 0]), gather(gather(input0, [0]), [0])))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_dot_slice_output

```text
labs/attribute_inference/tepl/corpus/catalog_dot_slice_output/full_auto.tepl:6:1: cannot verify rule 'catalog_dot_slice_output': attribute construction: no scoped coordinate match for calculated-read attributes; composed or inverse inference may be required. Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### catalog_dot_scalar_multipliers

```text
labs/attribute_inference/tepl/corpus/catalog_dot_scalar_multipliers/full_auto.tepl:6:1: cannot verify rule 'catalog_dot_scalar_multipliers': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(inverse(concat(positions(empty_axes(), n1.S), ordered_complement([], positions(empty_axes(), n1.S)))), concat(gather(shape(Input(#1)), [0]), gather(shape(Input(#2)), [1]))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

## Validation and limits

The incorrect permutation-order control is **not_proved**, as expected. Acceptance requires an explicit proved status, true additional guard and completed requested constructors.
All variants preserve original guards, tensor restrictions, operation structure and local dialect semantics. Some source cases specialize ranks, layouts or attribute domains; case scopes are retained in the manifest.
No compiler changes, stronger applicability guards or operation-semantic substitutions were introduced to improve coverage.
This run completes the classified XLA attribute cohort. It does not complete the planned general-symbolic, independent numerical, scalability or ablation experiments.
Single CLI wall times are reproducibility evidence, not performance results. Algebraic verification does not establish complete IEEE or XLA backend equivalence.
