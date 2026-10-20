# Attribute inference: batch 01

Five predeclared family strata: first remaining axis/extent case with all operation semantics in broadcast, reduction, dot, indexing, convolution, in original manifest order. Reduction and indexing strata require a recorded unproved explicit baseline to probe synthesis beyond baseline-proved rules. Exclude pilot cases. Not representative sampling.

For the two selected mixed-operation cases, infer only broadcast descriptors and retain the other operations' captured descriptors. No guard changes. No post-result tuning.

These are five new cases, disjoint from the first pilot. Fractions below are within-batch outcomes, not corpus-wide coverage.

- Explicit versions proved: **2/5**.
- Full-auto versions proved, all batch cases: **2/5**.
- Full-auto versions proved, conditioned on current explicit proof: **2/2**.
- Full-auto successes with an unproved explicit baseline: **0**.
- Assisted variants proved: **0/2**; additional recovered cases: **0**.

| Case | Explicit | Full auto | Assisted operation |
|---|---|---|---|
| broadcast_add | proved | proved | not attempted |
| reduce_compose | not_proved | not_proved | not attempted |
| dot_associate | proved | proved | not attempted |
| reverse_add_scalar | not_proved | not_proved | not_proved |
| expr_convolution_scale_filter_bias | not_proved | not_proved | not_proved |

## Attribute code

Same token proxy as the pilot: RHS annotations plus derivations. Conditions, restrictions and semantics are unchanged.
Inferred-field counts exclude pins. Only proved variants receive a savings estimate.

| Case | Variant | Requested fields | Attribute tokens | Reduction vs explicit |
|---|---|---:|---:|---:|
| broadcast_add | explicit | 0 | 3 | 0 (0.0%) |
| broadcast_add | full_auto | 2 | 3 | 0 (0.0%) |
| reduce_compose | explicit | 0 | 18 | unproved |
| reduce_compose | full_auto | 1 | 3 | unproved |
| dot_associate | explicit | 0 | 6 | 0 (0.0%) |
| dot_associate | full_auto | 8 | 6 | 0 (0.0%) |
| reverse_add_scalar | explicit | 0 | 6 | unproved |
| reverse_add_scalar | full_auto | 3 | 6 | unproved |
| reverse_add_scalar | assisted_operation | 2 | 6 | unproved |
| expr_convolution_scale_filter_bias | explicit | 0 | 30 | unproved |
| expr_convolution_scale_filter_bias | full_auto | 6 | 9 | unproved |
| expr_convolution_scale_filter_bias | assisted_operation | 2 | 9 | unproved |

## Successful constructors

### broadcast_add: full_auto

```text
dimensions = @left.dimensions
new_extents = @left.new_extents
```

### dot_associate: full_auto

```text
lhs_contracting_dimensions = [1]
rhs_contracting_dimensions = [0]
lhs_batching_dimensions = []
rhs_batching_dimensions = []
lhs_contracting_dimensions = [1]
rhs_contracting_dimensions = [0]
lhs_batching_dimensions = []
rhs_batching_dimensions = []
```

## Unproved cases

A failed explicit baseline is permitted and reported; it is not grounds to remove the case.

### reduce_compose: explicit

```text
labs/attribute_inference/tepl/batch_01/reduce_compose/explicit.tepl:5:1: cannot verify rule 'reduce_compose': attribute construction: declared conditions and LHS legality do not establish legality obligation: same_set(gather(n2.F, [0]), difference_axes(gather(concat(n1.K, gather(concat(n2.K, n2.F), [1, 0])), [1, 0, 2]), gather(concat(n1.K, gather(concat(n2.K, n2.F), [1, 0])), [0, 2]))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reduce_compose: full_auto

```text
labs/attribute_inference/tepl/batch_01/reduce_compose/full_auto.tepl:5:1: cannot verify rule 'reduce_compose': attribute construction: declared conditions and LHS legality do not establish legality obligation: same_set(gather(n2.F, [0]), difference_axes(gather(concat(n1.K, gather(concat(n2.K, n2.F), [1, 0])), [1, 0, 2]), gather(concat(n1.K, gather(concat(n2.K, n2.F), [1, 0])), [2, 0]))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reverse_add_scalar: explicit

```text
labs/attribute_inference/tepl/batch_01/reverse_add_scalar/explicit.tepl:5:1: cannot verify rule 'reverse_add_scalar': coordinate correspondence: declared conditions and LHS legality do not establish legality obligation: same_set(n4.S, gather(n4.S, positions(empty_axes(), n2.S))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reverse_add_scalar: full_auto

```text
labs/attribute_inference/tepl/batch_01/reverse_add_scalar/full_auto.tepl:5:1: cannot verify rule 'reverse_add_scalar': coordinate correspondence: declared conditions and LHS legality do not establish legality obligation: same_set(n4.S, gather(n4.S, positions(empty_axes(), n2.S))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### reverse_add_scalar: assisted_operation

```text
labs/attribute_inference/tepl/batch_01/reverse_add_scalar/assisted_operation.tepl:5:1: cannot verify rule 'reverse_add_scalar': coordinate correspondence: declared conditions and LHS legality do not establish legality obligation: same_set(n4.S, gather(n4.S, positions(empty_axes(), n2.S))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### expr_convolution_scale_filter_bias: explicit

```text
labs/attribute_inference/tepl/batch_01/expr_convolution_scale_filter_bias/explicit.tepl:4:1: cannot verify rule 'expr_convolution_scale_filter_bias': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(gather(shape(Input(#1)), gather([2], [0, 1])), [index(shape(Input(#1)), 0), index(shape(Input(#0)), 2)]). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### expr_convolution_scale_filter_bias: full_auto

```text
labs/attribute_inference/tepl/batch_01/expr_convolution_scale_filter_bias/full_auto.tepl:4:1: cannot verify rule 'expr_convolution_scale_filter_bias': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(shape(Input(#1)), shape(Input(#3))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

### expr_convolution_scale_filter_bias: assisted_operation

```text
labs/attribute_inference/tepl/batch_01/expr_convolution_scale_filter_bias/assisted_operation.tepl:4:1: cannot verify rule 'expr_convolution_scale_filter_bias': attribute construction: declared conditions and LHS legality do not establish legality obligation: ==(shape(Input(#1)), shape(Input(#3))). Add an explicit where condition establishing the unresolved requirement, or change the rewrite.
```

## Scope and next checkpoint

Broadcast-add is a descriptor-reuse opportunity. Reduction composition, scalar-broadcast reversal and convolution scaling also test cases whose explicit baselines were previously unproved. Dot reassociation in this corpus is restricted to rank-two matrices; its constant attributes are not general-rank reconstruction evidence.
Full-auto and assisted success require checked constructors and `additional-guard: true`. Unproved outcomes are not impossibility results.
Original explicit source files, guards and operation semantics were not edited. No compiler changes were made for inference.
The classifier is a static inventory, not an execution of the remaining corpus. CLI wall times are not a performance study.
Independent numerical validation, general symbolic additions, and scaling are still pending.
**Stop for user review before selecting or running the next batch.**
