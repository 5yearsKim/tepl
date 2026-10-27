# Semantic Verification of HLO Rewrites

Every explicit case in [compilation_cases.json](../data/compilation_cases.json) has a verification counterpart in [rules/](../../tepl/rules/semantic/). The cases use the local [HLO dialect](../../tepl/dialects/hlo.tepl), with project-root imports configured by [tepl.toml](../../tepl/tepl.toml). No case imports another project's dialect.

## Result

The sweep tested **209 explicit rule cases**, spanning **199 of 239 catalog schemas**:

| Outcome | Rule cases | Meaning |
|---|---:|---|
| Accepted by semantic verification | 94 | Explicit proof status is `proved`, with no additional guard. |
| Missing operation semantics | 90 | At least one required operation has no faithful definition in the current semantic fragment. |
| Not proved with the declared semantics | 25 | Required operation semantics exist, but construction, legality or equivalence was not established. |
| Timeout | 0 | The per-case time limit was exceeded. |

At least one case was verified for **87 catalog schemas**. A specialization does not prove all variants of its catalog schema. The deliberately incorrect subtraction-to-addition control was rejected and is excluded from the table.

The remaining **40 catalog schemas** have no concrete rule in this corpus; they are not counted as attempted or rejected.

All tested rules supply RHS attributes explicitly and use no `auto`. Proofs use the ideal `algebraic` arithmetic policy. They do not establish IEEE rounding, NaN, signed-zero, fixed-width overflow or complete XLA backend equivalence. A rejected case means unproved, not mathematically false.


## General reasoning comparison

The frozen baseline verified **80 cases**. The general reasoning extension verifies **94**, adding **14** with **no previous proof lost**. Rule bodies, conditions, rank constraints and both HLO dialects are unchanged. Manifest edits only relocate case files, as recorded in [layout_migration.json](../baselines/layout_migration.json); [reasoning_baseline.json](../baselines/reasoning_baseline.json) preserves their hashes and the original per-case outcomes.

The compiler now lowers metadata indexing and pointwise comprehensions, propagates integer intervals and source extent inequalities, composes ordered sequence mappings, and combines reversals by selector parity. Original derived metadata programs retain checked-evaluation and final attribute-range obligations. No case names or HLO-specific identities are used by the reasoning implementation.

Examples include [reverse composition](../../tepl/rules/semantic/expr_reverse_compose_general.tepl), [slice composition](../../tepl/rules/semantic/expr_slice_compose_general.tepl), and [dot operand transposes](../../tepl/rules/semantic/catalog_dot_transpose_both_operands.tepl). The baseline and current result JSON retain all per-case outcomes. This comparison measures proof coverage, not compiler speed.

## Results by category

| Category | Catalog schemas | Cases tested | Verified cases | Schemas with a verified case | Missing semantics | Not proved | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| elementwise | 99 | 84 | 62 | 62 | 8 | 14 | 0 |
| indexing | 60 | 49 | 8 | 4 | 37 | 4 | 0 |
| broadcast | 20 | 20 | 6 | 4 | 9 | 5 | 0 |
| transpose-reshape | 9 | 8 | 2 | 2 | 6 | 0 | 0 |
| dot | 18 | 17 | 11 | 11 | 5 | 1 | 0 |
| reduction | 26 | 24 | 5 | 4 | 19 | 0 | 0 |
| convolution | 7 | 7 | 0 | 0 | 6 | 1 | 0 |

## Local HLO semantics

The local dialect declares **61 operations** for the evaluation and defines value/axis semantics for **33**: `add`, `multiply`, `negate`, `subtract`, `zero_like`, `dot_general`, `convolution`, `slice`, `broadcast_in_dim`, `transpose`, `reverse`, `reduce_sum`, `divide`, `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `power`, `maximum`, `minimum`, `logical_not`, `logical_and`, `logical_or`, `logical_xor`, `select`, `remainder`, `compare`, `clamp`, `cbrt`, `erf`, `logistic`, `tanh`.

These cover polynomial arithmetic, registered pointwise scalar functions, sum reductions, dot, transpose, axis-insertion broadcast, reverse, positive-size static slice, and valid fixed-layout ungrouped convolution. `zero_like` is a helper for shaped zero. HLO attributes are flattened for this experiment; this is not a complete HLO importer.

Broadcast retains existing extents and inserts new axes. Slice sizes are positive. The simple convolution has positive strides/dilations and no padding. `reduce_sum` has an implicit zero initializer. Source assumptions and case restrictions remain in the manifests.

Scalar functions share one representation and recursive coordinate tracking. Arguments normalize using existing polynomial rules; other functions stay intact until an implemented law applies. The current laws include min/max associativity, commutativity, idempotence and absorption; Boolean identities; select simplification; integer powers; and selected exp/log/sqrt/abs identities. Numeric domains are checked conservatively: generic quotient reassociation or log/sqrt rewrites without the required domain facts remain unproved. The registry does not make every identity involving its functions provable.

Logical operations in this profile are Boolean, not integer bitwise operations. Select uses a predicate with the same shape as its arms. Compare uses EQ=0, NE=1, GE=2, GT=3, LE=4, LT=5. These are explicit local representations, not XLA's native attribute encoding.

### Semantics the current language cannot express

The other **28 operations** remain declarations. Their absence is intentional: substituting identity, sum, or the desired RHS for their actual meaning would produce invalid evidence. [hlo_support.json](../data/hlo_support.json) records each operation's reason.

| Missing capability | Operations | Explanation |
|---|---|---|
| complex | `real`, `imag`, `complex` | The current value language and dtype model do not represent complex components and projections. |
| reshape coordinates | `reshape` | General rank-changing reshape requires coordinate linearization/delinearization, including division/modulo of coordinates. Current calculated reads support addition, scaling, offsets, selection and reversal only. |
| piecewise variadic | `concatenate` | Concatenation needs coordinate-dependent branch selection and reads with shifted coordinates. Scalar select operates on values, and cannot test logical coordinates. Semantic operation operands cannot be variadic. |
| piecewise coordinates | `pad`, `nonnegative_pad` | Padding requires coordinate bounds predicates and selecting a valid input read or padding value. Scalar select cannot test logical coordinates, and calculated reads do not support coordinate-dependent branches. |
| tensor indices | `dynamic_slice`, `dynamic_update_slice`, `scatter`, `scatter_add`, `gather` | Tensor index values cannot be used as calculated-read coordinates. Updates/scatter also need piecewise writes or gather-style inverse indexing; variadic starts cannot be semantic operands. |
| region body | `reduce`, `map` | Opaque region handles cannot be interpreted as scalar value functions. Arbitrary combiners and their initializers cannot be replaced with sum semantics. |
| conversion | `convert` | Numeric conversions require rounding/truncation/fixed-width value semantics. A dtype contract alone does not describe conversion values. |
| attribute constant | `fill`, `scalar` | Const accepts a fixed signed integer literal, not attrs.value. Graph numeric literals are supported, but a constant operation with a captured payload still needs value-attribute lowering. |
| coordinate value | `iota` | The value language cannot turn a logical coordinate into a scalar value; output coordinate identities are distinct from tensor arithmetic values. |
| modulo coordinates | `rotate` | Rotation requires coordinate modulo or piecewise wraparound, absent from calculated-read primitives. |
| non sum reducer | `reduce_min`, `reduce_max`, `reduce_and`, `reduce_or` | Reduce represents summation only. Registered scalar min/max/Boolean functions do not supply reduction binders or their initializer behavior. |
| window initializer | `reduce_window`, `reduce_window_sum`, `reduce_window_min`, `reduce_window_max` | General windows need padding predicates and declared initializer behavior; generic/min/max reducers need additional primitives. Modeling reduce_window_sum by ignoring its initializer would be incorrect. |
| general convolution | `general_convolution` | General dimension numbers, grouping, signed padding and reversal exceed the current coordinate and semantic-attribute fragments. The simpler convolution declaration covers valid ungrouped fixed-layout convolution. |

The value vocabulary includes `Const`, operand/read, `Add`, binary `Product`, sum `Reduce`, and typed registered scalar applications. Graph numeric literals are also supported. Calculated coordinates support addition, scaling, offsets, selection and reversal. Required scalar numeric attributes can parameterize functions, such as compare's direction. `Const` still requires an integer literal; it cannot read a constant payload from an attribute. Scalar `select` does not supply coordinate predicates or a general reduction binder.

## Unproved cases with operation semantics

These cases reached the semantic path with definitions for all required operations. Their diagnostics distinguish unsupported attribute expressions from unresolved coordinate/legality/value proofs.

| Case | Limitation |
|---|---|
| [slice_reverse](../../tepl/rules/semantic/slice_reverse.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_divide_two_quotients](../../tepl/rules/semantic/expr_divide_two_quotients.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_divide_left_quotient](../../tepl/rules/semantic/expr_divide_left_quotient.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_divide_right_quotient](../../tepl/rules/semantic/expr_divide_right_quotient.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_log_sqrt](../../tepl/rules/semantic/expr_log_sqrt.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_power_reciprocal](../../tepl/rules/semantic/expr_power_reciprocal.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_remainder_compose](../../tepl/rules/semantic/expr_remainder_compose.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [expr_slice_reverse_general](../../tepl/rules/semantic/expr_slice_reverse_general.tepl) | The comprehension needs reasoning beyond the supported pointwise sequence fragment. |
| [expr_convolution_scale_filter_bias](../../tepl/rules/semantic/expr_convolution_scale_filter_bias.tepl) | The checker does not establish nonnegative extents for the explicitly constructed shape list. |
| [catalog_elementwise_divide_power](../../tepl/rules/semantic/catalog_elementwise_divide_power.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_elementwise_divide_sqrt](../../tepl/rules/semantic/catalog_elementwise_divide_sqrt.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_elementwise_divide_rsqrt](../../tepl/rules/semantic/catalog_elementwise_divide_rsqrt.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_elementwise_log_power](../../tepl/rules/semantic/catalog_elementwise_log_power.tepl) | The checker does not establish nonnegative extents for the explicitly constructed shape list. |
| [catalog_elementwise_log_rsqrt](../../tepl/rules/semantic/catalog_elementwise_log_rsqrt.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_elementwise_max_clamp_bound](../../tepl/rules/semantic/catalog_elementwise_max_clamp_bound.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_elementwise_select_compare_gt](../../tepl/rules/semantic/catalog_elementwise_select_compare_gt.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_elementwise_select_compare_lt](../../tepl/rules/semantic/catalog_elementwise_select_compare_lt.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_broadcast_divide_broadcast](../../tepl/rules/semantic/catalog_broadcast_divide_broadcast.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_broadcast_transpose](../../tepl/rules/semantic/catalog_broadcast_transpose.tepl) | The comprehension needs reasoning beyond the supported pointwise sequence fragment. |
| [catalog_broadcast_transpose_broadcast](../../tepl/rules/semantic/catalog_broadcast_transpose_broadcast.tepl) | The comprehension needs reasoning beyond the supported pointwise sequence fragment. |
| [catalog_broadcast_slice_broadcast_only](../../tepl/rules/semantic/catalog_broadcast_slice_broadcast_only.tepl) | The original derived metadata expression needs an indexing, arithmetic or attribute-range proof. |
| [catalog_broadcast_slice_broadcast](../../tepl/rules/semantic/catalog_broadcast_slice_broadcast.tepl) | The checker does not establish nonnegative extents for the explicitly constructed shape list. |
| [catalog_indexing_reverse_trivial](../../tepl/rules/semantic/catalog_indexing_reverse_trivial.tepl) | Normalization does not establish scalar-value, tensor-access and reduction-domain equality. |
| [catalog_indexing_slice_singleton_stride](../../tepl/rules/semantic/catalog_indexing_slice_singleton_stride.tepl) | The comprehension needs reasoning beyond the supported pointwise sequence fragment. |
| [catalog_dot_slice_output](../../tepl/rules/semantic/catalog_dot_slice_output.tepl) | The checker does not establish nonnegative extents for the explicitly constructed shape list. |

## Reproduction

[verification.json](../results/verification.json) records diagnostics, required/missing operations, proof evidence and file/dependency hashes, including the semantic compiler sources and untracked additions. The revision and semantic source hashes identify the measured compiler. [verification_cases.json](../data/verification_cases.json) records scope and arithmetic policy. The validators reject stale inputs and missing or duplicate cases. [Language coverage](compilation.md) separately measures ordinary compilation using [the structural HLO profile](../../tepl/dialects/hlo_structural.tepl).

- TEPL revision: `17d48f05087b540cae87648fed3b43d9f14c2166`.
- XLA revision: `e02127ba9c9c48912b0e6776fe5bfe3843d81042`.
- Compiler SHA-256: `be7c72d324619a53ea1758b65d009f73e6213f388f9e4ecdb034282e6d8ca465`.
- Environment: Linux-7.0.0-38-generic-x86_64-with-glibc2.43; Python 3.14.4.
- Timeout: 30 seconds per case.

See the [case outcome table](synthesis_cases.md) for explicit and inferred results, and the [reproduction commands](../README.md#reproduce) to rerun the evaluation.
