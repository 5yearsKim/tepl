# Semantic Verification of HLO Rewrites

Every explicit case in [compilation_cases.json](../data/compilation_cases.json) has a verification counterpart in [rules/](../../tepl/rules/semantic/). The cases use the local [HLO dialect](../../tepl/dialects/hlo.tepl), with project-root imports configured by [tepl.toml](../../tepl/tepl.toml). No case imports another project's dialect.

## Result

The sweep tested **209 explicit rule cases**, spanning **199 of 239 catalog schemas**:

| Outcome | Rule cases | Meaning |
|---|---:|---|
| Accepted by semantic verification | 95 | Explicit proof status is `proved`, with no additional guard. |
| Missing operation semantics | 90 | At least one required operation has no faithful definition in the current semantic fragment. |
| Not proved with the declared semantics | 24 | Required operation semantics exist, but construction, legality or equivalence was not established. |
| Timeout | 0 | The per-case time limit was exceeded. |

At least one case was verified for **88 catalog schemas**. A specialization does not prove all variants of its catalog schema. The deliberately incorrect subtraction-to-addition control was rejected and is excluded from the table.

The remaining **40 catalog schemas** have no concrete rule in this corpus; they are not counted as attempted or rejected.

All tested rules supply RHS attributes explicitly and use no `auto`. Proofs use the ideal `algebraic` arithmetic policy. They do not establish IEEE rounding, NaN, signed-zero, fixed-width overflow or complete XLA backend equivalence. A rejected case means unproved, not mathematically false.


## General reasoning comparison

The frozen baseline verified **80 cases**. The general reasoning extension verifies **95**, adding **15** with **no previous proof lost**. Rule bodies, conditions, rank constraints and both HLO dialects are unchanged. Manifest edits only relocate case files, as recorded in [layout_migration.json](../baselines/layout_migration.json); [reasoning_baseline.json](../baselines/reasoning_baseline.json) preserves their hashes and the original per-case outcomes.

The compiler now lowers metadata indexing and pointwise comprehensions, propagates integer intervals and source extent inequalities, composes ordered sequence mappings, and combines reversals by selector parity. Original derived metadata programs retain checked-evaluation and final attribute-range obligations. No case names or HLO-specific identities are used by the reasoning implementation.

Newly verified cases:

- [broadcast_compose_append](../../tepl/rules/semantic/broadcast_compose_append.tepl)
- [catalog_broadcast_multiply_constant_into_broadcast](../../tepl/rules/semantic/catalog_broadcast_multiply_constant_into_broadcast.tepl)
- [catalog_dot_scalar_multipliers](../../tepl/rules/semantic/catalog_dot_scalar_multipliers.tepl)
- [catalog_dot_transpose_both_operands](../../tepl/rules/semantic/catalog_dot_transpose_both_operands.tepl)
- [catalog_indexing_reverse_trivial](../../tepl/rules/semantic/catalog_indexing_reverse_trivial.tepl)
- [dot_move_reverse](../../tepl/rules/semantic/dot_move_reverse.tepl)
- [dot_no_contraction](../../tepl/rules/semantic/dot_no_contraction.tepl)
- [expr_reverse_compose_general](../../tepl/rules/semantic/expr_reverse_compose_general.tepl)
- [expr_slice_compose_general](../../tepl/rules/semantic/expr_slice_compose_general.tepl)
- [reduce_compose](../../tepl/rules/semantic/reduce_compose.tepl)
- [reduce_compose_leading](../../tepl/rules/semantic/reduce_compose_leading.tepl)
- [reverse_add_scalar](../../tepl/rules/semantic/reverse_add_scalar.tepl)
- [reverse_compose_disjoint](../../tepl/rules/semantic/reverse_compose_disjoint.tepl)
- [slice_compose](../../tepl/rules/semantic/slice_compose.tepl)
- [slice_identity](../../tepl/rules/semantic/slice_identity.tepl)

This comparison measures proof coverage, not compiler speed. Missing operation semantics remain unchanged. Division, metadata reductions, more general comprehensions and unresolved layout or numeric-domain obligations can still remain unproved.

## Results by category

| Category | Catalog schemas | Cases tested | Verified cases | Schemas with a verified case | Missing semantics | Not proved | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| elementwise | 99 | 84 | 62 | 62 | 8 | 14 | 0 |
| indexing | 60 | 49 | 9 | 5 | 37 | 3 | 0 |
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
| [catalog_indexing_slice_singleton_stride](../../tepl/rules/semantic/catalog_indexing_slice_singleton_stride.tepl) | The comprehension needs reasoning beyond the supported pointwise sequence fragment. |
| [catalog_dot_slice_output](../../tepl/rules/semantic/catalog_dot_slice_output.tepl) | The checker does not establish nonnegative extents for the explicitly constructed shape list. |

## Measured rules

Every row is a real compiler invocation. A successful ordinary type check without `status=proved` does not count as verification. Missing semantics lists all required operations without definitions, even when the compiler reports only its first such operation.

| Case | Catalog schema | Result | Missing semantics | Scope |
|---|---|---|---|---|
| [add_zero](../../tepl/rules/semantic/add_zero.tepl) | elementwise-add-zero | proved |  | One operand order. zero_like models a shaped literal zero; no HLO literal matcher is evaluated. |
| [add_constant_order](../../tepl/rules/semantic/add_constant_order.tepl) | elementwise-add-constant-order | proved |  | Proves commutation for arbitrary operands, including the catalog constant case. Constant/nonconstant matching is not implemented here. |
| [add_constants](../../tepl/rules/semantic/add_constants.tepl) | elementwise-add-constants | proved |  | C1/C2 remain tensor captures; proves the equation for all values, including constants. RHS constant folding is outside this check. |
| [subtract_add_constants](../../tepl/rules/semantic/subtract_add_constants.tepl) | elementwise-subtract-add-constants | proved |  | C1/C2 are tensor captures; the algebraic equation holds for all values. No literal folding. |
| [add_subtract_cancel](../../tepl/rules/semantic/add_subtract_cancel.tepl) | elementwise-add-subtract-cancel | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [factor_common_product](../../tepl/rules/semantic/factor_common_product.tepl) | elementwise-factor-common-product | proved |  | One matched factor order under algebraic policy; source numerical gates remain an integration obligation. |
| [subtract_self](../../tepl/rules/semantic/subtract_self.tepl) | elementwise-subtract-self | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [subtract_zero](../../tepl/rules/semantic/subtract_zero.tepl) | elementwise-subtract-zero | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [subtract_constant](../../tepl/rules/semantic/subtract_constant.tepl) | elementwise-subtract-constant | proved |  | C is a tensor capture; proves a general identity containing the constant case. No literal evaluator. |
| [subtract_subtract_cancel](../../tepl/rules/semantic/subtract_subtract_cancel.tepl) | elementwise-subtract-subtract-cancel | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [subtract_add_cancel](../../tepl/rules/semantic/subtract_add_cancel.tepl) | elementwise-subtract-add-cancel | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [add_subtract_cancel_right](../../tepl/rules/semantic/add_subtract_cancel_right.tepl) | elementwise-add-subtract-cancel-right | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [multiply_zero](../../tepl/rules/semantic/multiply_zero.tepl) | elementwise-multiply-zero | proved |  | zero_like models literal zero. Requires algebraic arithmetic policy. |
| [multiply_negatives](../../tepl/rules/semantic/multiply_negatives.tepl) | elementwise-multiply-negatives | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [multiply_constant_pairs](../../tepl/rules/semantic/multiply_constant_pairs.tepl) | elementwise-multiply-constant-pairs | proved |  | C1/C2 are tensor captures; no literal folding. |
| [multiply_constants](../../tepl/rules/semantic/multiply_constants.tepl) | elementwise-multiply-constants | proved |  | C1/C2 are tensor captures; no literal folding. |
| [double_negate](../../tepl/rules/semantic/double_negate.tepl) | elementwise-double-negate | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [transpose_identity](../../tepl/rules/semantic/transpose_identity.tepl) | transpose-reshape-transpose-identity | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [transpose_compose](../../tepl/rules/semantic/transpose_compose.tepl) | transpose-reshape-transpose-compose | proved |  | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [broadcast_add](../../tepl/rules/semantic/broadcast_add.tepl) | broadcast-elementwise | proved |  | Only binary addition, equal base shapes, identical broadcast maps and added extents; excludes other elementwise functions and scalar expansion. |
| [broadcast_multiply](../../tepl/rules/semantic/broadcast_multiply.tepl) | broadcast-elementwise | proved |  | Only binary multiplication, equal base shapes, identical maps and added extents. |
| [scalar_broadcast_reverse](../../tepl/rules/semantic/scalar_broadcast_reverse.tepl) | broadcast-scalar-through-indexing | proved |  | Only the reverse member of the parameterized indexing schema; input broadcast map is empty. |
| [scalar_broadcast_transpose](../../tepl/rules/semantic/scalar_broadcast_transpose.tepl) | broadcast-scalar-through-indexing | proved |  | Only a rank-two transpose with permutation [1,0] and scalar broadcast. |
| [reduce_negate](../../tepl/rules/semantic/reduce_negate.tepl) | reduction-negate | proved |  | Only sum with implicit zero initializer, matching the catalog arithmetic case. |
| [reduce_compose](../../tepl/rules/semantic/reduce_compose.tepl) | reduction-compose | proved |  | Rank-three sum, inner axis 1 then outer axis 1; symbolic extents. Does not cover arbitrary combiners/initializers. |
| [reduce_transpose](../../tepl/rules/semantic/reduce_transpose.tepl) | reduction-transpose | proved |  | Rank-two transpose [1,0], sum over output axis 0, implicit zero initializer. |
| [reverse_cancel](../../tepl/rules/semantic/reverse_cancel.tepl) | indexing-reverse-compose | proved |  | Identical ordered dimension lists; a subset of general symmetric-difference composition. |
| [reverse_compose_disjoint](../../tepl/rules/semantic/reverse_compose_disjoint.tepl) | indexing-reverse-compose | proved |  | Rank-two disjoint lists [0] and [1]; probes a case beyond identical-list cancellation. |
| [reverse_add_scalar](../../tepl/rules/semantic/reverse_add_scalar.tepl) | indexing-reverse-elementwise | proved |  | Only addition and a scalar broadcast; tests the compound rule directly, not a two-rule sequence. |
| [slice_identity](../../tepl/rules/semantic/slice_identity.tepl) | indexing-slice-identity | proved |  | Rank-one positive-size slice; empty outputs are excluded by the supplied dialect. |
| [slice_compose](../../tepl/rules/semantic/slice_compose.tepl) | indexing-slice-compose | proved |  | Rank-one, positive-size, unit-stride slices with an explicit computed start. |
| [slice_reverse](../../tepl/rules/semantic/slice_reverse.tepl) | indexing-slice-reverse | not_proved |  | Rank-one, positive-size, unit-stride slice of a reversed axis; explicit reflected start. |
| [dot_associate](../../tepl/rules/semantic/dot_associate.tepl) | dot-associate | proved |  | Ordinary rank-two matrix products without batch axes; every attribute is reused explicitly. |
| [dot_transpose_output](../../tepl/rules/semantic/dot_transpose_output.tepl) | dot-transpose-output | proved |  | Rank-two matrix product, output transpose [1,0]; explicit swapped contraction attributes. |
| [dot_move_reverse](../../tepl/rules/semantic/dot_move_reverse.tepl) | dot-move-reverse | proved |  | Rank-one inner product; reversal of its sole contracting axis; no free/batch axes. |
| [negative_subtract_to_add](../../tepl/rules/semantic/negative_subtract_to_add.tepl) | negative control | not_proved |  | Deliberately incorrect subtraction-to-addition rewrite; excluded from coverage counts. |
| [reduce_compose_leading](../../tepl/rules/semantic/reduce_compose_leading.tepl) | reduction-compose | proved |  | Rank-three sum with inner axis 0 then outer axis 0; implicit zero initializer. Compared with the nonleading-axis case. |
| [slice_compose_zero_starts](../../tepl/rules/semantic/slice_compose_zero_starts.tepl) | indexing-slice-compose | proved |  | Rank-one, positive-size, unit-stride slices with both starts zero. Copies outer attributes; avoids computed constructors. |
| [broadcast_compose_append](../../tepl/rules/semantic/broadcast_compose_append.tepl) | broadcast-compose | proved |  | Arbitrary input rank, broadcasts only append new dimensions; concatenates new extents explicitly. |
| [dot_swap_operands](../../tepl/rules/semantic/dot_swap_operands.tepl) | dot-swap-parameter-operand | proved |  | Rank-two matrix product. Proves the identity for any captured X/Y; does not implement parameter-like policy screening. |
| [dot_no_contraction](../../tepl/rules/semantic/dot_no_contraction.tepl) | dot-no-contraction | proved |  | Rank-one operands, no batches or contractions; same-type ideal arithmetic, no conversions. |
| [dot_reduce_product](../../tepl/rules/semantic/dot_reduce_product.tepl) | dot-reduce-product | proved |  | Rank-one inner product to multiply then sum; same-type ideal arithmetic, no widened accumulation. |
| [dot_reduce_batch](../../tepl/rules/semantic/dot_reduce_batch.tepl) | dot-reduce-output-batch | proved |  | Rank-one batched product reduced over its sole batch axis; no remaining output axes. |
| [expr_divide_one](../../tepl/rules/semantic/expr_divide_one.tepl) | elementwise-divide-one | proved |  | Scalar literal member; source numeric assumptions apply. |
| [expr_divide_two_quotients](../../tepl/rules/semantic/expr_divide_two_quotients.tepl) | elementwise-divide-two-quotients | not_proved |  | Algebraic equation; denominator and numeric assumptions remain source conditions. |
| [expr_divide_left_quotient](../../tepl/rules/semantic/expr_divide_left_quotient.tepl) | elementwise-divide-left-quotient | not_proved |  | Algebraic equation under source conditions. |
| [expr_divide_right_quotient](../../tepl/rules/semantic/expr_divide_right_quotient.tepl) | elementwise-divide-right-quotient | not_proved |  | Algebraic equation under source conditions. |
| [expr_multiply_one](../../tepl/rules/semantic/expr_multiply_one.tepl) | elementwise-multiply-one | proved |  | Scalar literal member; tensor splat matching is an adapter detail. |
| [expr_square_abs](../../tepl/rules/semantic/expr_square_abs.tepl) | elementwise-square-abs | proved |  | Noncomplex mathematical values; source numeric conditions apply. |
| [expr_multiply_exponentials](../../tepl/rules/semantic/expr_multiply_exponentials.tepl) | elementwise-multiply-exponentials | proved |  | Mathematical equation; source fast-math gate is recorded in the catalog. |
| [expr_double_not](../../tepl/rules/semantic/expr_double_not.tepl) | elementwise-double-not | proved |  | Logical/bitwise operation supplied by a dialect declaration. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [expr_xor_self](../../tepl/rules/semantic/expr_xor_self.tepl) | elementwise-xor-self | proved |  | Scalar integer member; literal dtype follows the matched rule context. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [expr_xor_zero](../../tepl/rules/semantic/expr_xor_zero.tepl) | elementwise-xor-zero | proved |  | Scalar integer member. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [expr_and_zero](../../tepl/rules/semantic/expr_and_zero.tepl) | elementwise-and-zero | proved |  | Scalar integer member. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [expr_or_zero](../../tepl/rules/semantic/expr_or_zero.tepl) | elementwise-or-zero | proved |  | Scalar integer member. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [expr_log_exp](../../tepl/rules/semantic/expr_log_exp.tepl) | elementwise-log-exp | proved |  | Real mathematical values; source fast-math gate remains a source condition. |
| [expr_log_sqrt](../../tepl/rules/semantic/expr_log_sqrt.tepl) | elementwise-log-sqrt | not_proved |  | Scalar X; real domain and source numeric assumptions apply. |
| [expr_maximum_self](../../tepl/rules/semantic/expr_maximum_self.tepl) | elementwise-max-self | proved |  | Operation semantics and source numeric policy supplied by target dialect. |
| [expr_maximum_nested](../../tepl/rules/semantic/expr_maximum_nested.tepl) | elementwise-max-nested-left | proved |  | One matched operand form; source ordering retained. |
| [expr_maximum_nested_right](../../tepl/rules/semantic/expr_maximum_nested_right.tepl) | elementwise-max-nested-right | proved |  | Written operand order matches the catalog. |
| [expr_minimum_self](../../tepl/rules/semantic/expr_minimum_self.tepl) | elementwise-min-self | proved |  | Operation semantics supplied by target dialect. |
| [expr_minimum_nested](../../tepl/rules/semantic/expr_minimum_nested.tepl) | elementwise-min-nested-left | proved |  | One matched operand form. |
| [expr_minimum_nested_right](../../tepl/rules/semantic/expr_minimum_nested_right.tepl) | elementwise-min-nested-right | proved |  | Written operand order matches the catalog. |
| [expr_real_complex](../../tepl/rules/semantic/expr_real_complex.tepl) | elementwise-real-complex | missing_semantics | complex, real | Typed complex construction/projection supplied by target dialect. |
| [expr_imag_complex](../../tepl/rules/semantic/expr_imag_complex.tepl) | elementwise-imag-complex | missing_semantics | complex, imag | Typed complex construction/projection supplied by target dialect. |
| [expr_complex_components](../../tepl/rules/semantic/expr_complex_components.tepl) | elementwise-complex-components | missing_semantics | complex, imag, real | Same captured complex tensor X. |
| [expr_select_identical](../../tepl/rules/semantic/expr_select_identical.tepl) | elementwise-select-identical | proved |  | Shared selected operand; target supplies compatible shape/dtype metadata. Local HLO select requires the predicate and both branches to have the same shape. |
| [expr_select_not](../../tepl/rules/semantic/expr_select_not.tepl) | elementwise-select-not | proved |  | Predicate operation supplied by target dialect. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. Local HLO select requires the predicate and both branches to have the same shape. |
| [expr_power_zero](../../tepl/rules/semantic/expr_power_zero.tepl) | elementwise-power-zero | proved |  | Scalar literal exponent member. |
| [expr_power_one](../../tepl/rules/semantic/expr_power_one.tepl) | elementwise-power-one | proved |  | Scalar literal exponent member. |
| [expr_power_two](../../tepl/rules/semantic/expr_power_two.tepl) | elementwise-power-two | proved |  | Scalar literal exponent member. |
| [expr_power_three](../../tepl/rules/semantic/expr_power_three.tepl) | elementwise-power-three | proved |  | Scalar literal exponent member. |
| [expr_power_reciprocal](../../tepl/rules/semantic/expr_power_reciprocal.tepl) | elementwise-power-reciprocal | not_proved |  | Scalar nonintegral member under the source conditions. |
| [expr_remainder_compose](../../tepl/rules/semantic/expr_remainder_compose.tepl) | elementwise-remainder-compose | not_proved |  | Same divisor capture; target remainder semantics retained. |
| [expr_reshape_identity](../../tepl/rules/semantic/expr_reshape_identity.tepl) | transpose-reshape-reshape-identity | missing_semantics | reshape | Arbitrary-rank shape guard; no semantic reshape primitive is required for expression. |
| [expr_reshape_compose](../../tepl/rules/semantic/expr_reshape_compose.tepl) | transpose-reshape-reshape-compose | missing_semantics | reshape | Explicitly reuses the outer shape descriptor. |
| [expr_concat_unary](../../tepl/rules/semantic/expr_concat_unary.tepl) | indexing-concat-unary | missing_semantics | concatenate | Single operand case is expressible despite missing semantic concat reads. |
| [expr_concat_flatten](../../tepl/rules/semantic/expr_concat_flatten.tepl) | indexing-concat-flatten | missing_semantics | concatenate | Binary inner/outer instance; arbitrary operand-list capture is assessed separately. |
| [expr_update_full](../../tepl/rules/semantic/expr_update_full.tepl) | indexing-update-full | missing_semantics | dynamic_update_slice | Equal base/update shapes are expressed by shared shape variables. |
| [expr_update_empty](../../tepl/rules/semantic/expr_update_empty.tepl) | indexing-update-empty | missing_semantics | dynamic_update_slice | Rank-one empty update instance. |
| [expr_reverse_compose_general](../../tepl/rules/semantic/expr_reverse_compose_general.tepl) | indexing-reverse-compose | proved |  | General symmetric difference computed with existing list builtins; no auto. |
| [expr_pad_identity](../../tepl/rules/semantic/expr_pad_identity.tepl) | indexing-pad-identity | missing_semantics | pad | Arbitrary-rank zero-padding guard using existing comprehensions. |
| [expr_reduce_compose_general](../../tepl/rules/semantic/expr_reduce_compose_general.tepl) | reduction-compose | missing_semantics | reduce | Single-result same-combiner/same-initializer case; region handle forwarded, explicit sorted union of lifted axes. |
| [expr_slice_compose_general](../../tepl/rules/semantic/expr_slice_compose_general.tepl) | indexing-slice-compose | proved |  | Arbitrary-rank unit-stride composition; explicit metadata arithmetic and comprehensions. |
| [expr_slice_reverse_general](../../tepl/rules/semantic/expr_slice_reverse_general.tepl) | indexing-slice-reverse | not_proved |  | Arbitrary-rank positive-size strided slice; explicit reflected starts. Empty output cases can be handled separately. |
| [expr_convolution_scale_filter_bias](../../tepl/rules/semantic/expr_convolution_scale_filter_bias.tepl) | convolution-scale-filter-bias | not_proved |  | One spatial dimension with fixed kernel/output feature positions. Proves expression of explicit scale broadcasts and convolution attribute reuse; source constant and numerical-policy conditions remain in the catalog. |
| [catalog_elementwise_and_true](../../tepl/rules/semantic/catalog_elementwise_and_true.tepl) | elementwise-and-true | proved |  | Scalar literal instance; source conditions retained in the catalog. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [catalog_elementwise_or_true](../../tepl/rules/semantic/catalog_elementwise_or_true.tepl) | elementwise-or-true | proved |  | Scalar literal instance; source conditions retained in the catalog. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [catalog_elementwise_divide_exponentials](../../tepl/rules/semantic/catalog_elementwise_divide_exponentials.tepl) | elementwise-divide-exponentials | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_divide_exponential](../../tepl/rules/semantic/catalog_elementwise_divide_exponential.tepl) | elementwise-divide-exponential | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_divide_power](../../tepl/rules/semantic/catalog_elementwise_divide_power.tepl) | elementwise-divide-power | not_proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_divide_sqrt](../../tepl/rules/semantic/catalog_elementwise_divide_sqrt.tepl) | elementwise-divide-sqrt | not_proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_divide_rsqrt](../../tepl/rules/semantic/catalog_elementwise_divide_rsqrt.tepl) | elementwise-divide-rsqrt | not_proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_multiply_predicate](../../tepl/rules/semantic/catalog_elementwise_multiply_predicate.tepl) | elementwise-multiply-predicate | missing_semantics | convert | Written operand/opcode variant under the catalog conditions. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_log_power](../../tepl/rules/semantic/catalog_elementwise_log_power.tepl) | elementwise-log-power | not_proved |  | Scalar literal instance; source conditions retained in the catalog. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_log_rsqrt](../../tepl/rules/semantic/catalog_elementwise_log_rsqrt.tepl) | elementwise-log-rsqrt | not_proved |  | Scalar literal instance; source conditions retained in the catalog. |
| [catalog_elementwise_max_identity](../../tepl/rules/semantic/catalog_elementwise_max_identity.tepl) | elementwise-max-identity | proved |  | Scalar literal instance; source conditions retained in the catalog. |
| [catalog_elementwise_min_identity](../../tepl/rules/semantic/catalog_elementwise_min_identity.tepl) | elementwise-min-identity | proved |  | Scalar literal instance; source conditions retained in the catalog. |
| [catalog_elementwise_max_nested_right_reorder](../../tepl/rules/semantic/catalog_elementwise_max_nested_right_reorder.tepl) | elementwise-max-nested-right-reorder | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_min_nested_right_reorder](../../tepl/rules/semantic/catalog_elementwise_min_nested_right_reorder.tepl) | elementwise-min-nested-right-reorder | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_max_clamp_bound](../../tepl/rules/semantic/catalog_elementwise_max_clamp_bound.tepl) | elementwise-max-clamp-bound | not_proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_max_monotone](../../tepl/rules/semantic/catalog_elementwise_max_monotone.tepl) | elementwise-max-monotone | proved |  | tanh member of the finite monotone-opcode family. |
| [catalog_elementwise_clamp_clamp](../../tepl/rules/semantic/catalog_elementwise_clamp_clamp.tepl) | elementwise-clamp-clamp | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_real_noncomplex](../../tepl/rules/semantic/catalog_elementwise_real_noncomplex.tepl) | elementwise-real-noncomplex | missing_semantics | real | Noncomplex f32 inputs of arbitrary rank. |
| [catalog_elementwise_imag_noncomplex](../../tepl/rules/semantic/catalog_elementwise_imag_noncomplex.tepl) | elementwise-imag-noncomplex | missing_semantics | imag | Noncomplex f32 inputs of arbitrary rank. |
| [catalog_elementwise_power_exp](../../tepl/rules/semantic/catalog_elementwise_power_exp.tepl) | elementwise-power-exp | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_sqrt_square](../../tepl/rules/semantic/catalog_elementwise_sqrt_square.tepl) | elementwise-sqrt-square | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_compare_self](../../tepl/rules/semantic/catalog_elementwise_compare_self.tepl) | elementwise-compare-self | proved |  | Integral scalar EQ/GE/LE cases; false-result directions can use a separate rule. |
| [catalog_elementwise_compare_predicate_false](../../tepl/rules/semantic/catalog_elementwise_compare_predicate_false.tepl) | elementwise-compare-predicate-false | proved |  | Scalar literal instance; source conditions retained in the catalog. |
| [catalog_elementwise_compare_predicate_true](../../tepl/rules/semantic/catalog_elementwise_compare_predicate_true.tepl) | elementwise-compare-predicate-true | proved |  | Scalar literal instance; source conditions retained in the catalog. |
| [catalog_elementwise_compare_selected_mask](../../tepl/rules/semantic/catalog_elementwise_compare_selected_mask.tepl) | elementwise-compare-selected-mask | proved |  | Scalar literal instance; source conditions retained in the catalog. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_compare_max](../../tepl/rules/semantic/catalog_elementwise_compare_max.tepl) | elementwise-compare-max | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_compare_min](../../tepl/rules/semantic/catalog_elementwise_compare_min.tepl) | elementwise-compare-min | proved |  | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_select_predicate_identity](../../tepl/rules/semantic/catalog_elementwise_select_predicate_identity.tepl) | elementwise-select-predicate-identity | proved |  | Scalar literal instance; source conditions retained in the catalog. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_select_predicate_not](../../tepl/rules/semantic/catalog_elementwise_select_predicate_not.tepl) | elementwise-select-predicate-not | proved |  | Scalar literal instance; source conditions retained in the catalog. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_select_compare_gt](../../tepl/rules/semantic/catalog_elementwise_select_compare_gt.tepl) | elementwise-select-compare-gt | not_proved |  | Written operand/opcode variant under the catalog conditions. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_select_compare_lt](../../tepl/rules/semantic/catalog_elementwise_select_compare_lt.tepl) | elementwise-select-compare-lt | not_proved |  | Written operand/opcode variant under the catalog conditions. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_select_compare_eq](../../tepl/rules/semantic/catalog_elementwise_select_compare_eq.tepl) | elementwise-select-compare-eq | proved |  | Written operand/opcode variant under the catalog conditions. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_select_compare_ne](../../tepl/rules/semantic/catalog_elementwise_select_compare_ne.tepl) | elementwise-select-compare-ne | proved |  | Written operand/opcode variant under the catalog conditions. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_divide_converted_predicate](../../tepl/rules/semantic/catalog_elementwise_divide_converted_predicate.tepl) | elementwise-divide-converted-predicate | missing_semantics | convert | Written operand/opcode variant under the catalog conditions. |
| [catalog_broadcast_divide_broadcast](../../tepl/rules/semantic/catalog_broadcast_divide_broadcast.tepl) | broadcast-divide-broadcast | not_proved |  | Scalar Y broadcast to an arbitrary output shape. |
| [catalog_broadcast_multiply_constant_into_broadcast](../../tepl/rules/semantic/catalog_broadcast_multiply_constant_into_broadcast.tepl) | broadcast-multiply-constant-into-broadcast | proved |  | Scalar factor C is a captured scalar; literal-only profitability does not affect the identity. |
| [catalog_broadcast_reshape](../../tepl/rules/semantic/catalog_broadcast_reshape.tepl) | broadcast-reshape | missing_semantics | reshape | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_broadcast_transpose](../../tepl/rules/semantic/catalog_broadcast_transpose.tepl) | broadcast-transpose | not_proved |  | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_broadcast_inserted_unit_dims](../../tepl/rules/semantic/catalog_broadcast_inserted_unit_dims.tepl) | broadcast-inserted-unit-dims | missing_semantics | reshape | Rank-one X; reshape inserts a leading unit axis; the outer broadcast appends axes. |
| [catalog_broadcast_iota](../../tepl/rules/semantic/catalog_broadcast_iota.tepl) | broadcast-iota | missing_semantics | iota | Arbitrary-rank iota; broadcast appends axes. |
| [catalog_broadcast_drop_unit_dims](../../tepl/rules/semantic/catalog_broadcast_drop_unit_dims.tepl) | broadcast-drop-unit-dims | missing_semantics | reshape | Rank-two operand with leading unit axis; broadcast appends axes. |
| [catalog_broadcast_reshape_broadcast](../../tepl/rules/semantic/catalog_broadcast_reshape_broadcast.tepl) | broadcast-reshape-broadcast | missing_semantics | reshape | Rank-one base; reshape merges only two broadcast-created trailing axes. |
| [catalog_broadcast_transpose_broadcast](../../tepl/rules/semantic/catalog_broadcast_transpose_broadcast.tepl) | broadcast-transpose-broadcast | not_proved |  | General insertion-only broadcast/permutation equation; source sorted-map/layout policy remains in the catalog. |
| [catalog_broadcast_pad_broadcast](../../tepl/rules/semantic/catalog_broadcast_pad_broadcast.tepl) | broadcast-pad-broadcast | missing_semantics | nonnegative_pad | Rank-one X broadcast to rank two; padding only affects the mapped first axis. Edge padding is nonnegative. |
| [catalog_broadcast_slice_broadcast_only](../../tepl/rules/semantic/catalog_broadcast_slice_broadcast_only.tepl) | broadcast-slice-broadcast-only | not_proved |  | Rank-one X maps to axis zero; only newly created axes are sliced. |
| [catalog_broadcast_slice_broadcast](../../tepl/rules/semantic/catalog_broadcast_slice_broadcast.tepl) | broadcast-slice-broadcast | not_proved |  | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_broadcast_dynamic_slice_broadcast](../../tepl/rules/semantic/catalog_broadcast_dynamic_slice_broadcast.tepl) | broadcast-dynamic-slice-broadcast | missing_semantics | dynamic_slice | Rank-one base broadcast to rank two; two scalar dynamic start operands. |
| [catalog_broadcast_gather_scalar](../../tepl/rules/semantic/catalog_broadcast_gather_scalar.tepl) | broadcast-gather-scalar | missing_semantics | gather, reshape | Effective-scalar operand; result shape is adapter metadata on gather. |
| [catalog_transpose_reshape_reshape_empty](../../tepl/rules/semantic/catalog_transpose_reshape_reshape_empty.tepl) | transpose-reshape-reshape-empty | missing_semantics | fill, reshape | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_transpose_reshape_reshape_transpose_cancel_chain](../../tepl/rules/semantic/catalog_transpose_reshape_reshape_transpose_cancel_chain.tepl) | transpose-reshape-reshape-transpose-cancel-chain | missing_semantics | reshape | A reshape/transpose/reshape chain moving a single unit axis. |
| [catalog_transpose_reshape_reshape_hoist_transpose](../../tepl/rules/semantic/catalog_transpose_reshape_reshape_hoist_transpose.tepl) | transpose-reshape-reshape-hoist-transpose | missing_semantics | reshape | Two contiguous rank-two chunks of a rank-four tensor; symbolic extents. |
| [catalog_transpose_reshape_transpose_unit_dims](../../tepl/rules/semantic/catalog_transpose_reshape_transpose_unit_dims.tepl) | transpose-reshape-transpose-unit-dims | missing_semantics | reshape | Rank-two input with one unit axis. |
| [catalog_indexing_add_sparse_update](../../tepl/rules/semantic/catalog_indexing_add_sparse_update.tepl) | indexing-add-sparse-update | missing_semantics | dynamic_slice, dynamic_update_slice | Rank-one update/slice with a scalar start. |
| [catalog_indexing_add_scatter](../../tepl/rules/semantic/catalog_indexing_add_scatter.tepl) | indexing-add-scatter | missing_semantics | scatter_add | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_indexing_add_two_scatters_concat](../../tepl/rules/semantic/catalog_indexing_add_two_scatters_concat.tepl) | indexing-add-two-scatters-concat | missing_semantics | concatenate, scatter_add | Rank-one point-scatter operands with [count,1] indices and no batching. |
| [catalog_indexing_add_two_scatters_nested](../../tepl/rules/semantic/catalog_indexing_add_two_scatters_nested.tepl) | indexing-add-two-scatters-nested | missing_semantics | scatter_add | Single-operand unbatched scatter-add composition; source shape screening remains applicable. |
| [catalog_indexing_select_update](../../tepl/rules/semantic/catalog_indexing_select_update.tepl) | indexing-select-update | missing_semantics | dynamic_slice, dynamic_update_slice | Rank-one base, predicate and update; one scalar start. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_indexing_reshape_update](../../tepl/rules/semantic/catalog_indexing_reshape_update.tepl) | indexing-reshape-update | missing_semantics | dynamic_update_slice, reshape | Insert a leading unit dimension into a rank-one dynamic update. |
| [catalog_indexing_concat_adjacent_slices](../../tepl/rules/semantic/catalog_indexing_concat_adjacent_slices.tepl) | indexing-concat-adjacent-slices | missing_semantics | concatenate | Adjacent rank-one unit-stride slices. |
| [catalog_indexing_concat_splat_to_pad](../../tepl/rules/semantic/catalog_indexing_concat_splat_to_pad.tepl) | indexing-concat-splat-to-pad | missing_semantics | concatenate, nonnegative_pad | Rank-one low-edge scalar padding; high-edge form is symmetric. Edge padding is nonnegative. |
| [catalog_indexing_iota_trivial](../../tepl/rules/semantic/catalog_indexing_iota_trivial.tepl) | indexing-iota-trivial | missing_semantics | fill, iota | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_indexing_pad_empty](../../tepl/rules/semantic/catalog_indexing_pad_empty.tepl) | indexing-pad-empty | missing_semantics | nonnegative_pad | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. Edge padding is nonnegative. |
| [catalog_indexing_pad_interior_unit](../../tepl/rules/semantic/catalog_indexing_pad_interior_unit.tepl) | indexing-pad-interior-unit | missing_semantics | nonnegative_pad | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. Edge padding is nonnegative. |
| [catalog_indexing_reverse_trivial](../../tepl/rules/semantic/catalog_indexing_reverse_trivial.tepl) | indexing-reverse-trivial | proved |  | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_indexing_reverse_inserted_unit_dims](../../tepl/rules/semantic/catalog_indexing_reverse_inserted_unit_dims.tepl) | indexing-reverse-inserted-unit-dims | missing_semantics | reshape | Insert one leading unit axis into rank-one input. |
| [catalog_indexing_rotate_compose](../../tepl/rules/semantic/catalog_indexing_rotate_compose.tepl) | indexing-rotate-compose | missing_semantics | rotate | Arbitrary-rank rotation over positive-extent selected axes; shifts combined modulo extent. Shift metadata is nonnegative. |
| [catalog_indexing_slice_pad_only](../../tepl/rules/semantic/catalog_indexing_slice_pad_only.tepl) | indexing-slice-pad-only | missing_semantics | nonnegative_pad | Rank-one edge-padding-only slice, with arbitrary positive stride. Edge padding is nonnegative. |
| [catalog_indexing_slice_pad_cancel](../../tepl/rules/semantic/catalog_indexing_slice_pad_cancel.tepl) | indexing-slice-pad-cancel | missing_semantics | nonnegative_pad | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. Edge padding is nonnegative. |
| [catalog_indexing_slice_inside_pad](../../tepl/rules/semantic/catalog_indexing_slice_inside_pad.tepl) | indexing-slice-inside-pad | missing_semantics | nonnegative_pad | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. Edge padding is nonnegative. |
| [catalog_indexing_slice_reshape](../../tepl/rules/semantic/catalog_indexing_slice_reshape.tepl) | indexing-slice-reshape | missing_semantics | reshape | Rank-two input flattened to rank one; slice selects complete rows. |
| [catalog_indexing_slice_singleton_stride](../../tepl/rules/semantic/catalog_indexing_slice_singleton_stride.tepl) | indexing-slice-singleton-stride | not_proved |  | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_indexing_dynamic_slice_identity](../../tepl/rules/semantic/catalog_indexing_dynamic_slice_identity.tepl) | indexing-dynamic-slice-identity | missing_semantics | dynamic_slice | Rank-one full-shape dynamic slice. |
| [catalog_indexing_dynamic_slice_zero_index](../../tepl/rules/semantic/catalog_indexing_dynamic_slice_zero_index.tepl) | indexing-dynamic-slice-zero-index | missing_semantics | dynamic_slice | Rank-two dynamic slice; first size covers its entire operand axis. |
| [catalog_indexing_dynamic_slice_unit_reshape](../../tepl/rules/semantic/catalog_indexing_dynamic_slice_unit_reshape.tepl) | indexing-dynamic-slice-unit-reshape | missing_semantics | dynamic_slice, reshape | Rank-one base with one inserted leading unit axis. |
| [catalog_indexing_dynamic_slice_transpose](../../tepl/rules/semantic/catalog_indexing_dynamic_slice_transpose.tepl) | indexing-dynamic-slice-transpose | missing_semantics | dynamic_slice | Rank-two transpose with two scalar start operands. |
| [catalog_indexing_dynamic_slice_iota](../../tepl/rules/semantic/catalog_indexing_dynamic_slice_iota.tepl) | indexing-dynamic-slice-iota | missing_semantics | convert, dynamic_slice, iota, reshape, scalar | Rank-one, scalar-sized slice, i32 starts, optional scalar factor captured as Scale. |
| [catalog_indexing_dynamic_slice_compose](../../tepl/rules/semantic/catalog_indexing_dynamic_slice_compose.tepl) | indexing-dynamic-slice-compose | missing_semantics | dynamic_slice, scalar | Rank-one scalar i64 starts; explicit independent clamping before addition. |
| [catalog_indexing_update_zero_index](../../tepl/rules/semantic/catalog_indexing_update_zero_index.tepl) | indexing-update-zero-index | missing_semantics | dynamic_update_slice | Rank-one, full-shape update; start becomes zero. |
| [catalog_indexing_update_compose](../../tepl/rules/semantic/catalog_indexing_update_compose.tepl) | indexing-update-compose | missing_semantics | dynamic_slice, dynamic_update_slice, scalar | Rank-one scalar i64 starts; explicit independently clamped composition. |
| [catalog_indexing_gather_empty](../../tepl/rules/semantic/catalog_indexing_gather_empty.tepl) | indexing-gather-empty | missing_semantics | fill, gather | Empty input; gather result shape is adapter metadata. |
| [catalog_indexing_gather_pad](../../tepl/rules/semantic/catalog_indexing_gather_pad.tepl) | indexing-gather-pad | missing_semantics | gather, nonnegative_pad | Rank-two gather over rows; edge padding only on the passthrough column axis. Edge padding is nonnegative. |
| [catalog_indexing_gather_reshape_pad](../../tepl/rules/semantic/catalog_indexing_gather_reshape_pad.tepl) | indexing-gather-reshape-pad | missing_semantics | gather, nonnegative_pad, reshape | Rank-two input, inserted leading unit axis, padded passthrough column axis. Edge padding is nonnegative. |
| [catalog_indexing_scatter_empty_update](../../tepl/rules/semantic/catalog_indexing_scatter_empty_update.tepl) | indexing-scatter-empty-update | missing_semantics | scatter | Single-operand/result scatter; arbitrary combiner forwarded by source descriptor. |
| [catalog_indexing_scatter_scalar_map](../../tepl/rules/semantic/catalog_indexing_scatter_scalar_map.tepl) | indexing-scatter-scalar-map | missing_semantics | map, scatter | Single operand/result, empty scalar index vector, whole-tensor combining region. |
| [catalog_indexing_transpose_scatter](../../tepl/rules/semantic/catalog_indexing_transpose_scatter.tepl) | indexing-transpose-scatter | missing_semantics | scatter | Rank-two operand, rank-three updates with leading scatter axis, no batching. |
| [catalog_dot_empty](../../tepl/rules/semantic/catalog_dot_empty.tepl) | dot-empty | missing_semantics | fill | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_dot_drop_unit_dims](../../tepl/rules/semantic/catalog_dot_drop_unit_dims.tepl) | dot-drop-unit-dims | missing_semantics | reshape | Remove a leading free unit axis from a rank-three left operand. |
| [catalog_dot_transpose_both_operands](../../tepl/rules/semantic/catalog_dot_transpose_both_operands.tepl) | dot-transpose-both-operands | proved |  | Rank-two matrix transposes, no batches. |
| [catalog_dot_constant_dynamic_slice](../../tepl/rules/semantic/catalog_dot_constant_dynamic_slice.tepl) | dot-constant-dynamic-slice | missing_semantics | dynamic_slice | Rank-two dot with one selected left free-axis element. Captured C/D may be constants; the identity also holds for nonconstants. |
| [catalog_dot_move_contracting_permutation](../../tepl/rules/semantic/catalog_dot_move_contracting_permutation.tepl) | dot-move-contracting-permutation | missing_semantics | reshape | Swap two contraction subdimensions before flattening; rank-three left operand. |
| [catalog_dot_move_zero_pad](../../tepl/rules/semantic/catalog_dot_move_zero_pad.tepl) | dot-move-zero-pad | missing_semantics | nonnegative_pad | Rank-two dot; nonnegative padding only on its left contraction axis. |
| [catalog_dot_move_broadcast](../../tepl/rules/semantic/catalog_dot_move_broadcast.tepl) | dot-move-broadcast | proved |  | One broadcast-created contraction axis; replacement dot is an outer product. |
| [catalog_dot_slice_output](../../tepl/rules/semantic/catalog_dot_slice_output.tepl) | dot-slice-output | not_proved |  | Rank-two dot with arbitrary output starts, sizes and positive strides. |
| [catalog_dot_reduce_output_free](../../tepl/rules/semantic/catalog_dot_reduce_output_free.tepl) | dot-reduce-output-free | proved |  | Reduce the left free axis of a rank-two dot; implicit sum initializer zero. |
| [catalog_dot_scalar_multipliers](../../tepl/rules/semantic/catalog_dot_scalar_multipliers.tepl) | dot-scalar-multipliers | proved |  | Scalar factor pulled from a rank-two f32 dot; source constant/profitability screening is outside the algebraic identity. |
| [catalog_reduction_empty](../../tepl/rules/semantic/catalog_reduction_empty.tepl) | reduction-empty | missing_semantics | reduce | Single-output reduction with arbitrary initial value and empty input. |
| [catalog_reduction_unit_axes](../../tepl/rules/semantic/catalog_reduction_unit_axes.tepl) | reduction-unit-axes | missing_semantics | reshape | Zero-initialized sum specialization of the generic-combiner schema. |
| [catalog_reduction_reshape_reduced](../../tepl/rules/semantic/catalog_reduction_reshape_reduced.tepl) | reduction-reshape-reduced | missing_semantics | reshape | Sum over a merged pair of reduced axes; rank-three input. |
| [catalog_reduction_reshape_free](../../tepl/rules/semantic/catalog_reduction_reshape_free.tepl) | reduction-reshape-free | missing_semantics | reshape | Rank-three input; sum over first axis while reshape merges free axes. |
| [catalog_reduction_splat_min](../../tepl/rules/semantic/catalog_reduction_splat_min.tepl) | reduction-splat-min | missing_semantics | reduce_min | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_reduction_splat_max](../../tepl/rules/semantic/catalog_reduction_splat_max.tepl) | reduction-splat-max | missing_semantics | reduce_max | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_reduction_splat_and](../../tepl/rules/semantic/catalog_reduction_splat_and.tepl) | reduction-splat-and | missing_semantics | reduce_and | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [catalog_reduction_splat_or](../../tepl/rules/semantic/catalog_reduction_splat_or.tepl) | reduction-splat-or | missing_semantics | reduce_or | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. Local HLO logical operations use Boolean tensors; integer bitwise variants are not verified. |
| [catalog_reduction_broadcast_original_axes](../../tepl/rules/semantic/catalog_reduction_broadcast_original_axes.tepl) | reduction-broadcast-original-axes | proved |  | Sum specialization; rank-one base and one broadcast-created axis. |
| [catalog_reduction_broadcast_replication_axes](../../tepl/rules/semantic/catalog_reduction_broadcast_replication_axes.tepl) | reduction-broadcast-replication-axes | missing_semantics | scalar | Reduce all appended replication axes; zero-initialized sum, symbolic replication count. |
| [catalog_reduction_window_empty](../../tepl/rules/semantic/catalog_reduction_window_empty.tepl) | reduction-window-empty | missing_semantics | reduce_window | Single-output reduction window with nonnegative edge padding and empty input. |
| [catalog_reduction_window_scalar](../../tepl/rules/semantic/catalog_reduction_window_scalar.tepl) | reduction-window-scalar | missing_semantics | reduce_window_sum | Rank-zero zero-initialized sum window. |
| [catalog_reduction_window_splat_cumsum](../../tepl/rules/semantic/catalog_reduction_window_splat_cumsum.tepl) | reduction-window-splat-cumsum | missing_semantics | iota, reduce_window_sum | Rank-one cumulative sum of a scalar f32 broadcast; scalar constant can be a capture. |
| [catalog_reduction_window_singleton_add](../../tepl/rules/semantic/catalog_reduction_window_singleton_add.tepl) | reduction-window-singleton-add | missing_semantics | nonnegative_pad, reduce_window_sum | Unit windows and nonnegative edge padding; scalar initializer preserved. |
| [catalog_reduction_window_singleton_min](../../tepl/rules/semantic/catalog_reduction_window_singleton_min.tepl) | reduction-window-singleton-min | missing_semantics | nonnegative_pad, reduce_window_min | Unit windows and nonnegative edge padding; scalar initializer preserved. |
| [catalog_reduction_window_singleton_max](../../tepl/rules/semantic/catalog_reduction_window_singleton_max.tepl) | reduction-window-singleton-max | missing_semantics | nonnegative_pad, reduce_window_max | Unit windows and nonnegative edge padding; scalar initializer preserved. |
| [catalog_reduction_window_to_reduce](../../tepl/rules/semantic/catalog_reduction_window_to_reduce.tepl) | reduction-window-to-reduce | missing_semantics | reduce_window_sum, reshape | Arbitrary-rank f32 zero-initialized sum windows, each axis unit or full. |
| [catalog_reduction_slice_window_to_reduce](../../tepl/rules/semantic/catalog_reduction_slice_window_to_reduce.tepl) | reduction-slice-window-to-reduce | missing_semantics | reduce_window_sum, reshape | Last element of a rank-one full cumulative window; f32 sum. |
| [catalog_reduction_window_pad](../../tepl/rules/semantic/catalog_reduction_window_pad.tepl) | reduction-window-pad | missing_semantics | nonnegative_pad, reduce_window | Single-result window; nonnegative padding equals the initializer; reducer region forwarded. |
| [catalog_broadcast_window_padded_unit_broadcast](../../tepl/rules/semantic/catalog_broadcast_window_padded_unit_broadcast.tepl) | broadcast-window-padded-unit-broadcast | missing_semantics | nonnegative_pad, reduce_window_sum, reshape | Rank-one f32 unit input, symmetric padding, every window includes the sole unpadded value. |
| [catalog_convolution_empty](../../tepl/rules/semantic/catalog_convolution_empty.tepl) | convolution-empty | missing_semantics | fill, general_convolution | General convolution; result shape is ordinary source/adapter metadata. |
| [catalog_convolution_fold_input_pad](../../tepl/rules/semantic/catalog_convolution_fold_input_pad.tepl) | convolution-fold-input-pad | missing_semantics | general_convolution, nonnegative_pad | One canonical spatial axis; no existing input dilation or convolution edge padding. |
| [catalog_convolution_fold_filter_pad](../../tepl/rules/semantic/catalog_convolution_fold_filter_pad.tepl) | convolution-fold-filter-pad | missing_semantics | general_convolution, nonnegative_pad | One canonical spatial axis; filter interior padding becomes kernel dilation. |
| [catalog_convolution_pointwise_dot](../../tepl/rules/semantic/catalog_convolution_pointwise_dot.tepl) | convolution-pointwise-dot | missing_semantics | general_convolution, reshape | One-spatial-axis pointwise convolution in canonical layouts, no grouping or padding. |
| [catalog_convolution_depthwise_product](../../tepl/rules/semantic/catalog_convolution_depthwise_product.tepl) | convolution-depthwise-product | missing_semantics | general_convolution, reshape | Depthwise rank-three convolution whose kernel spans the full spatial extent. |
| [catalog_convolution_swap_operands](../../tepl/rules/semantic/catalog_convolution_swap_operands.tepl) | convolution-swap-operands | missing_semantics | general_convolution | One spatial axis, unit strides/dilations, nonnegative transformed padding; kernel reversal, role swap and precision swap are explicit. |
| [catalog_indexing_pad_negative](../../tepl/rules/semantic/catalog_indexing_pad_negative.tepl) | indexing-pad-negative | missing_semantics | pad | Rank-one low crop -1 and high padding +2; signed pad metadata retained and crop bounds use nonnegative shape arithmetic. |
| [native_power_two_splat](../../tepl/rules/semantic/native_power_two_splat.tepl) | elementwise-power-two | missing_semantics | fill | Arbitrary-rank numeric tensor and same-shape/type exponent splat represented by fill; where checks the value attribute equals 2. Does not inspect opaque dense literals. |
| [native_abs_unsigned](../../tepl/rules/semantic/native_abs_unsigned.tepl) | elementwise-abs-nonnegative | proved |  | Arbitrary-rank unsigned integer tensor; is_unsigned_integer establishes nonnegativity. Signed/floating range analysis is not implemented. |

## Reproduction

[verification.json](../results/verification.json) records diagnostics, required/missing operations, proof evidence and file/dependency hashes, including the semantic compiler sources and untracked additions. The revision and semantic source hashes identify the measured compiler. [verification_cases.json](../data/verification_cases.json) records scope and arithmetic policy. The validators reject stale inputs and missing or duplicate cases. [Language coverage](compilation.md) separately measures ordinary compilation using [the structural HLO profile](../../tepl/dialects/hlo_structural.tepl).

- TEPL revision: `145eb44f10ef4826575cfd1c15c1fdd28f66f8d5`.
- XLA revision: `e02127ba9c9c48912b0e6776fe5bfe3843d81042`.
- Compiler SHA-256: `af87e6453056559ec5319f39e40a90401910bf10120168cea7f5eba153e020e2`.
- Environment: Linux-7.0.0-38-generic-x86_64-with-glibc2.43; Python 3.14.4.
- Timeout: 30 seconds per case.

```sh
bazel build //:tepl
python3 labs/xla_rewrite/coverage/scripts/run_checks.py
python3 labs/xla_rewrite/coverage/scripts/expression_coverage.py --run
python3 labs/xla_rewrite/coverage/scripts/summarize_coverage.py
python3 labs/xla_rewrite/coverage/scripts/summarize_coverage.py --check
python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_*.py'
```
