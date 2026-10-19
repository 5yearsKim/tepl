# XLA Algebraic Rewrite Catalog

Generated from `rules.json` by `summarize_catalog.py`. See [README.md](README.md) for scope, notation, and counting policy.

XLA revision: `e02127ba9c9c48912b0e6776fe5bfe3843d81042`. These are source summaries, not TEPL verification results.

## Category counts

| Category | Schemas |
|---|---:|
| Elementwise | 99 |
| Transpose / reshape | 9 |
| Broadcast | 20 |
| Reduction | 26 |
| Dot | 18 |
| Convolution | 7 |
| Indexing | 60 |
| **Total** | **239** |

232 entries link related upstream tests; 7 are source-only. Links may include negative tests and do not establish that a schema was executed or proved.

## Elementwise

### elementwise-abs-nonnegative

```text
abs(X) -> X
```

- **Conditions:** IsNonNegative(X) succeeds under the source type/range analysis.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleAbs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L927-L935).
- **Related tests:** [AlgebraicSimplifierTest.AbsEliminationMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12220), [AlgebraicSimplifierTest.DoesNotFoldSignedIntegerAbsLtZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11299).
- **Note:** Signed abs is not assumed nonnegative when the minimum integer can occur.

### elementwise-add-zero

```text
X + 0 -> X (either operand order)
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.AddZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L334).

### elementwise-add-constant-order

```text
C + X -> X + C
```

- **Conditions:** C is a constant and X is nonconstant.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.AddConstOnLHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2141).

### elementwise-add-constants

```text
(X + C1) + C2 -> X + (C1 + C2)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation. Constants may be scalar broadcasts.
- **RHS attributes:** Preserve scalar broadcast to the result shape.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.AddReassociateMergeConstants](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2162), [AlgebraicSimplifierTest.AddReassociateMergeBroadcastedConstants](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2238).

### elementwise-subtract-add-constants

```text
(C1 - X) + C2 -> (C1 + C2) - X
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation. Constants may be scalar broadcasts.
- **RHS attributes:** Preserve scalar broadcast to the result shape.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.SubAddReassociateMergeConstants](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2322).

### elementwise-add-subtract-cancel

```text
X + (Y - X) -> Y (either outer operand order)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.AddSubSame](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L354), [AlgebraicSimplifierTest.AddSubSameInt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L396).

### elementwise-factor-common-product

```text
X*C + Y*C -> (X + Y)*C (matched factor orders)
```

- **Conditions:** Integral type, enable_floats_are_real, or scalar/broadcast floating C equal to a signed power of two.
- **RHS attributes:** No new axis or indexing attributes.
- **Source policy:** Both products have one user.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.FactorIntegerAddition](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L470), [AlgebraicSimplifierTest.FactorFpAddition](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L492), [AlgebraicSimplifierTest.FactorFpAdditionNotPowerOf2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L845).
- **Note:** The power-of-two floating case permits intermediate rounding/underflow/overflow differences in the source; not an unconditional IEEE identity.

### elementwise-and-true

```text
X and true -> X (either operand order)
```

- **Conditions:** Predicate operands.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleAnd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L1417-L1460).
- **Related tests:** [AlgebraicSimplifierTest.AndTrue](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6474), [AlgebraicSimplifierTest.AndTrue2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6495).

### elementwise-and-zero

```text
X and 0 -> 0 (either operand order)
```

- **Conditions:** Predicate or bitwise integer and.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleAnd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L1417-L1460).
- **Related tests:** [AlgebraicSimplifierTest.AndFalse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6516), [AlgebraicSimplifierTest.AndFalse2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6537).

### elementwise-and-upper-bounds

```text
(X < C1) and (X < C2) -> X < min(C1,C2)
```

- **Conditions:** Same X; constants are effective-scalar S32. Reversed greater-than comparisons are also matched.
- **RHS attributes:** New scalar bound is min(C1,C2); comparison direction LT.
- **Source:** [TrySimplifyTautologicalCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L1275-L1317).
- **Related tests:** [AlgebraicSimplifierTest.CompareSimplified](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11441), [AlgebraicSimplifierTest.CompareSimplifiedReversed](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11460).

### elementwise-subtract-self

```text
X - X -> 0
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSubtract](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2368-L2440).
- **Related tests:** [AlgebraicSimplifierTest.ReplaceSubtractOfEqualOperandsWithZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2287).

### elementwise-subtract-zero

```text
X - 0 -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSubtract](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2368-L2440).
- **Related tests:** [AlgebraicSimplifierTest.SubZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2544).

### elementwise-subtract-constant

```text
X - C -> X + (-C)
```

- **Conditions:** X nonconstant; C constant or broadcast constant.
- **RHS attributes:** Preserve broadcast mapping for -C.
- **Source:** [HandleSubtract](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2368-L2440).
- **Related tests:** [AlgebraicSimplifierTest.SubConstCanonicalization](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2565), [AlgebraicSimplifierTest.SubBroadcastConstCanonicalization](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2587).

### elementwise-subtract-subtract-cancel

```text
X - (X - Y) -> Y
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSubtract](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2368-L2440).
- **Related tests:** [AlgebraicSimplifierTest.SubSubSame](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L413).

### elementwise-subtract-add-cancel

```text
X - (X + Y) -> -Y (either inner operand order)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSubtract](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2368-L2440).
- **Related tests:** [AlgebraicSimplifierTest.SubAddSame](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L432).

### elementwise-add-subtract-cancel-right

```text
(X + Y) - X -> Y (either inner operand order)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSubtract](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2368-L2440).
- **Related tests:** [AlgebraicSimplifierTest.AddSubSame2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L451).

### elementwise-divide-one

```text
X / 1 -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.DivOneScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2967), [AlgebraicSimplifierTest.DivOneArray](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2988).

### elementwise-divide-power-two-unsigned

```text
X / 2^k -> logical_shift_right(X,k)
```

- **Conditions:** Unsigned integer X; constant effective-scalar divisor is a nonzero power of two.
- **RHS attributes:** k=log2(divisor).
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702), [TryDivideToShift](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2450-L2503).
- **Related tests:** [AlgebraicSimplifierTest.UnsignedDivideByPowerOf2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L901).
- **Note:** Implementation in free helper TryDivideToShift, lines 2465–2503.

### elementwise-divide-power-two-signed

```text
X / 2^k -> select(X<0, -logical_shift_right(select(X<0,-X,X),k), logical_shift_right(select(X<0,-X,X),k))
```

- **Conditions:** Signed integer X; positive constant divisor 2^k; fixed-width negation and logical shift preserve truncation toward zero.
- **RHS attributes:** k=log2(divisor).
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702), [TryDivideToShift](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2450-L2503).
- **Related tests:** [AlgebraicSimplifierTest.SignedDivideByPowerOf2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L918).
- **Note:** Implementation in free helper TryDivideToShift, lines 2465–2503; do not replace with an arithmetic shift.

### elementwise-divide-exponentials

```text
exp(X) / exp(Y) -> exp(X-Y)
```

- **Conditions:** enable_fast_math.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.ExpDiv](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3108), [AlgebraicSimplifierTest.DivOfExp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2759).

### elementwise-divide-exponential

```text
X / exp(Y) -> X * exp(-Y)
```

- **Conditions:** enable_fast_math.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.DivOfExp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2759).

### elementwise-divide-power

```text
X / pow(Y,Z) -> X * pow(Y,-Z)
```

- **Conditions:** Non-integral result type; this source branch has no fast-math gate.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.DivOfPower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2795), [AlgebraicSimplifierTest.IntegerDivOfPower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2828).
- **Note:** Preserves the source numerical contract; not claimed exact for arbitrary floating/complex inputs.

### elementwise-divide-sqrt

```text
X / sqrt(Y) -> X * rsqrt(Y)
```

- **Conditions:** The denominator operation has one user.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.RecipSqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10847).
- **Note:** Source numerical transformation; no exact IEEE guarantee is inferred here.

### elementwise-divide-rsqrt

```text
X / rsqrt(Y) -> X * sqrt(Y)
```

- **Conditions:** The denominator operation has one user.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.RecipRsqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10865).
- **Note:** Source numerical transformation; no exact IEEE guarantee is inferred here.

### elementwise-divide-constant

```text
X / C -> X * reciprocal(C)
```

- **Conditions:** Floating or complex type; X nonconstant; C literal or broadcast literal.
- **RHS attributes:** Evaluate reciprocal literals in the source element type; preserve broadcast mapping.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.DivideByConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2883), [AlgebraicSimplifierTest.DivideByBroadcastedConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2905).

### elementwise-divide-two-quotients

```text
(A/B)/(C/D) -> (A*D)/(B*C)
```

- **Conditions:** Non-integral type; no local fast-math gate in the source.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.DivOfDivAndDiv](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2723).
- **Note:** Floating/complex exceptional values and rounding follow the source optimization contract.

### elementwise-divide-left-quotient

```text
(A/B)/C -> A/(B*C)
```

- **Conditions:** Non-integral type; no local fast-math gate in the source.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.LhsDivOfDiv](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2660).

### elementwise-divide-right-quotient

```text
A/(B/C) -> (A*C)/B
```

- **Conditions:** Non-integral type; no local fast-math gate in the source.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.RhsDivOfDiv](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2691).

### elementwise-multiply-one

```text
X * 1 -> X (either operand order)
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplyChain](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L663).

### elementwise-multiply-zero

```text
X * 0 -> 0 (either operand order)
```

- **Conditions:** Integral type or enable_fast_math.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MulZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L982), [AlgebraicSimplifierTest.FastMulZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1001).

### elementwise-multiply-negatives

```text
(-X)*(-Y) -> X*Y
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplyNegateNegate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12151).

### elementwise-square-abs

```text
abs(X)*abs(X) -> X*X
```

- **Conditions:** Same abs node; X is noncomplex.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.SquareOfAbs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L647).

### elementwise-multiply-predicate

```text
convert(P) * X -> select(P,X,0) (either operand order)
```

- **Conditions:** P has predicate type; legal conversion and matching result shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplyOfConvertedPred](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14065).
- **Note:** The source has no finite-X gate; do not treat this as an exact floating identity for NaN/Inf without a semantic restriction.

### elementwise-multiply-constant-pairs

```text
(X*C1)*(Y*C2) -> (X*Y)*(C1*C2)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation. C1,C2 literals.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplyChain](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L663), [AlgebraicSimplifierTest.MultiplyChain2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L692).

### elementwise-multiply-constants

```text
(X*C1)*C2 -> X*(C1*C2)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation. Literals and scalar-broadcast literals are matched.
- **RHS attributes:** Preserve scalar broadcasts.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplyReassociateMergeConstants](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1022), [AlgebraicSimplifierTest.MultiplyReassociateMergeBroadcastedConstants](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1047).
- **Note:** Overlapping literal and broadcast implementations are counted once.

### elementwise-multiply-exponentials

```text
exp(X)*exp(Y) -> exp(X+Y)
```

- **Conditions:** enable_fast_math.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.ExpMul](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3148).

### elementwise-square-sqrt

```text
sqrt(X)*sqrt(X) -> X
```

- **Conditions:** Same captured X; IsNonNegative(X).
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplySelfSqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12053), [AlgebraicSimplifierTest.MultiplySelfSqrt_NegativeTestCase](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12088).

### elementwise-square-rsqrt

```text
rsqrt(X)*rsqrt(X) -> 1/X
```

- **Conditions:** Same captured X; IsNonNegative(X).
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplySelfRsqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12102), [AlgebraicSimplifierTest.MultiplySelfRsqrt_NegativeTestCase](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12138).

### elementwise-double-negate

```text
-(-X) -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleNegate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5295-L5303).
- **Related tests:** [AlgebraicSimplifierTest.NegateNegate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9108).

### elementwise-double-not

```text
not(not(X)) -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleNot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5305-L5314).
- **Related tests:** [AlgebraicSimplifierTest.NotNot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9128).

### elementwise-or-true

```text
X or true -> true (either operand order)
```

- **Conditions:** Predicate operands.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleOr](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5316-L5350).
- **Related tests:** [AlgebraicSimplifierTest.OrTrue](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6558), [AlgebraicSimplifierTest.OrTrue2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6579).

### elementwise-or-zero

```text
X or 0 -> X (either operand order)
```

- **Conditions:** Predicate or bitwise integer or.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleOr](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5316-L5350).
- **Related tests:** [AlgebraicSimplifierTest.OrFalse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6600), [AlgebraicSimplifierTest.OrFalse2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6621).

### elementwise-xor-self

```text
X xor X -> 0
```

- **Conditions:** Same node, or structurally identical side-effect-free nodes under layout/sharding-sensitive comparison.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleXor](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5352-L5385).
- **Related tests:** [AlgebraicSimplifierTest.XorSamePred](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6642), [AlgebraicSimplifierTest.XorSameInt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6657), [AlgebraicSimplifierTest.XorIdenticalOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6673), [AlgebraicSimplifierTest.XorOfRngsNotSimplified](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6693).

### elementwise-xor-zero

```text
X xor 0 -> X (either operand order)
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleXor](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5352-L5385).
- **Related tests:** [AlgebraicSimplifierTest.XorFalse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6709), [AlgebraicSimplifierTest.XorFalse2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6726).

### elementwise-log-exp

```text
log(exp(X)) -> X
```

- **Conditions:** enable_fast_math.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleLog](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5387-L5436).
- **Related tests:** [AlgebraicSimplifierTest.LnExp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3325), [AlgebraicSimplifierTest.LnExpNotFoldedByDefault](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3350).

### elementwise-log-power

```text
log(pow(X,Y)) -> select(Y==0,0,Y*log(abs(X)))
```

- **Conditions:** enable_fast_math; for complex X use X instead of abs(X).
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleLog](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5387-L5436).
- **Related tests:** [AlgebraicSimplifierTest.LnPow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3223), [AlgebraicSimplifierTest.LnPowNotFoldedByDefault](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3254).

### elementwise-log-sqrt

```text
log(sqrt(X)) -> 0.5*log(X)
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleLog](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5387-L5436).
- **Related tests:** [AlgebraicSimplifierTest.LnSqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3276).
- **Note:** No local fast-math gate; source numeric behavior is retained.

### elementwise-log-rsqrt

```text
log(rsqrt(X)) -> -0.5*log(X)
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleLog](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5387-L5436).
- **Related tests:** [AlgebraicSimplifierTest.LnRsqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3300).

### elementwise-max-self

```text
max(X,X) -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.TrivialMax](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14962).

### elementwise-max-identity

```text
max(X,identity) -> X (either operand order)
```

- **Conditions:** Identity is minimum representable integer or -infinity; integral type or floating type with minmax_propagate_nan; surviving operand type equals output type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.MaximumWithNegativeInfinityLhs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4082).

### elementwise-max-nested-left

```text
max(max(X,Y),X or Y) -> max(X,Y)
```

- **Conditions:** Preserve the existing inner operand order and NaN behavior.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.MaximumOfMaximum1](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4146).

### elementwise-max-nested-right

```text
max(X,max(X,Y)) -> max(X,Y)
```

- **Conditions:** Exact matched operand order.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.MaximumOfMaximum3](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4182).

### elementwise-max-nested-right-reorder

```text
max(Y,max(X,Y)) -> max(Y,X)
```

- **Conditions:** Retain the indicated order for NaN sign behavior.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.MaximumOfMaximum4](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4200).

### elementwise-min-self

```text
min(X,X) -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMinimum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4912-L4979).
- **Related tests:** [AlgebraicSimplifierTest.TrivialMin](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14949).

### elementwise-min-identity

```text
min(X,identity) -> X (either operand order)
```

- **Conditions:** Identity is maximum representable integer or +infinity; integral type or floating type with minmax_propagate_nan; surviving operand type equals output type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMinimum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4912-L4979).
- **Related tests:** [AlgebraicSimplifierTest.MinimumWithInfinityLhs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3946).

### elementwise-min-nested-left

```text
min(min(X,Y),X or Y) -> min(X,Y)
```

- **Conditions:** Preserve the existing inner operand order and NaN behavior.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMinimum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4912-L4979).
- **Related tests:** [AlgebraicSimplifierTest.MinimumOfMinimum1](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4010).

### elementwise-min-nested-right

```text
min(X,min(X,Y)) -> min(X,Y)
```

- **Conditions:** Exact matched operand order.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMinimum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4912-L4979).
- **Related tests:** [AlgebraicSimplifierTest.MinimumOfMinimum3](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4046).

### elementwise-min-nested-right-reorder

```text
min(Y,min(X,Y)) -> min(Y,X)
```

- **Conditions:** Retain the indicated order for NaN sign behavior.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMinimum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4912-L4979).
- **Related tests:** [AlgebraicSimplifierTest.MinimumOfMinimum4](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4064).

### elementwise-max-min-clamp

```text
max(lo,min(X,hi)) -> clamp(lo,X,hi) (matched commuted forms)
```

- **Conditions:** lo and hi are broadcasts of effective-scalar constants; evaluator proves lo < hi.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.MaxOfMinToClamp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11786).

### elementwise-min-max-clamp

```text
min(hi,max(X,lo)) -> clamp(lo,X,hi) (matched commuted forms)
```

- **Conditions:** lo and hi are broadcasts of effective-scalar constants; evaluator proves lo < hi.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMinimum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4912-L4979).
- **Related tests:** [AlgebraicSimplifierTest.MinOfMaxToClamp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11765).

### elementwise-max-clamp-bound

```text
max(lo,clamp(lo,X,hi)) -> clamp(lo,X,hi) (either outer operand order)
```

- **Conditions:** Identical lower-bound nodes; compatible result.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.MaxOfClamp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11825).

### elementwise-max-monotone

```text
max(f(X),f(Y)) -> f(max(X,Y))
```

- **Conditions:** Same unary opcode f in {cbrt,erf,logistic,tanh}; source classifies f as nondecreasing/sublinear.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleMaximum](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4801-L4910).
- **Related tests:** [AlgebraicSimplifierTest.SinkCbrtThroughMax](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14757).

### elementwise-clamp-clamp

```text
clamp(lo,clamp(lo,X,hi),hi) -> clamp(lo,X,hi)
```

- **Conditions:** Same bound nodes in both clamps.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleClamp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4981-L5016).
- **Related tests:** [AlgebraicSimplifierTest.ClampOfClamp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11807).

### elementwise-exp-zero

```text
exp(0) -> 1
```

- **Conditions:** Zero scalar literal or scalar-broadcast literal.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleExp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6019-L6027).
- **Related tests:** [AlgebraicSimplifierTest.ExpOfZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2347).

### elementwise-complex-components

```text
complex(real(X),imag(X)) -> X
```

- **Conditions:** Same captured complex X.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleComplex](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6030-L6038).
- **Related tests:** [AlgebraicSimplifierTest.ComplexOfRealImagC](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3009).

### elementwise-real-complex

```text
real(complex(X,Y)) -> X
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleReal](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6041-L6051).
- **Related tests:** [AlgebraicSimplifierTest.RealOfComplex](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3033).

### elementwise-real-noncomplex

```text
real(X) -> X
```

- **Conditions:** X is noncomplex.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleReal](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6041-L6051).
- **Related tests:** [AlgebraicSimplifierTest.RealWithNonComplexInput](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15228).

### elementwise-imag-complex

```text
imag(complex(X,Y)) -> Y
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleImag](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6054-L6065).
- **Related tests:** [AlgebraicSimplifierTest.ImagOfComplex](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3057).

### elementwise-imag-noncomplex

```text
imag(X) -> zeros(shape(X))
```

- **Conditions:** X is noncomplex.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleImag](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6054-L6065).
- **Related tests:** [AlgebraicSimplifierTest.ImagWithNonComplexInput](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15246).

### elementwise-power-zero

```text
pow(X,0) -> 1
```

- **Conditions:** Exponent all zero.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.Pow0Scalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3406).

### elementwise-power-one

```text
pow(X,1) -> X
```

- **Conditions:** Exponent all one.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.Pow1](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3460).

### elementwise-power-exp

```text
pow(exp(X),Y) -> exp(X*Y)
```

- **Conditions:** enable_fast_math.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.PowExp](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3187).

### elementwise-power-two

```text
pow(X,2) -> X*X
```

- **Conditions:** Exponent all two.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.Pow2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3483).

### elementwise-power-three

```text
pow(X,3) -> X*(X*X)
```

- **Conditions:** Exponent all three.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.Pow3](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3507).

### elementwise-power-reciprocal

```text
pow(X,-1) -> 1/X
```

- **Conditions:** Non-integral type; exponent all -1.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.PowNegative1](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3533).

### elementwise-power-half

```text
pow(X,0.5) -> sqrt(X)
```

- **Conditions:** Floating exponent all 0.5; IsNonNegative(X).
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePower](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6285-L6342).
- **Related tests:** [AlgebraicSimplifierTest.PowHalf](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3577).

### elementwise-rsqrt-negative-power

```text
rsqrt(pow(X,-2)) -> X
```

- **Conditions:** IsNonNegative(X); exponent all -2.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleRsqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7999-L8020).
- **Related tests:** [AlgebraicSimplifierTest.RsqrtOfRPower2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11981), [AlgebraicSimplifierTest.RsqrtOfRPower2_NegativeTestCase](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12001).

### elementwise-rsqrt-reciprocal

```text
rsqrt(1/X) -> sqrt(X)
```

- **Conditions:** IsNonNegative(X); numerator all one.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleRsqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7999-L8020).
- **Related tests:** [AlgebraicSimplifierTest.RsqrtDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12017).

### elementwise-sqrt-square

```text
sqrt(X*X) -> abs(X)
```

- **Conditions:** enable_fast_math; identical operands; source inserts convert(abs(X)) for complex result type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSqrt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10102-L10130).
- **Related tests:** [AlgebraicSimplifierTest.SqrtOfSelfMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11882), [AlgebraicSimplifierTest.SquaredComplexSqrtIsFloat](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13884).
- **Note:** Complex and floating behavior follows the source fast-math contract.

### elementwise-compare-nonnegative-zero

```text
X < 0 -> false; X >= 0 -> true (also reversed comparisons)
```

- **Conditions:** Unsigned X or IsNonNegative(X), with the corresponding LT/GE or reversed GT/LE direction.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.CompareLtZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11313), [AlgebraicSimplifierTest.CompareGeZero](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11342), [AlgebraicSimplifierTest.CompareAbsLtZeroBecomesFalse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11284).

### elementwise-compare-self

```text
compare(X,X,d) -> truth(d)
```

- **Conditions:** Integral X; truth is true for EQ/GE/LE and false for GT/LT/NE.
- **RHS attributes:** Preserve predicate result shape.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.CompareSame](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11428).

### elementwise-compare-predicate-false

```text
P != false -> P (either operand order)
```

- **Conditions:** Predicate operands.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.NeFalse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11480), [AlgebraicSimplifierTest.NeFalse2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11501).

### elementwise-compare-predicate-true

```text
P == true -> P (either operand order)
```

- **Conditions:** Predicate operands.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.EqTrue](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11522), [AlgebraicSimplifierTest.EqTrue2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11543).

### elementwise-compare-selected-mask

```text
select(A>=B,1,0) != 0 -> A>=B
```

- **Conditions:** Exact GE/NE pattern; selected values are all one/all zero; compatible predicate shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.CompareSelectCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11563).

### elementwise-compare-max

```text
max(A,B)>A -> B>A; max(A,B)>B -> A>B
```

- **Conditions:** Exact GT comparison and matching operand nodes.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.CompareGtMaxA](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11197), [AlgebraicSimplifierTest.CompareGtMaxB](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11215).

### elementwise-compare-min

```text
A>min(A,B) -> A>B; B>min(A,B) -> B>A
```

- **Conditions:** Exact GT comparison and matching operand nodes.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5787-L5931).
- **Related tests:** [AlgebraicSimplifierTest.CompareGtAMin](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11233), [AlgebraicSimplifierTest.CompareGtBMin](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11251).

### elementwise-select-identical

```text
select(P,X,X) -> X
```

- **Conditions:** Compatible replacement type/shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectIdentical](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1266), [AlgebraicSimplifierTest.SelectIdenticalMixedPrecision](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1286).

### elementwise-select-constant

```text
select(true,X,Y) -> X; select(false,X,Y) -> Y
```

- **Conditions:** Predicate all true or all false; compatible surviving type/shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectTrue](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1178), [AlgebraicSimplifierTest.SelectFalse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1222), [AlgebraicSimplifierTest.SelectTrueMixedPrecision](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1200).

### elementwise-select-not

```text
select(not(P),X,Y) -> select(P,Y,X)
```

- **Conditions:** Well-formed HLO operands and compatible replacement shape/type.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectWithNotPred](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1307).

### elementwise-select-predicate-identity

```text
select(P,true,false) -> P
```

- **Conditions:** Predicate result.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectPredPred](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1335).

### elementwise-select-predicate-not

```text
select(P,false,true) -> not(P)
```

- **Conditions:** Predicate result.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectPredPred2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1357).

### elementwise-select-compare-gt

```text
select(compare(A,B,GT/GE),A,B) -> A or B
```

- **Conditions:** Predicate result; compare operands are the selected operands in the same order.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectGtCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1381).

### elementwise-select-compare-lt

```text
select(compare(A,B,LT/LE),A,B) -> A and B
```

- **Conditions:** Predicate result; compare operands are the selected operands in the same order.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectLtCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1401).

### elementwise-select-compare-eq

```text
select(compare(A,B,EQ),A,B) -> B
```

- **Conditions:** Predicate result; compare operands are the selected operands in the same order.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectEqCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1421).

### elementwise-select-compare-ne

```text
select(compare(A,B,NE),A,B) -> A
```

- **Conditions:** Predicate result; compare operands are the selected operands in the same order.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectNeCompare](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1438).

### elementwise-divide-converted-predicate

```text
convert(P) / broadcast(Y,[]) -> broadcast(1/Y,[]) * convert(P)
```

- **Conditions:** P predicate; Y scalar; non-integral result (integral path returns earlier).
- **RHS attributes:** Scalar reciprocal broadcast to the original result shape.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.ScalarDividePredicate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12595).
- **Note:** No finite-value or fast-math gate in this source branch; preserve its numerical contract.

### elementwise-remainder-compose

```text
(X % Y) % Y -> X % Y
```

- **Conditions:** Same divisor node.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleRemainder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6494-L6584).
- **Related tests:** [AlgebraicSimplifierTest.RepeatedRemainder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11730).

### elementwise-remainder-power-two-unsigned

```text
X % 2^k -> X and (2^k-1)
```

- **Conditions:** Unsigned integer; nonzero constant effective-scalar power-of-two divisor.
- **RHS attributes:** Mask is divisor-1, broadcast if needed.
- **Source:** [HandleRemainder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6494-L6584), [TryRemainderToAnd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6432-L6491).
- **Related tests:** [AlgebraicSimplifierTest.UnsignedRemainderByPowerOf2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L941).
- **Note:** Free helper TryRemainderToAnd, lines 6432–6491.

### elementwise-remainder-power-two-signed

```text
X % 2^k -> select(X<0,-(select(X<0,-X,X) and (2^k-1)),select(X<0,-X,X) and (2^k-1))
```

- **Conditions:** Signed integer; positive constant effective-scalar power-of-two divisor; fixed-width negation.
- **RHS attributes:** Mask divisor-1; restore dividend sign.
- **Source:** [HandleRemainder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6494-L6584), [TryRemainderToAnd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6432-L6491).
- **Related tests:** [AlgebraicSimplifierTest.SignedRemainderByPowerOf2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L958).

## Transpose / reshape

### transpose-reshape-reshape-empty

```text
reshape(X,empty_shape) -> empty_constant(empty_shape)
```

- **Conditions:** Result has zero elements.
- **RHS attributes:** Preserve result shape and type.
- **Source:** [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.ZeroSizedReshapeWithoutLayout](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10794).

### transpose-reshape-reshape-identity

```text
reshape(X,shape(X)) -> X
```

- **Conditions:** Compatible shapes/layout when required.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.RemoveNoopReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6147).

### transpose-reshape-reshape-compose

```text
reshape(reshape(X,s1),s2) -> reshape(X,s2)
```

- **Conditions:** Valid equal element counts.
- **RHS attributes:** Outer result shape s2.
- **Source:** [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.ReshapesMerged](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5386).

### transpose-reshape-reshape-transpose-cancel-chain

```text
unit_reshape/transpose_chain(X) -> X
```

- **Conditions:** Reshapes only insert/delete/move unit dimensions; at most one unit dimension per shape; effective nonunit permutation is identity; endpoint shapes compatible.
- **RHS attributes:** Compose effective nonunit permutations; logical reshape variant only.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [TryRemovingBitcastOrReshapeTransposeChain](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6587-L6744), [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571), [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.EliminateReshapeTransposeChain](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L514), [AlgebraicSimplifierTest.NotEliminateReshapeTransposeChain](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L612).
- **Note:** Physical bitcast cases in this helper are excluded from the logical catalog.

### transpose-reshape-reshape-hoist-transpose

```text
reshape(transpose(reshape(X,s1),p),s2) -> transpose(reshape(X,s3),q)
```

- **Conditions:** Outer reshape collapses contiguous chunks of transpose axes; no split or inserted unit dimension; chunks map to consecutive input axes.
- **RHS attributes:** Sort chunks by original start; s3 lists sorted chunk sizes; q restores chunk result order.
- **Source policy:** Layout-insensitive mode; enable_hoist_transpose_of_reshape.
- **Source:** [TryHoistTransposeOfReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6747-L6789), [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.HoistTransposeOfReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15325), [AlgebraicSimplifierTest.HoistTransposeOfReshapeNonContiguous](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15348), [AlgebraicSimplifierTest.HoistTransposeOfReshapeReorderChunks](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15434).

### transpose-reshape-transpose-identity

```text
transpose(X,identity) -> X
```

- **Conditions:** Permutation sorted (therefore identity).
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.NoOpTransposeDoesNotPropagateFrontendAttributesToOperand](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14539).

### transpose-reshape-transpose-compose

```text
transpose(transpose(X,p),q) -> transpose(X,gather(p,q))
```

- **Conditions:** Valid permutations.
- **RHS attributes:** r[i]=p[q[i]].
- **Source:** [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.TransposesMerged](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5442).

### transpose-reshape-transpose-unit-dims

```text
transpose(X,p) -> reshape(X,output_shape)
```

- **Conditions:** p changes only positions of unit axes; nonunit relative order preserved.
- **RHS attributes:** Use transpose result shape.
- **Source:** [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.TransposeIsReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5560).

### transpose-reshape-space-depth-to-slices

```text
reshape(transpose(reshape(X),p)) -> concat(slice(X,chunk_0),...,slice(X,chunk_n))
```

- **Conditions:** Inner reshape splits exactly one dimension into two; p moves only the smaller split dimension; outer reshape has the corresponding scaled shapes.
- **RHS attributes:** Chunk slices split original dimension into num_chunks; concatenate on destination dimension.
- **Source policy:** Layout-insensitive mode; rewrite_reshape_transpose_as_slice_concatenate; transpose has one user, an outer reshape; at most 5 chunks.
- **Source:** [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.TransposeReshapeToConcatSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7548), [AlgebraicSimplifierTest.TransposeReshapeTooLarge](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7600).
- **Note:** See exact source threshold when changing revisions; shapes are symbolic except for the bounded number of chunks.

## Broadcast

### broadcast-divide-broadcast

```text
X / broadcast(Y,d) -> X * broadcast(1/Y,d)
```

- **Conditions:** Non-integral; noncomplex unless enable_fast_math.
- **RHS attributes:** Preserve d and output shape.
- **Source policy:** CPU; enable_sink_broadcast; Y has one user.
- **Source:** [HandleDivide](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2506-L2702).
- **Related tests:** [AlgebraicSimplifierTest.DivideByBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10883).

### broadcast-multiply-constant-into-broadcast

```text
(X*C)*broadcast(Y,d) -> X*broadcast(Y*C,d)
```

- **Conditions:** Source requires integral arithmetic or enable_fast_math for floating/complex reassociation. C is scalar constant; Y nonconstant.
- **RHS attributes:** Broadcast C to shape(Y), then reuse d.
- **Source:** [HandleMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5063-L5293).
- **Related tests:** [AlgebraicSimplifierTest.MultiplyReassociateMultiplyOfConstantAndBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1148), [AlgebraicSimplifierTest.MultiplyBroadcastReassoc](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L719).

### broadcast-literal-splat

```text
constant_splat(c,shape) -> broadcast(c,shape,[])
```

- **Conditions:** Literal contains more than one element, all equal.
- **RHS attributes:** Extract first scalar; output shape unchanged.
- **Source:** [HandleConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2321-L2366).
- **Related tests:** [AlgebraicSimplifierTest.ConstantToBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2497).

### broadcast-reshape

```text
broadcast(X,d) -> reshape(X,output_shape)
```

- **Conditions:** d sorted; input and output element counts equal.
- **RHS attributes:** Use original broadcast output shape.
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.BroadcastAndReshapeMerged](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5604).

### broadcast-transpose

```text
broadcast(X,d) -> transpose(X,inverse(d))
```

- **Conditions:** Input/output ranks and element counts equal; non-sorted d reaches this branch.
- **RHS attributes:** Permutation inverse(d).
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.BroadcastToTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14420), [AlgebraicSimplifierTest.BroadcastToTranspose2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14436).

### broadcast-inserted-unit-dims

```text
broadcast(reshape_insert_ones(X),d) -> broadcast(X,d_filtered)
```

- **Conditions:** Reshape only inserts unit dimensions.
- **RHS attributes:** Remove entries of d at inserted dimension positions.
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.ReshapeAndBroadcastMerged](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5581).

### broadcast-elementwise

```text
f(broadcast(X1,d),...,broadcast(Xm,d)) -> broadcast(f(X1,...,Xm),d)
```

- **Conditions:** Compatible base shapes and broadcast maps; scalar-broadcast operands may be expanded to the common base shape; f elementwise, excluding copy/fusion.
- **RHS attributes:** Reuse common non-scalar d and original output shape.
- **Source policy:** enable_sink_broadcast.
- **Source:** [TryToSinkBroadcastAfterElementwiseOps](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6345-L6428).
- **Related tests:** [AlgebraicSimplifierTest.ElementwiseSinkMultipleBroadcastsScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1075), [AlgebraicSimplifierTest.ElementwiseSinkMultipleBroadcastsNonScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1112), [AlgebraicSimplifierTest.ElementwiseNoSinkBroadcastsDifferentDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1130).

### broadcast-scalar-through-indexing

```text
g(broadcast(scalar)) -> broadcast(scalar,shape(g),[])
```

- **Conditions:** g is reshape, transpose, reverse, single-output sort, static slice, or data-operand dynamic slice; broadcast source is scalar.
- **RHS attributes:** Empty broadcast map; use g output shape.
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.ScalarBroadcastToSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7480), [AlgebraicSimplifierTest.ScalarBroadcastToTransposeReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7514).
- **Note:** This parameterized schema groups operations that only permute or select identical scalar values.

### broadcast-iota

```text
broadcast(iota(X,k),d) -> iota(output_shape,d[k])
```

- **Conditions:** Valid iota and broadcast.
- **RHS attributes:** New iota axis d[k].
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.MergeBroadcastAndIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8195), [AlgebraicSimplifierTest.MergeBroadcastAndIota2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8215).

### broadcast-compose

```text
broadcast(broadcast(X,p),q) -> broadcast(X,gather(q,p))
```

- **Conditions:** Compatible nested broadcast shapes.
- **RHS attributes:** New map [q[i] for i in p].
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.MergeBroadcasts](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8146), [AlgebraicSimplifierTest.MergeBroadcasts2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8169).

### broadcast-drop-unit-dims

```text
broadcast(X,d) -> broadcast(reshape_drop_ones(X),d_filtered)
```

- **Conditions:** X has unit dimensions.
- **RHS attributes:** Delete d entries for unit operand dimensions.
- **Source policy:** Source applies in layout-insensitive mode. enable_broadcast_degenerate_dimension.
- **Source:** [HandleBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5650-L5785).
- **Related tests:** [AlgebraicSimplifierTest.DegenerateDimsInOperandRemovedFromBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2622).

### broadcast-reshape-broadcast

```text
reshape(broadcast(X,d),s) -> broadcast(X,d_new)
```

- **Conditions:** Reshape leaves all dimensions mapped from X unmodified.
- **RHS attributes:** Map d through the unmodified-dimension correspondence; result shape s.
- **Source:** [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.ReshapeBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4386).

### broadcast-transpose-broadcast

```text
transpose(broadcast(X,d),p) -> broadcast(X,gather(inverse(p),d))
```

- **Conditions:** New broadcast map sorted; d sorted; source additionally checks relative mapped-axis layouts in layout-sensitive mode.
- **RHS attributes:** d_new=[inverse(p)[i] for i in d]; output shape is transpose result.
- **Source:** [SimplifyTransposeOfBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2797-L2862), [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.TransposeOfBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14091), [AlgebraicSimplifierTest.TransposeOfBroadcastSkipped](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14146).

### broadcast-concat-repeat-unit-axis

```text
concat(X,...,X,d) -> broadcast(reshape_drop_axis(X,d),remaining_axes)
```

- **Conditions:** All operands identical; extent of X on d is one; source binary-concat branch must not return first.
- **RHS attributes:** Drop d then broadcast to concatenated shape.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2017-L2197).
- **Related tests:** [AlgebraicSimplifierTest.ConcatToBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9089).

### broadcast-pad-broadcast

```text
pad(broadcast(X),c,p) -> broadcast(pad(broadcast_small(X),c,p_small))
```

- **Conditions:** Mapped operand axes are a subset of padded axes; padded axes are a strict subset of all output axes.
- **RHS attributes:** Restrict shapes and p to padded axes; inner map positions within padded axes; outer map is padded-axis list.
- **Source policy:** Broadcast has one user.
- **Source:** [HandlePad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6078-L6283).
- **Related tests:** [AlgebraicSimplifierTest.BroadcastAndPadReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12864), [AlgebraicSimplifierTest.BroadcastAndPadReorderWithNonScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12899).

### broadcast-slice-broadcast-only

```text
slice(broadcast(X,d),a,b,s) -> broadcast(X,d,new_shape)
```

- **Conditions:** Every mapped X axis is unsliced (full bounds and unit stride); only broadcast-created axes may change.
- **RHS attributes:** Keep d and slice result shape.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfBroadcastToBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6227).

### broadcast-slice-broadcast

```text
slice(broadcast(X,d),a,b,s) -> broadcast(slice(X,gather(a,d),gather(b,d),gather(s,d)),d)
```

- **Conditions:** Valid broadcast and slice.
- **RHS attributes:** Gather starts/limits/strides by d; retain d and final shape.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5470), [AlgebraicSimplifierTest.SliceOfBroadcastPreserveLayout](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5489).

### broadcast-dynamic-slice-broadcast

```text
slice_dynamic(broadcast(X,d),s,z) -> broadcast(slice_dynamic(X,gather(s,d),gather(z,d)),d)
```

- **Conditions:** Valid dynamic slice; scalar X bypasses inner slice.
- **RHS attributes:** Gather starts/sizes by broadcast map d; keep final shape.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.DynamicSliceOfBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5511), [AlgebraicSimplifierTest.DynamicSliceOfBroadcastPreserveLayout](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5534).

### broadcast-gather-scalar

```text
gather(effective_scalar,indices,g) -> broadcast(reshape_scalar(operand),output_shape,[])
```

- **Conditions:** Operand has exactly one element.
- **RHS attributes:** Reshape operand to scalar if necessary; empty broadcast map.
- **Source:** [HandleGather](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4514-L4738).
- **Related tests:** [AlgebraicSimplifierTest.GatherOfScalarToBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10428).

### broadcast-window-padded-unit-broadcast

```text
reduce_window(pad(X,init,p),init,w,f) -> broadcast_sequence(X,output_shape)
```

- **Conditions:** Source window-pad prerequisites; symmetric edge padding only on unit axes; no window stride; each window contains exactly one unpadded value; nonunit axes have window size one.
- **RHS attributes:** Broadcast/reshape unit axes to output extents; preserve optional convert.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.FoldPadIntoReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7639).

## Reduction

### reduction-empty

```text
reduce(empty,init,f,r) -> broadcast(init,output_shape,[])
```

- **Conditions:** Input or result has zero elements; tuple reductions handled componentwise.
- **RHS attributes:** One scalar broadcast per result.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.TupleReduceBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10760).

### reduction-unit-axes

```text
reduce(X,init,f,r_unit) -> reshape(X,output_shape)
```

- **Conditions:** Input/output element counts equal; source assumes valid identity initial value for f.
- **RHS attributes:** Delete reduced unit axes; retain result shape.
- **Source policy:** If layout-sensitive, reshape must be bitcast-compatible.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.TupleReduceReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10726), [AlgebraicSimplifierTest.TrivialReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14610).

### reduction-negate

```text
reduce_sum(-X,r,0) -> -reduce_sum(X,r,0)
```

- **Conditions:** Floating result; additive reducer; scalar zero init.
- **RHS attributes:** Keep r.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceOfNegate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1662).

### reduction-argextreme-splat

```text
argmin/argmax(broadcast(c),iota,r) -> tuple(broadcast(c),broadcast(0))
```

- **Conditions:** Source recognized ArgMin/ArgMax combiner; value broadcast is scalar literal; iota axis is reduced; nonempty domain.
- **RHS attributes:** Preserve value/index result shapes and types.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceArgMaxBroadcastOfScalarConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1734), [AlgebraicSimplifierTest.ReduceArgMinBroadcastOfScalarConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1776).

### reduction-transpose

```text
reduce(transpose(X,p),init,f,r) -> transpose_if_needed(reduce(X,init,f,gather(p,r)),q)
```

- **Conditions:** Single-output reduction.
- **RHS attributes:** Map reduction axes through p; q is surviving p entries renumbered after deletion of reduced input axes.
- **Source policy:** Layout-insensitive mode; allow if unconditional flag, result rank<2, one transpose user, or all transpose users are reductions.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** No dedicated test identified in the captured file.
- **Note:** The permutation formula is explicit in HandleReduce; no dedicated test identified in the captured test file.

### reduction-compose

```text
reduce(reduce(X,init,f,r1),init,f,r2) -> reduce(X,init,f,union(r1,lift(r2)))
```

- **Conditions:** Same initial value by Identical; equivalent reduction computations; single-output path.
- **RHS attributes:** Lift outer positions past removed inner axes, union and sort reduction positions.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [MergeReduces](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8890-L8918).
- **Related tests:** [AlgebraicSimplifierTest.TwoReducesToOne](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1544), [AlgebraicSimplifierTest.ReduceAddIsCommutative](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11954).

### reduction-reshape-reduced

```text
reduce(reshape(X),init,f,r) -> reduce(X,init,f,r_new)
```

- **Conditions:** Every surviving axis is unmodified by reshape.
- **RHS attributes:** Reduce all original axes except the unmodified surviving ones.
- **Source policy:** Source applies in layout-insensitive mode. enable_reduce_of_reshape.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceOfReshapeOfContractingAndNoncontractingDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1640).

### reduction-reshape-free

```text
reduce(reshape(X),init,f,r) -> reshape(reduce(X,init,f,r_new))
```

- **Conditions:** Every reduced axis is unmodified by reshape; only free axes split/merge.
- **RHS attributes:** Transport reduced positions through unmodified-dimension map; reshape reduced result.
- **Source policy:** Source applies in layout-insensitive mode. enable_reduce_of_reshape; one user or unconditional simplify flag.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceOfMergeNoncontractingDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1581), [AlgebraicSimplifierTest.ReduceOfSplitNoncontractingDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1611).

### reduction-concat

```text
reduce(concat(X1,...,Xn,d),init,f,r) -> fold_f(reduce(X1,init,f,r),...,reduce(Xn,init,f,r))
```

- **Conditions:** Concat axis d is reduced; source same-shape requirement unless unconditional flag.
- **RHS attributes:** Reuse reduction positions and combiner; fold results elementwise.
- **Source policy:** Layout-insensitive mode; equal operand shapes, scalar result, or enable_unconditional_reduce_of_concat_replacement.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.SimplifyReduceOfConcat](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4831), [AlgebraicSimplifierTest.SimplifyReduceOfConcatWithDifferentShapes](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4882).

### reduction-reuse-argmax

```text
reduce_max(X,r,-inf) -> get_tuple_element(argmax(X,r),0)
```

- **Conditions:** Existing recognized ArgMax reduction on same X and same reduction dimensions.
- **RHS attributes:** Tuple index 0.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReplaceReduceMaxWithReduceArgMax](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12709).

### reduction-reuse-argmin

```text
reduce_min(X,r,+inf) -> get_tuple_element(argmin(X,r),0)
```

- **Conditions:** Existing recognized ArgMin reduction on same X and same reduction dimensions.
- **RHS attributes:** Tuple index 0.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReplaceReduceMinWithReduceArgMin](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12770).

### reduction-splat-min

```text
reduce_min(broadcast(scalar X),all_axes,init) -> min(X,init)
```

- **Conditions:** Nonempty input; scalar X; all broadcast axes reduced; exact source combiner.
- **RHS attributes:** No new axis attributes.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceBroadcastOfScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1686).

### reduction-splat-max

```text
reduce_max(broadcast(scalar X),all_axes,init) -> max(X,init)
```

- **Conditions:** Nonempty input; scalar X; all broadcast axes reduced; exact source combiner.
- **RHS attributes:** No new axis attributes.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceBroadcastOfScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1686).

### reduction-splat-and

```text
reduce_and(broadcast(scalar X),all_axes,init) -> and(X,init)
```

- **Conditions:** Nonempty input; scalar X; all broadcast axes reduced; exact source combiner.
- **RHS attributes:** No new axis attributes.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceBroadcastOfScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1686).

### reduction-splat-or

```text
reduce_or(broadcast(scalar X),all_axes,init) -> or(X,init)
```

- **Conditions:** Nonempty input; scalar X; all broadcast axes reduced; exact source combiner.
- **RHS attributes:** No new axis attributes.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceBroadcastOfScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1686).

### reduction-broadcast-original-axes

```text
reduce(broadcast(X,d),init,f,r) -> broadcast(reduce(X,init,f,r_new),d_new)
```

- **Conditions:** Every reduced output axis belongs to d (no replication axis is reduced).
- **RHS attributes:** r_new=index_of(r in d); shift surviving d entries past removed result axes; preserve surviving base-axis order.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.CommuteReduceAndBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15494), [AlgebraicSimplifierTest.CommuteReduceAndBroadcastUnsorted](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15519).

### reduction-broadcast-replication-axes

```text
reduce_sum(broadcast(X,d),r,init) -> broadcast(X*product(extents(r))+init,d_new)
```

- **Conditions:** All reduced axes are broadcast-created; d sorted; source additive combiner.
- **RHS attributes:** Multiply replication extents; delete reduced dimensions; shift surviving broadcast positions.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceBroadcastScalarToBroadcastMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14731), [AlgebraicSimplifierTest.ReplaceReduceSumOfConstantBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12681).

### reduction-window-empty

```text
reduce_window(empty,init,w,f) -> broadcast(init,output_shape,[])
```

- **Conditions:** Input or output has zero elements; variadic results treated componentwise.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.ZeroSizedReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3700), [AlgebraicSimplifierTest.ZeroSizedVariadicReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3742).

### reduction-window-scalar

```text
reduce_window(scalar,init,w,f) -> scalar
```

- **Conditions:** Rank-zero input; source identity-initializer convention; supported tuple-root form for variadic case.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** No dedicated test identified in the captured file.

### reduction-window-splat-cumsum

```text
reduce_window_sum(broadcast(c),0,cumulative_window_d) -> c*(iota(output_shape,d)+1)
```

- **Conditions:** Zero init and additive reducer; scalar literal through source-recognized reshape/convert/slice/copy/value-preserving-bitcast chains; exactly one nontrivial full-extent cumulative window; stride=1, low pad=size-1, high pad=0, no dilation/reversal; other axes trivial.
- **RHS attributes:** Derive counting iota axis; preserve conversion order for constant and count; use appropriate iota integer type.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.ReduceWindowCumSumBroadcastOfConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1818), [AlgebraicSimplifierTest.ReduceWindowCumSumBroadcastOfConstantWithElementTypeChangingBitcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L2093).

### reduction-window-singleton-add

```text
reduce_window_add(X,init,unit_window) -> pad_if_needed(add(X,broadcast(init)),init,window_edges)
```

- **Conditions:** Every window size/stride/base-dilation/window-dilation is one; no reversal; exact source combiner.
- **RHS attributes:** Convert window edge padding into ordinary edge padding; preserve output shape.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.TrivialReduceWindow_Add](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4218).

### reduction-window-singleton-min

```text
reduce_window_min(X,init,unit_window) -> pad_if_needed(min(X,broadcast(init)),init,window_edges)
```

- **Conditions:** Every window size/stride/base-dilation/window-dilation is one; no reversal; exact source combiner.
- **RHS attributes:** Convert window edge padding into ordinary edge padding; preserve output shape.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.TrivialReduceWindow_Min](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4244).

### reduction-window-singleton-max

```text
reduce_window_max(X,init,unit_window) -> pad_if_needed(max(X,broadcast(init)),init,window_edges)
```

- **Conditions:** Every window size/stride/base-dilation/window-dilation is one; no reversal; exact source combiner.
- **RHS attributes:** Convert window edge padding into ordinary edge padding; preserve output shape.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.TrivialReduceWindow_Max](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4271).

### reduction-window-to-reduce

```text
reduce_window(X,init,w,f) -> reshape(reduce(X,init,f,full_window_axes))
```

- **Conditions:** No stride/dilation/padding; each window axis is size one or covers the whole input axis and yields output extent one.
- **RHS attributes:** Reduce full-window axes; reshape to retain unit result dimensions.
- **Source policy:** Source applies in layout-insensitive mode. enable_window_reduce_to_reduce_replacement.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.ReduceWindowIsReduceAndReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3655).

### reduction-slice-window-to-reduce

```text
slice(reduce_window(X,init,w,f),a,b,1) -> reshape(reduce(X,init,f,full_window_axes))
```

- **Conditions:** Padded full-window pattern; no stride or dilation; each axis either unchanged with window size one, or selected last singleton of full-window output.
- **RHS attributes:** Derive full reduced axes; reshape output with unit axes.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfReduceWindowOneReduceDim](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9029), [AlgebraicSimplifierTest.SliceOfReduceWindowTwoReduceDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9059).

### reduction-window-pad

```text
reduce_window(pad(X,c,p),init,w,f) -> reduce_window(X,init,w_new,f)
```

- **Conditions:** No negative pad; cannot combine interior padding with existing base dilation; padding value and init must agree under applicable literal/type checks; optional convert is preserved.
- **RHS attributes:** new_low=w.low+w.base_dilation*p.low; likewise high; if interior>0 set base_dilation=interior+1.
- **Source policy:** Source applies in layout-insensitive mode. Source scan-padding profitability screening and reduce-window-rewriter option.
- **Source:** [HandleReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9446-L9948).
- **Related tests:** [AlgebraicSimplifierTest.FoldPadIntoReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7639), [AlgebraicSimplifierTest.DoNotFoldNegativePadIntoReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7678), [AlgebraicSimplifierTest.FoldConvertedPadIntoReduceWindow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7704).

## Dot

### dot-empty

```text
dot(X,Y,d) -> broadcast(0,output_shape,[])
```

- **Conditions:** Result or either operand has zero elements.
- **RHS attributes:** Preserve dot result shape/type.
- **Source policy:** Source applies in layout-insensitive mode. Sparse dots with extra operands are skipped.
- **Source:** [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** No dedicated test identified in the captured file.
- **Note:** Direct empty-dot branch; no dedicated test identified in the captured test file.

### dot-no-contraction

```text
dot(X,Y,B,K=[]) -> broadcast(transpose(X))*broadcast(transpose(Y))
```

- **Conditions:** No contraction axes; precision algorithm ALG_UNSET or ALG_DOT_F32_F32_F32; convert operand types to result type when needed.
- **RHS attributes:** Order operand axes as batch then free; broadcast both operands into dot output order.
- **Source policy:** Source applies in layout-insensitive mode. enable_dot_to_multiply_rewrite.
- **Source:** [RewriteAsMultiplyDotWithZeroLhsContractingDim](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3784-L3827), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.CanDisableDotToMultiplyRewrite](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11587), [AlgebraicSimplifierTest.NoDotToMultiplyRewriteZeroContractingDimWithPrecisionConfigAlgorithm](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11626).

### dot-reduce-product

```text
dot(X,Y,B,K) -> reduce_sum(broadcast(transpose(X))*broadcast(transpose(Y)),K)
```

- **Conditions:** At least one operand has only batch and contracting axes; precision algorithm ALG_UNSET or ALG_DOT_F32_F32_F32; operands converted to result type before multiply.
- **RHS attributes:** Transpose operands batch-major/contracting-minor; broadcast free axes; reduce trailing contracting axes. Accumulate floating product in F64 for F64 result, otherwise F32; convert sum back to result type.
- **Source policy:** Layout-insensitive; enable_dot_strength_reduction; CPU additionally requires both operands to have only batch/contracting axes; skip unreduced sharding marker.
- **Source:** [RewriteBatchPlusContractingAsReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4182-L4243), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [BatchDotStrengthReductionTest.BatchDotStrengthReduction](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9424), [AlgebraicSimplifierTest.CpuDotStrengthReductionGEMV](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9523), [AlgebraicSimplifierTest.NoDotStrengthReductionWithPrecisionConfigAlgorithm](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12633).

### dot-drop-unit-dims

```text
dot(X,Y,d) -> reshape(dot(reshape_drop_ones(X),reshape_drop_ones(Y),d_new))
```

- **Conditions:** At least one operand has unit axes.
- **RHS attributes:** Delete unit axes; renumber batch/contracting lists; reshape result to original output shape.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [RemoveDegenerateDimensionFromDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2705-L2784), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.DotContractingReorder_SizeOneDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11150).

### dot-transpose-both-operands

```text
dot(transpose(X,p),transpose(Y,p),d) -> transpose(dot(X,Y,d_new),q) (possibly swapped X,Y)
```

- **Conditions:** Equal operand permutations; equal ranks; canonical leading batch axes; one free and one contracting axis per side; p only permutes batch axes and optionally swaps final matrix axes.
- **RHS attributes:** Keep canonical dot dimension numbers; swap operand order/precision when the last two axes swap; wrap result with the original permutation.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [RemoveTransposesFromDotOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2865-L2925), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.BatchDotTransposeOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9148), [AlgebraicSimplifierTest.BatchDotTransposeBatchDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9180), [AlgebraicSimplifierTest.BatchDotTransposeBatchDimsAndOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9212).

### dot-swap-parameter-operand

```text
dot(X,Y,d) -> transpose(dot(Y,X,swap_sides(d)),q)
```

- **Conditions:** X parameter-like; Y not parameter-like.
- **RHS attributes:** Swap batch/contraction lists; q restores [batch,left-free,right-free] output order.
- **Source policy:** Source applies in layout-insensitive mode. enable_move_dot_param_to_rhs.
- **Source:** [MoveDotParamToRhs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2943-L2993), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.SwapDotOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14559).

### dot-concat-contraction

```text
dot(concat(X1,...,Xn,k),C,d) -> sum_i dot(Xi,slice(C,paired_chunk_i),d)
```

- **Conditions:** No batch axes; one contraction axis per side; rank-two result; concatenation along contracting axis; other operand rank-two constant; symmetric side supported.
- **RHS attributes:** Cumulative concat extents select paired constant slices; retain incoming contraction orientation.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [OptimizeDotOfConcatHelper](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3055-L3170), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390), [OptimizeDotOfConcat](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3028-L3052).
- **Related tests:** [DotOfConcatSimplificationTest.ConstantLHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9636), [DotOfGatherSimplificationTest.ConstantLHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10309), [DotOfConcatSimplificationTest.ConstantRHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9695), [DotOfGatherSimplificationTest.ConstantRHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10237), [DotOfConcatSimplificationTest.DotOfConcatRank3LhsRank1Rhs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10190).
- **Note:** Tests named ConstantLHS/RHS exist in several fixtures; see fixture-qualified references.

### dot-constant-dynamic-slice

```text
dot(slice_dynamic(C,s,z),D,d) -> slice_dynamic(dot(C,D,d),s_new,z_new)
```

- **Conditions:** Rank-two operands; one contraction axis each; C,D literals; slice preserves full contracting axis and selects one element on the other axis.
- **RHS attributes:** Transport selected free-axis start to output; other start zero; symmetric side supported.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [OptimizeDotOfGather](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3172-L3299), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [DotOfConcatSimplificationTest.ConstantLHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9636), [DotOfGatherSimplificationTest.ConstantLHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10309), [DotOfConcatSimplificationTest.ConstantRHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9695), [DotOfGatherSimplificationTest.ConstantRHS](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10237).

### dot-move-contracting-permutation

```text
dot(reshape(transpose(X,p)),C,d) -> dot(reshape(X),reshape(transpose(reshape(C),q)),d)
```

- **Conditions:** C constant; one contraction axis; reshape collapses a consecutive block into that axis; p only permutes the block.
- **RHS attributes:** Unsquash constant contraction axis, apply inverse block permutation, squash it again; restore any operand swap.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [OptimizeDotOfReorderContractingDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3311-L3517), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.DotContractingReorder_RL](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10922), [AlgebraicSimplifierTest.DotContractingReorder_NegTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11057), [AlgebraicSimplifierTest.DotContractingReorder_NegReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11074).

### dot-move-reverse

```text
dot(reverse(X,k),Y,d) -> dot(X,reverse(Y,paired(k)),d)
```

- **Conditions:** Moved reversed axes are contracting; unmatched noncontracting reversals stay on X; either side may be the producer.
- **RHS attributes:** Map moved reverse axes through paired contracting lists.
- **Source policy:** Source applies in layout-insensitive mode. use_associative_reordering; source size-ratio threshold.
- **Source:** [AssociativeReorderDotOperator](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3522-L3781), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.DotReverseLeftReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8378), [AlgebraicSimplifierTest.DotReverseRightReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8403).

### dot-move-zero-pad

```text
dot(pad(X,0,p),Y,d) -> dot(X,slice(Y,paired_nonpadding_coordinates),d_new)
```

- **Conditions:** Padding value zero; only nonnegative contracting-axis padding is moved; other padding remains.
- **RHS attributes:** For paired axis: start=low, limit=extent-high, stride=interior+1; clear moved padding.
- **Source policy:** Source applies in layout-insensitive mode. use_associative_reordering; FLOP threshold.
- **Source:** [AssociativeReorderDotOperator](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3522-L3781), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.DotPadLeftReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8429), [AlgebraicSimplifierTest.DotPadRightReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8455), [AlgebraicSimplifierTest.AssociativeReorderDotPadNonZeroOrInteriorPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8482).

### dot-move-broadcast

```text
dot(broadcast(X,b),Y,d) -> dot(X,reduce_sum(Y,paired_broadcast_contractions),d_new)
```

- **Conditions:** Moved broadcast-created axes are contracting; remaining broadcast axes stay on the producer side.
- **RHS attributes:** Reduce paired axes on other operand; remove moved contraction pairs; renumber remaining lists.
- **Source policy:** Source applies in layout-insensitive mode. use_associative_reordering; FLOP threshold.
- **Source:** [AssociativeReorderDotOperator](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3522-L3781), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.DotBroadcastLeftReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8551), [AlgebraicSimplifierTest.DotBroadcastRightReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8578).

### dot-associate

```text
dot(dot(X,Y,d1),Z,d2) -> transpose_if_needed(dot(X,dot(Y,Z,d3),d4),p) (also reverse association)
```

- **Conditions:** Reject if outer contraction consumes an inner batch axis; shared batches and contractions must map through the inner output; valid inferred shapes.
- **RHS attributes:** Build pairwise X/Y/Z axis correspondences; derive new inner/outer batch and contraction lists; p restores original output order.
- **Source policy:** Source applies in layout-insensitive mode. use_associative_reordering; FLOP improvement threshold; preserve precision configuration.
- **Source:** [AssociativeReorderNestedDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L3830-L4180), [HandleDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4258-L4390).
- **Related tests:** [AlgebraicSimplifierTest.DotAssociativeReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8261), [AlgebraicSimplifierTest.DotLeftDotSharedBatchReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8290), [AlgebraicSimplifierTest.DotRightDotSharedBatchReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8320), [AlgebraicSimplifierTest.DotRightDotContractBatchReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8350).
- **Note:** Algebraic association is subject to the source reduction/numerical contract, not exact IEEE equality.

### dot-slice-output

```text
slice(dot(X,Y,d),a,b,s) -> dot(slice(X,a_X,b_X,s_X),slice(Y,a_Y,b_Y,s_Y),d)
```

- **Conditions:** Slice affects output/free/batch axes, not hidden contracting axes; batch selections transported to both operands; resulting operands still each have free axes.
- **RHS attributes:** Project output starts/limits/strides to operand axes; leave contracting axes full.
- **Source policy:** Source applies in layout-insensitive mode. raise_slice_and_reduce_through_dot.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceDotReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8638), [AlgebraicSimplifierTest.SliceDotReorderWithStrides](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8662).

### dot-reduce-output-free

```text
reduce_sum(dot(X,Y,d),r,0) -> dot(reduce_sum(X,r_X,0),reduce_sum(Y,r_Y,0),d_new)
```

- **Conditions:** Dot has no batch axes; reducer is addition with zero initial value.
- **RHS attributes:** Map reduced output axes back to each operand; delete them and renumber contraction axes.
- **Source policy:** Source applies in layout-insensitive mode. raise_slice_and_reduce_through_dot; FLOP threshold.
- **Source:** [ReorderReduceDotToDotReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8780-L8888).
- **Related tests:** [AlgebraicSimplifierTest.ReduceDotReorder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8604).

### dot-reduce-output-batch

```text
reduce_sum(dot(X,Y,d),r,0) -> reduce_sum(dot(X,Y,d_batch_to_contract),r_remaining,0)
```

- **Conditions:** Reducer addition with zero init; at least one reduced output axis is a dot batch axis.
- **RHS attributes:** Move selected batch-axis pairs to contraction lists; renumber remaining output reduction positions; omit final reduce if empty.
- **Source policy:** Source applies in layout-insensitive mode. supports_non_canonical_dots.
- **Source:** [HandleReduce](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8920-L9426).
- **Related tests:** [AlgebraicSimplifierTest.ReduceOfBatchDotToContractingDimension](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11899), [AlgebraicSimplifierTest.ReduceOfBatchDotToContractingDimensionDisabled](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11927).

### dot-transpose-output

```text
transpose(dot(X,Y,d),p) -> dot(Y,X,d_new)
```

- **Conditions:** p swaps the complete left/right free-axis blocks while preserving each block order, optionally permuting leading batch axes; canonical path has one free axis per side.
- **RHS attributes:** Swap side-specific contraction lists; reorder batch lists according to p.
- **Source policy:** Noncanonical/general form requires supports_non_canonical_dots; parameter-placement policy may suppress swap.
- **Source:** [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.TransposeOfDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8235), [AlgebraicSimplifierTest.TransposeOfBatchDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8687), [AlgebraicSimplifierTest.TransposeOfBatchDimsInBatchDotCantSimplify](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8723).
- **Note:** Canonical and general implementations of the same block-swap schema are merged.

### dot-scalar-multipliers

```text
bilinear_op(...,a*X,...) -> a*bilinear_op(...,X,...) (or move scalar to other operand)
```

- **Conditions:** Constant scalar factors along multiply-commuting operations; BF16 or F32; shared-user paths screened before mutation.
- **RHS attributes:** Broadcast combined scalar factor at chosen operand/result shape.
- **Source policy:** enable_scalar_multiply_reduction; choose smallest tensor location.
- **Source:** [ScalarMultiplyReduction](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L704-L853).
- **Related tests:** [AlgebraicSimplifierTest.ScalarMultiplyReduction](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10137), [AlgebraicSimplifierTest.ScalarMultiplyReductionMultiUser](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10167).
- **Note:** Shared dot/convolution helper counted once here; includes constant-factor combination under source numerical policy.

## Convolution

### convolution-empty

```text
conv(empty,W,c) or conv(X,empty,c) -> broadcast(0,output_shape,[])
```

- **Conditions:** Input or filter has zero elements.
- **RHS attributes:** Preserve result shape and element type.
- **Source:** [HandleConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L11036-L11089).
- **Related tests:** [AlgebraicSimplifierTest.ZeroSizedConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L3610).

### convolution-fold-input-pad

```text
conv(pad(X,0,p),W,c) -> conv(X,W,c_new)
```

- **Conditions:** Zero padding; batch/feature axes unpadded; no negative edges; interior padding does not conflict with convolution base dilation or other window padding.
- **RHS attributes:** Add edge padding to window; move interior padding into base dilation when permitted.
- **Source policy:** enable_folding_pad_into_convolution.
- **Source:** [FoldConvInputPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10573-L10640), [HandleConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L11036-L11089).
- **Related tests:** [AlgebraicSimplifierTest.DoNotFoldNegativePadIntoConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6884).

### convolution-fold-filter-pad

```text
conv(X,pad(W,0,p),c) -> conv(X,W,c_new)
```

- **Conditions:** Zero filter padding; no feature-axis padding/dilation; spatial edge padding zero; no simultaneous existing filter dilation.
- **RHS attributes:** Set kernel window size to unpadded spatial extent; window dilation=interior+1.
- **Source policy:** enable_folding_pad_into_convolution.
- **Source:** [FoldConvFilterPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10642-L10709), [HandleConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L11036-L11089).
- **Related tests:** [ConvFilterPaddingTest.DoIt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6955), [PadReduceWindowEffectiveBroadcastTest.DoIt](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9312).

### convolution-pointwise-dot

```text
conv_1x1(X,W,c) -> reshape(dot(reshape(X,[M,Cin]),reshape(W,[Cin,Cout])),output_shape)
```

- **Conditions:** All kernel spatial extents one; unit strides; no edge padding/base dilation; feature_group_count=batch_group_count=1; layouts permit value-preserving bitcasts.
- **RHS attributes:** M=product(nonfeature input extents); dot contracts Cin; reshape to original output.
- **Source policy:** enable_conv_simplification; layout-sensitive; input/output feature axis most minor; filter Cin more major than Cout.
- **Source:** [SimplifyConvToDot](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10804-L10923), [HandleConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L11036-L11089).
- **Related tests:** [AlgebraicSimplifierTest.ConvertConvToMatmul](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7031).
- **Note:** Logical equation abstracts the source physical bitcasts; its layout requirements are retained.

### convolution-depthwise-product

```text
conv(X,W,c) -> reshape(reduce_sum(transpose(X)*broadcast(transpose(W)),full_kernel_axes))
```

- **Conditions:** No sparse RHS; each spatial kernel extent one or output extent one; unit strides, no padding/dilation; kernel input feature extent one; input/output features equal feature_group_count.
- **RHS attributes:** Permute input/kernel to output axis order; broadcast kernel; reduce spatial axes whose kernel extent is not one; reshape.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [SimplifyConvToMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10925-L11034), [HandleConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L11036-L11089).
- **Related tests:** [ConvTestBase.FeatureDimension](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7397), [ConvTestBase.ContractedDimension](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7402), [ConvTestWindow.NoMatch](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7463).

### convolution-scale-filter-bias

```text
(conv(X,W,c)+bias)*broadcast(scale,output_feature) -> conv(X,W*broadcast(scale,kernel_output_feature),c)+bias*broadcast(scale,output_feature)
```

- **Conditions:** Filter constant; scale is a 1D constant broadcast exactly to convolution output-feature axis.
- **RHS attributes:** Map scale broadcast to kernel output-feature axis; preserve convolution attributes.
- **Source policy:** enable_conv_add_multiply_reorder; convolution, bias, and scale broadcast each have one user.
- **Source:** [TryToReorderConvAddMultiply](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L5018-L5061).
- **Related tests:** [AlgebraicSimplifierTest.ReorderConvAddMul](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L748), [AlgebraicSimplifierTest.DoNotReorderConvAddMulWithUnmatchingOutputFeatureDimension](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L798).

### convolution-swap-operands

```text
conv(X,W,c) -> conv(reverse_if_needed(W),X,c_swapped)
```

- **Conditions:** Ungrouped; no sparsity/block scaling. Each spatial axis is either a source-recognized group/contraction axis or uses the transformed reversed window.
- **RHS attributes:** Swap input/kernel dimension roles and output batch/feature roles; swap operand precision; on general spatial axes swap base/window dilation, set reversal=true, new_edge=dilated_input+old_edge-dilated_kernel, new_window_size=input_size.
- **Source policy:** enable_conv_operand_swap; layout-insensitive; smaller transformed kernel; backend lowerability callback.
- **Source:** [SwapConvOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10711-L10724), [SwapConvolutionOperandsIfBeneficial](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/conv_operand_swapper.cc#L34-L173), [HandleConvolution](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L11036-L11089).
- **Related tests:** [AlgebraicSimplifierTest.SwapConvOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12568).
- **Note:** Delegated implementation is captured in upstream/conv_operand_swapper.cc.

## Indexing

### indexing-add-sparse-update

```text
X + update(zeros, Y, s) -> update(X, slice_dynamic(X, s, shape(Y)) + Y, s)
```

- **Conditions:** The updated base is a broadcast of scalar zero; shapes and index types are valid.
- **RHS attributes:** Reuse update starts and shape(Y) as slice sizes.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.AddDynamicUpdateSliceToAddSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10070), [AlgebraicSimplifierTest.AddDynamicUpdateSliceToAddSliceOnLhs](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10107).

### indexing-add-scatter

```text
X + scatter_add(A, indices, U) -> scatter_add(X + A, indices, U)
```

- **Conditions:** Single scatter operand; scatter combiner is parameter addition.
- **RHS attributes:** Reuse scatter dimension numbers and indices.
- **Source policy:** Source applies in layout-insensitive mode. Scatter has one user.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.ScatterAddCombined](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12254).

### indexing-add-two-scatters-concat

```text
scatter_add(A,i,U) + scatter_add(B,j,V) -> scatter_add(A+B, concat(i,j), concat(U,V))
```

- **Conditions:** Equal scatter dimension numbers and index types; compatible update windows; at most one unequal non-vector/non-batching index dimension; batching extents match; unique_indices and indices_are_sorted are false.
- **RHS attributes:** Select compatible index/update concatenation axes; keep scatter dimension numbers; set sorted/unique flags false.
- **Source policy:** Source applies in layout-insensitive mode. Both scatters have one user and concatenated updates are smaller than the result.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.ScatterAddCombined](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12254), [AlgebraicSimplifierTest.ScatterAddCombinedWithBatchDim](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12428).

### indexing-add-two-scatters-nested

```text
scatter_add(A,i,U) + scatter_add(B,j,V) -> scatter_add(scatter_add(A+B,j,V),i,U)
```

- **Conditions:** Both scatters use addition; equal index rank; source batching/shape screening passes; concatenation path is unavailable.
- **RHS attributes:** Retain each scatter specification.
- **Source policy:** Source applies in layout-insensitive mode. One user per scatter; update-size screening as in the source.
- **Source:** [HandleAdd](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L937-L1273).
- **Related tests:** [AlgebraicSimplifierTest.ScatterAddCombinedWeirdDnums](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12342).

### indexing-select-update

```text
select(P,X,update(X,Y,s)) -> update(X,select(P,slice_dynamic(X,s,shape(Y)),Y),s)
```

- **Conditions:** Same base X; source also matches the reversed branch order.
- **RHS attributes:** Slice a shaped P at s, or broadcast scalar P to update shape; reuse update starts and sizes.
- **Source:** [HandleSelect](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L9950-L10062).
- **Related tests:** [AlgebraicSimplifierTest.SelectDUSWithShapedPred](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1471), [AlgebraicSimplifierTest.ReverseSelectDUSWithShapedPred](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L1495).

### indexing-literal-iota

```text
constant([0,1,...,n-1]) -> iota(n,axis=0)
```

- **Conditions:** Rank-one literal recognized as iota.
- **RHS attributes:** iota dimension 0.
- **Source:** [HandleConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2321-L2366).
- **Related tests:** [AlgebraicSimplifierTest.ConstantToIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13018).

### indexing-literal-strided-iota

```text
constant([0,s,2s,...]) -> iota(n,0)*s
```

- **Conditions:** Rank-one literal recognized as strided iota.
- **RHS attributes:** Extract literal stride; iota dimension 0.
- **Source:** [HandleConstant](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2321-L2366).
- **Related tests:** No dedicated test identified in the captured file.

### indexing-reshape-iota

```text
reshape(iota(shape,k),s) -> mixed_radix_iotas(s)
```

- **Conditions:** The old iota dimension boundaries align with common input/output shape factors.
- **RHS attributes:** For each output digit axis use iota(axis)*product(minor digit extents); add digits; broadcast if needed.
- **Source:** [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.IotaAndReshapeMerged](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5720), [AlgebraicSimplifierTest.IotaAndReshapeToMixedRadix](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5741), [AlgebraicSimplifierTest.IotaAndReshape_4_3x2x4_6x1x1x4](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5830).

### indexing-reshape-update

```text
reshape(update(X,Y,s)) -> update(reshape(X),reshape(Y),s_new)
```

- **Conditions:** Reshape only inserts/deletes unit dimensions; same changes valid for update and base.
- **RHS attributes:** Insert zero starts and unit sizes; delete starts/sizes for removed unit axes.
- **Source policy:** Source applies in layout-insensitive mode. Update has one user.
- **Source:** [HandleReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6791-L7078).
- **Related tests:** [AlgebraicSimplifierTest.ReshapeOfDupDoNotCloneMultiUserDup](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14043).

### indexing-concat-unary

```text
concat(X,axis) -> X
```

- **Conditions:** Single operand; replacement compatible.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2017-L2197).
- **Related tests:** [AlgebraicSimplifierTest.RemoveUnaryConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4637).

### indexing-concat-empty

```text
concat(...,empty,...) -> concat(nonempty operands)
```

- **Conditions:** Remove zero-element operands; if all empty use a compatible empty operand; if one remains return it.
- **RHS attributes:** Keep concat axis and output shape.
- **Source:** [HandleConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2017-L2197).
- **Related tests:** [AlgebraicSimplifierTest.RemoveEmptyConcatenateOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4797), [AlgebraicSimplifierTest.OnlyEmptyConcatenateOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4936).

### indexing-concat-flatten

```text
concat(X,concat(Y,Z,d),d) -> concat(X,Y,Z,d)
```

- **Conditions:** Nested concatenates have the same dimension.
- **RHS attributes:** Keep d; splice operand lists.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2017-L2197).
- **Related tests:** [DotOfConcatSimplificationTest.UnnestConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9781).

### indexing-concat-adjacent-slices

```text
concat(slice(X,a,b),slice(X,b,c),d) -> slice(X,a,c)
```

- **Conditions:** Unit strides; same X; adjacent along d; same starts in other dimensions.
- **RHS attributes:** Extend first slice limit on d; preserve other starts/limits.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2017-L2197).
- **Related tests:** [AlgebraicSimplifierTest.SimplifyConcatenateOfSlices](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4987).

### indexing-concat-splat-to-pad

```text
concat(broadcast(c),X,d) -> pad(X,c,low_d); symmetric high padding
```

- **Conditions:** Binary concat; one side is scalar broadcast.
- **RHS attributes:** Pad low or high on d by broadcast extent; other padding and interior padding zero.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleConcatenate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L2017-L2197).
- **Related tests:** [AlgebraicSimplifierTest.ConcatenateOfBroadcastBecomesPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4965).

### indexing-iota-trivial

```text
iota(shape,k) -> zeros(shape)
```

- **Conditions:** Extent of iota axis <= 1.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6067-L6076).
- **Related tests:** [AlgebraicSimplifierTest.IotaEffectiveScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5789).

### indexing-pad-empty

```text
pad(empty,c,p) -> broadcast(c,output_shape,[])
```

- **Conditions:** Input has zero elements.
- **RHS attributes:** Preserve output shape.
- **Source:** [HandlePad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6078-L6283).
- **Related tests:** [AlgebraicSimplifierTest.ZeroSizedPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4359).

### indexing-pad-interior-unit

```text
pad(X,c,p) -> pad(X,c,p_without_unit_interior)
```

- **Conditions:** Clear positive interior padding only on input axes of extent one.
- **RHS attributes:** Set corresponding interior entries to zero.
- **Source:** [HandlePad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6078-L6283).
- **Related tests:** [AlgebraicSimplifierTest.TrivialInteriorPadding](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6110).

### indexing-pad-identity

```text
pad(X,c,zero_padding) -> X
```

- **Conditions:** No effective padding and compatible result shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandlePad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6078-L6283).
- **Related tests:** [AlgebraicSimplifierTest.RemoveNoopPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5898).

### indexing-pad-negative

```text
pad(X,c,p) -> slice(pad(X,c,max_edges(p,0)),a,b,1)
```

- **Conditions:** Negative edge padding present; legal crop sizes.
- **RHS attributes:** a=max(-low,0); b=nonnegative_pad_shape+min(high,0); preserve interior padding.
- **Source policy:** enable_negative_padding_replacement.
- **Source:** [HandlePad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6078-L6283).
- **Related tests:** [AlgebraicSimplifierTest.NegativePadding](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5961).

### indexing-reverse-trivial

```text
reverse(X,d) -> X
```

- **Conditions:** All reversed dimensions have extent one.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7097-L7203).
- **Related tests:** [AlgebraicSimplifierTest.ReversalOfTrivialDimensionsToBitcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7801).

### indexing-reverse-compose

```text
reverse(reverse(X,p),q) -> reverse(X,symmetric_difference(p,q))
```

- **Conditions:** Valid reverse axis sets; empty resulting set returns X.
- **RHS attributes:** Axis symmetric difference.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7097-L7203).
- **Related tests:** [AlgebraicSimplifierTest.RemoveIdenticalNestedReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13711), [AlgebraicSimplifierTest.ShrinkNestedReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13732).

### indexing-reverse-elementwise

```text
reverse(f(X,broadcast(c)),d) -> f(reverse(X,d),broadcast(c))
```

- **Conditions:** f is elementwise binary, excluding compare and rng; other operand is broadcast scalar constant; either operand order.
- **RHS attributes:** Keep d and scalar broadcast.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7097-L7203).
- **Related tests:** [AlgebraicSimplifierTest.SwapConstantEwboWithReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13754), [AlgebraicSimplifierTest.SwapConstantEwboWithReverse2](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13779).

### indexing-reverse-inserted-unit-dims

```text
reverse(reshape_insert_ones(X),d) -> reshape_insert_ones(reverse(X,d_new))
```

- **Conditions:** Reshape only inserts unit dimensions.
- **RHS attributes:** Discard trivial reversed axes; subtract preceding inserted-axis count from other reverse positions.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [HandleReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7097-L7203).
- **Related tests:** [AlgebraicSimplifierTest.SwapNonContiguousDegenerateDimensionAddingReshapeAndReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13972).

### indexing-rotate-compose

```text
rotate(rotate(X,p,s),q,t) -> rotate(X,merged_axes,merged_shifts)
```

- **Conditions:** Shuffle type rotate; compose matching dimensions by adding shifts.
- **RHS attributes:** Aggregate per axis; normalize shifts modulo axis extents.
- **Source:** [HandleShuffle](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7205-L7281).
- **Related tests:** [AlgebraicSimplifierTest.FusesShuffleRotates](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15792).

### indexing-rotate-canonicalize

```text
rotate(X,d,s) -> rotate(X,d_filtered,s_normalized) or X
```

- **Conditions:** Drop axes of extent <=1, zero normalized shift, or splat along axis (constant-splat or broadcast-created axis).
- **RHS attributes:** Sort axes; normalize signed shifts modulo extents; drop no-op axes.
- **Source:** [HandleShuffle](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7205-L7281).
- **Related tests:** [AlgebraicSimplifierTest.RemovesShuffleRotateOnSplatBroadcast](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15816), [AlgebraicSimplifierTest.RemovesNoopDimAndCanonicalizesShift](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15835).

### indexing-slice-identity

```text
slice(X,full_bounds,1) -> X
```

- **Conditions:** Replacement has compatible shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.RemoveNoopSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6168).

### indexing-slice-pad-only

```text
slice(pad(X,c,p),a,b,s) -> broadcast(c,output_shape,[])
```

- **Conditions:** On at least one axis all selected coordinates lie in edge padding.
- **RHS attributes:** Broadcast padding scalar to slice result shape.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfPadLow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8833), [AlgebraicSimplifierTest.SliceOfPadHigh](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8854), [AlgebraicSimplifierTest.SliceOfPadSomeDimsInPadding](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8960).

### indexing-slice-pad-cancel

```text
slice(pad(X,c,p),a,b,s) -> X
```

- **Conditions:** Slice selects exactly original X positions: starts equal low padding; strides equal interior+1; compatible result shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.RemoveNoopSliceOfPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L5928).

### indexing-slice-inside-pad

```text
slice(pad(X,c,p),a,b,1) -> slice(X,a-low,b-low,1)
```

- **Conditions:** Entire slice inside unpadded region; no interior padding and unit strides.
- **RHS attributes:** Subtract low padding from starts/limits.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfPadMidNonScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8875), [AlgebraicSimplifierTest.SliceOfPadMidScalar](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8939).

### indexing-slice-update-cancel

```text
slice(update(X,Y,s),a,b,t) -> Y
```

- **Conditions:** Constant update starts, clamped to legal bounds; slice starts and sizes exactly match update, with unit strides.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfDynamicUpdateSliceMatching](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7963), [AlgebraicSimplifierTest.SliceOfDynamicUpdateSliceClampedMatching](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7983).

### indexing-slice-inside-update

```text
slice(update(X,Y,s),a,b,t) -> slice(Y,a-clamp(s),b-clamp(s),t)
```

- **Conditions:** Constant update starts; slice entirely within update after clamping.
- **RHS attributes:** Shift slice bounds by effective update starts; preserve strides.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfDynamicUpdateSliceSubslice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8003), [AlgebraicSimplifierTest.SliceOfDynamicUpdateSliceStrided](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8023).

### indexing-slice-compose

```text
slice(slice(X,a,b,1),c,d,1) -> slice(X,a+c,a+d,1)
```

- **Conditions:** Both slices unstrided.
- **RHS attributes:** Componentwise addition of starts to outer bounds.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfSliceToSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6192).

### indexing-slice-concat-scalar

```text
slice(concat(X1,...,Xn),[a:a+1]) -> slice(Xj,[a-offset_j:a-offset_j+1])
```

- **Conditions:** Rank-one effective-scalar slice; choose operand containing a. Return Xj directly if shapes match.
- **RHS attributes:** Compute cumulative concat offsets.
- **Source:** [TrySimplifyScalarSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7283-L7333), [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfConcatScalarInput](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8983), [AlgebraicSimplifierTest.SliceOfConcatNonScalarInput](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9005).

### indexing-slice-concat

```text
slice(concat(X1,...,Xn,d),a,b,s) -> slice(concat(overlapping operands,d),a-offset,b-offset,s)
```

- **Conditions:** Unit strides on every axis. Entire selected operand subset must be smaller than the original operand list, unless a single compatible operand replaces the slice.
- **RHS attributes:** Find start/end operands from cumulative extents; rebase bounds; retain other axes.
- **Source:** [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfConcat](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11843), [AlgebraicSimplifierTest.SliceOfMultipleConcatOperands](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11859).

### indexing-slice-reshape

```text
slice(reshape(X),a,b,1) -> reshape(slice(X,a_new,b_new,1))
```

- **Conditions:** Only reshape output axis zero is sliced; flattened offset and size align to operand subdimension boundaries without crossing a rectangular subrange.
- **RHS attributes:** Compute flattened offset/count; decompose backward into input starts and limits.
- **Source policy:** Source applies in layout-insensitive mode.
- **Source:** [TryToReorderSliceAndReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7335-L7448), [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceOfReshapeToReshapeOfSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6255), [AlgebraicSimplifierTest.SliceOfReshapeToReshapeOfSliceNonZeroStartMisaligned](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6320).

### indexing-slice-reverse

```text
slice(reverse(X,d),a,b,s) -> reverse(slice(X,a_new,b_new,s),d)
```

- **Conditions:** Valid possibly strided slice.
- **RHS attributes:** On reversed axis with extent n and m selected values: new_start=n-1-(a+(m-1)*s); new_limit=n-a; preserve stride.
- **Source:** [TryToReorderSliceAndReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7452-L7494), [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceReverse](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4679), [AlgebraicSimplifierTest.SliceReverseNonUnitEvenOddStrides](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4717).

### indexing-slice-singleton-stride

```text
slice(X,a,b,s) -> slice(X,a,b_new,1) on singleton selections
```

- **Conditions:** A strided axis selects at most one element.
- **RHS attributes:** For each redundant stride set limit=start+1 and stride=1; preserve other bounds.
- **Source:** [RemoveRedundantStride](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7496-L7533), [HandleSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L7535-L7997).
- **Related tests:** [AlgebraicSimplifierTest.SliceRedundantStrides](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L4753).

### indexing-dynamic-slice-identity

```text
slice_dynamic(X,s,shape(X)) -> X
```

- **Conditions:** Scalar result or compatible operand/result shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source policy:** Skip async computations.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.TrivialDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7874).

### indexing-dynamic-slice-zero-index

```text
slice_dynamic(X,s,z) -> slice_dynamic(X,s_with_full_axes_zero,z)
```

- **Conditions:** On each changed axis slice size equals input extent, so clamping forces start zero.
- **RHS attributes:** Set corresponding starts to scalar zero.
- **Source policy:** Skip async computations.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.DynamicSliceTrivialIndices](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12971).

### indexing-dynamic-slice-unit-reshape

```text
slice_dynamic(reshape_units(X),s,z) -> reshape_units(slice_dynamic(X,s_new,z_new))
```

- **Conditions:** Reshape only inserts/deletes unit axes.
- **RHS attributes:** Map unchanged nonunit axes; missing unit axes use zero start and size one.
- **Source policy:** Source applies in layout-insensitive mode. Skip async computations.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.DynamicSliceOfTrivialReshape](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8805).

### indexing-dynamic-slice-transpose

```text
slice_dynamic(transpose(X,p),s,z) -> transpose(slice_dynamic(X,permute(s,inverse(p)),permute(z,inverse(p))),p)
```

- **Conditions:** Valid transpose and dynamic slice.
- **RHS attributes:** Permute starts/sizes by inverse(p); retain p.
- **Source policy:** Source applies in layout-insensitive mode. Skip async computations.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.DynamicSliceOfTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8774).

### indexing-dynamic-slice-static

```text
slice_dynamic(X,c,z) -> slice(X,min(c,shape(X)-z),min(c,shape(X)-z)+z,1)
```

- **Conditions:** All starts are nonnegative integral constants representable by the extraction helper; X nonconstant.
- **RHS attributes:** Clamp each constant start to input_extent-slice_size; add slice size for limit.
- **Source policy:** Conversion enabled; skip async; same memory space if layout-sensitive.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.ConstantDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7897), [AlgebraicSimplifierTest.DynamicSliceConvertedToConstantSliceWithConstantIndices](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L14805).
- **Note:** This pinned source skips negative constant starts instead of materializing max(c,0).

### indexing-dynamic-slice-iota

```text
slice_dynamic(iota(n)*scale,start,[1]) -> reshape(convert(clamp(start,0,n-1))*scale,[1])
```

- **Conditions:** Rank-one scalar-sized slice; base iota optionally multiplied by scalar literal; index type S32/U32 with compatible width.
- **RHS attributes:** Clamp effective index; preserve element signedness/conversion and optional scale.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.DynamicSliceOfIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12947), [AlgebraicSimplifierTest.DynamicSliceOfStridedIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L13037).

### indexing-dynamic-slice-compose

```text
slice_dynamic(slice_dynamic(X,s,a),t,b) -> slice_dynamic(X,clamp(s,0,shape(X)-a)+clamp(t,0,a-b),b)
```

- **Conditions:** Input extents fit index element types; source widens/converts indices as needed; index addition cannot overflow.
- **RHS attributes:** Add independently clamped starts; outer sizes b.
- **Source:** [HandleDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8022-L8309).
- **Related tests:** [AlgebraicSimplifierTest.DynamicSliceOfDynamicSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15685), [AlgebraicSimplifierTest.DynamicSliceOfDynamicSliceNarrowIndexOverflow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15709).

### indexing-update-broadcast-to-pad

```text
update(broadcast(c),Y,s) -> pad(Y,c,p)
```

- **Conditions:** Scalar-broadcast base; constant starts successfully extracted and clamped.
- **RHS attributes:** low=clamp(s,0,base_shape-shape(Y)); high=base_shape-low-shape(Y); interior=0.
- **Source policy:** Replacement option enabled; skip async and host-offloading patterns.
- **Source:** [HandleDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8311-L8608).
- **Related tests:** [AlgebraicSimplifierTest.DynamicUpdateSliceOfBroadcastToPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9825), [AlgebraicSimplifierTest.DynamicUpdateSliceOfBroadcastToPadOob](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12919).

### indexing-update-full

```text
update(X,Y,s) -> Y
```

- **Conditions:** Update shape equals result shape.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8311-L8608).
- **Related tests:** [AlgebraicSimplifierTest.TrivialDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L7925).

### indexing-update-pad-to-concat

```text
update(pad(X,c,p),Y,s) -> concat(X,Y,d) or concat(Y,X,d)
```

- **Conditions:** Exactly one nonnegative edge-padded axis; no interior padding; constant clamped starts; update fills only that padded edge; all other axes fully covered.
- **RHS attributes:** Concat dimension is padded axis; order follows low/high padding.
- **Source policy:** Compatible layout if layout-sensitive.
- **Source:** [HandleDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8311-L8608).
- **Related tests:** [AlgebraicSimplifierTest.DynamicUpdateSliceOfPadToConcatHigh](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8061), [AlgebraicSimplifierTest.DynamicUpdateSliceOfPadToConcatLow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L8103).

### indexing-update-zero-index

```text
update(X,Y,s) -> update(X,Y,s_with_full_axes_zero)
```

- **Conditions:** On each changed axis update extent equals base extent.
- **RHS attributes:** Replace corresponding starts with zero.
- **Source policy:** Skip async computations.
- **Source:** [HandleDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8311-L8608).
- **Related tests:** [AlgebraicSimplifierTest.DynamicUpdateSliceTrivialIndices](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10047).

### indexing-update-empty

```text
update(X,empty,s) -> X
```

- **Conditions:** Any update dimension has extent zero.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8311-L8608).
- **Related tests:** [AlgebraicSimplifierTest.DynamicUpdateSliceZeroUpdate](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L9802).

### indexing-update-compose

```text
update(X,update(slice_dynamic(X,s,a),Y,t),s) -> update(X,Y,clamp(s,0,shape(X)-a)+clamp(t,0,a-shape(Y)))
```

- **Conditions:** Inner base is a dynamic slice of the same X at the same outer starts; input extents fit index types; checked combination of indices.
- **RHS attributes:** Compose clamped starts; preserve base and final update.
- **Source:** [HandleDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L8311-L8608).
- **Related tests:** [AlgebraicSimplifierTest.DynamicUpdateSliceOfDynamicUpdateSlice](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15746), [AlgebraicSimplifierTest.DynamicUpdateSliceOfDynamicUpdateSliceNarrowIndexOverflow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15772).

### indexing-gather-empty

```text
gather(empty,indices,g) -> zeros(output_shape)
```

- **Conditions:** Operand has zero elements; source constructs a zero-like result of gather shape.
- **RHS attributes:** Preserve gather result shape and element type.
- **Source:** [HandleGather](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4514-L4738).
- **Related tests:** No dedicated test identified in the captured file.

### indexing-gather-small

```text
gather(X,indices,g) -> select_chain(indices>=i,broadcast(slice(X,i)))
```

- **Conditions:** Rank-one operand; index_vector_dim equals index rank (implicit scalar vector); exactly one collapsed slice dimension.
- **RHS attributes:** Start with X[0]; in ascending i select X[i] when indices>=i. This clamps out-of-range indices to endpoint elements.
- **Source policy:** Operand extent <= very_small_gather_size.
- **Source:** [HandleGather](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4514-L4738).
- **Related tests:** No dedicated test identified in the captured file.
- **Note:** The source test name is shared by multiple fixtures; links identify related fixtures, not a proof of coverage.

### indexing-gather-pad

```text
gather(pad(X,c,p),indices,g) -> pad(gather(X,indices_unpadded,g_new),c,p_new)
```

- **Conditions:** Padding only on operand passthrough axes, or explicit batch axes padded identically in indices; indices pad no other axes.
- **RHS attributes:** Shrink gather slice sizes on padded axes; map padding to result axes; unpad indices when batch axes padded.
- **Source:** [HandleGather](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4514-L4738).
- **Related tests:** [AlgebraicSimplifierTest.GatherOfPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10452), [AlgebraicSimplifierTest.GatherOfPadWithPaddedBatchDims](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10571), [AlgebraicSimplifierTest.GatherOfPadStartIndicesPaddedDifferently](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10543).

### indexing-gather-reshape-pad

```text
gather(reshape(pad(X,c,p)),indices,g) -> pad(gather(reshape(X),indices,g_new),c,p_new)
```

- **Conditions:** Padded dimensions are unmodified by reshape and passthrough through gather.
- **RHS attributes:** Restrict reshape sizes and gather slice sizes; map padding to output axes.
- **Source:** [HandleGather](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4514-L4738).
- **Related tests:** [AlgebraicSimplifierTest.GatherOfReshapeOfPad](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10616), [AlgebraicSimplifierTest.GatherOfReshapeOfPad4](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L10695).

### indexing-scatter-empty-update

```text
scatter(X,indices,empty_updates,g,f) -> X
```

- **Conditions:** All update tensors have zero elements; compatible replacement shapes.
- **RHS attributes:** No new axis or indexing attributes.
- **Source:** [HandleScatter](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10064-L10085).
- **Related tests:** No dedicated test identified in the captured file.

### indexing-scatter-scalar-map

```text
scatter(X,empty_index_vector,Y,g,f) -> map(f,X,Y)
```

- **Conditions:** Single scatter operand; indices have zero elements; input/update/result shapes equal.
- **RHS attributes:** Preserve combining computation f.
- **Source:** [HandleScatter](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10064-L10085).
- **Related tests:** [AlgebraicSimplifierTest.ScalarScatter](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L12532).

### indexing-sort-trivial

```text
sort(X,d) -> X; sort(keys,values,d) -> tuple(keys,values)
```

- **Conditions:** Input empty or sorted-axis extent <=1.
- **RHS attributes:** Preserve tuple ordering for key/value sort.
- **Source:** [HandleSort](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10087-L10100).
- **Related tests:** [AlgebraicSimplifierTest.RemoveNoopSort](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6430), [AlgebraicSimplifierTest.ReplaceEffectiveScalarKeyValueSortWithTuple](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L6447).

### indexing-transpose-scatter

```text
transpose(scatter(X,i,U,g),p) -> scatter(transpose(X,p),i,transpose(U,[0]++(p+1)),g_new)
```

- **Conditions:** Single-operand simplified scatter; no operand/index batching dimensions.
- **RHS attributes:** scatter_dims_to_operand_dims becomes inverse(p)[old_dim]; update transpose preserves leading scatter axis.
- **Source policy:** Option enabled; one scatter user; written run length must not shrink.
- **Source:** [TryFoldTransposeIntoScatter](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10159-L10232), [HandleTranspose](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L10234-L10571).
- **Related tests:** [AlgebraicSimplifierTest.FoldTransposeIntoScatter](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15544), [AlgebraicSimplifierTest.DoNotFoldTransposeIntoScatterWhenWritesBecomeStrided](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L15591).

### indexing-remainder-iota

```text
iota(n) % broadcast(N) -> iota(n)
```

- **Conditions:** Positive effective-scalar integer literal N >= n; n-1 fits the iota type (or n=0).
- **RHS attributes:** Keep iota axis and shape.
- **Source:** [HandleRemainder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6494-L6584).
- **Related tests:** [AlgebraicSimplifierTest.RemainderOfIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11640).

### indexing-remainder-offset-iota

```text
(iota(n)+broadcast(N)) % broadcast(N) -> iota(n) % broadcast(N)
```

- **Conditions:** Same scalar literal node N; N>0, n>0; n-1 and checked N+n-1 fit element type.
- **RHS attributes:** Keep iota and scalar broadcast.
- **Source:** [HandleRemainder](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L6494-L6584).
- **Related tests:** [AlgebraicSimplifierTest.RemainderOfNPlusIota](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11655), [AlgebraicSimplifierTest.RemainderOfNPlusIotaOverflow](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11672), [AlgebraicSimplifierTest.RemainderOfNegativeNPlusIotaNotSimplified](https://github.com/openxla/xla/blob/e02127ba9c9c48912b0e6776fe5bfe3843d81042/xla/hlo/transforms/simplifiers/algebraic_simplifier_test.cc#L11686).
