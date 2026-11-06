# Nonzero reciprocal normalization: companion experiment

The symbolic scalar normalizer now rewrites `x / y` as `x * (1 / y)`
when the algebraic policy is active, the divisor is floating-point typed, and
the existing nonzero analysis proves it nonzero. The reciprocal `1 / y` is
kept in canonical form. The rule does not apply to strict policy, integer
division, zero, or a divisor whose nonzero status is unknown.

The [companion case](../../tepl/rules/experiments/reciprocal_nonzero_literal.tepl)
uses the nonzero literal `2:f32` as a scalar broadcast divisor. With the new
normalizer it proves with `additional-guard: true`, infers the forced empty
broadcast mapping, compiles and validates. This restricted case does **not**
replace the original XLA case, whose arbitrary scalar divisor is not known
nonzero. Rechecking the original cases with this compiler still returns unknown:

- `catalog_broadcast_divide_broadcast`: normalized coefficients, tensor accesses,
  or reduction domains differ. Nonzero is not established for its scalar input.
- `catalog_broadcast_transpose_broadcast`: supplied metadata uses a comprehension
  outside the supported pointwise definedness fragment.

No change to the metadata checker is claimed. The original 30/32 result and the
separate expanded run remain frozen and are not recomputed by this companion
experiment. The 2:f32 case uses the algebraic floating policy; it is not a
strict IEEE floating-point equivalence claim.

To reproduce from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check labs/attribute_inferences/xla_rewrite/tepl/rules/experiments/reciprocal_nonzero_literal.tepl --semantic-policy=algebraic --rule reciprocal_nonzero_literal
bazel test //tests:scalar_test //tests:semantic_test //tests:ordered_synthesis_test --test_output=errors
```
