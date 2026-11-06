# Host dependencies and native TEPL logic

No evaluation rule contains a host call or a host `fn` parameter. The recorded
compilation cases use the current parser and executable engine. Shape/dtype
metadata supplied with graph inputs is separate from calling application
functions inside rules.

The 33 `needs_extension` assessments describe unmet requirements of the general
source cases. They do not mean every instance needs a compiler change. A constant
represented by ordinary numeric attributes can be inspected by `where` and used
by `derive`. An opaque `elements` payload cannot be inspected that way.

## Two implemented native cases

- [Power of two](../../tepl/rules/structural/native_power_two_splat.tepl) matches the
  existing `fill` operation, then checks its `value`, shape and dtype attributes.
  It handles any rank without a splat-recognition callback.
- [Unsigned absolute value](../../tepl/rules/structural/native_abs_unsigned.tepl) uses
  `is_unsigned_integer(T)` to establish nonnegativity. It needs no range-analysis
  callback. The general signed/floating case remains unresolved.

Both rules compile to validated executable JSON. Execution tests require them
to fire and compare independent dense results. Negative cases check that an
exponent of three and a signed input do not trigger these rules. These checks
are separate from automatic semantic proofs; this dialect declares no value
semantics for `power` or `abs`.

## What the 33 cases need

| Cases | Count | Can native TEPL logic handle the requirement? |
|---|---:|---|
| Nonnegativity-dependent identities | 7 | Dtype or a narrower producer pattern can establish some cases. General range analysis for an arbitrary tensor is unavailable. |
| Fixed scalar/splat exponents, zero exponential, constant select | 7 | Literal patterns or readable fill attributes can express the conditions. General opaque-payload recognition is unavailable. |
| Comparison bounds and clamp bounds | 3 | Readable integer attributes allow comparisons and `min`; opaque typed constants need a different representation. |
| Power-of-two division/remainder | 4 | Readable integer attributes permit guards and derived shift/mask attributes. Typed shift/bitwise operations must be declared. |
| Materializing a typed reciprocal constant | 1 | General floating/complex payload evaluation and source-type rounding are unavailable. |
| Literal splat, iota and strided-iota recognition | 3 | Readable integer lists permit comprehensions and comparisons. Existing opaque `elements` values cannot be decoded by rules. |
| Rotate canonicalization | 1 | Shape/shift and broadcast-pattern cases use native logic. Opaque literal-splat recognition remains unavailable. |
| Constant-start slice/update transformations | 5 | Match constant producers and read start attributes; clamp/index arithmetic is native. Each fixed operand count needs a corresponding pattern. |
| Remainder with iota | 2 | Readable integer modulus attributes permit range guards and checked arithmetic. Opaque modulus extraction remains unavailable. |

The table identifies approaches, not completed ports of every source variant.
Only linked compilation and execution cases are measured. Exposing numeric
constant attributes is a dialect/graph representation choice; adding an
unverified `nonnegative` flag would merely assume the required fact.
