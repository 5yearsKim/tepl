# Scalar value semantics

An operation's `semantics` block describes the value at one output coordinate.
Its `axes` block supplies the coordinate relationships. Registered scalar
functions can be nested inside this value expression:

```tepl
dialect Example {
  op exponential(x: tensor) -> tensor {
    dtype(a) { assert is_float(a); yield a; }
    axes { bind S { x: all_axes(x) }; yield S; }
    semantics { yield exp(x); }
  }
}
```

A scalar application stores its function identity, value arguments, metadata
parameters and result dtype. The compiler follows tensor reads recursively through
its arguments. For example, `exp(Read(x, IndexReverse(Coords(S), attrs.dims)))`
still tracks the reversed access to `x`. A scalar function never introduces a new
tensor capture or reduction group.

## Functions and types

| Functions | Value arguments | Types |
|---|---:|---|
| `add`, `mul`, `sub`, `neg` | 2, 2, 2, 1 | Numeric |
| `min`, `max`, `div`, `pow`, `rem` | 2 | Matching numeric types |
| `abs` | 1 | Numeric |
| `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `cbrt`, `erf`, `logistic` | 1 | Floating point |
| `not`, `and`, `or`, `xor` | 1, 2, 2, 2 | Boolean |
| `select` | 3 | Boolean predicate and two matching branch types |
| `compare` | 2 | Matching numeric or Boolean types; Boolean result |
| `clamp` | 3 | Matching numeric types |

The declared dtype contract must establish these requirements. The compiler does
not assume new input restrictions to justify a scalar function.

`compare(a, b, direction)` also takes an integer metadata parameter. The direction
encoding is EQ=0, NE=1, GE=2, GT=3, LE=4, LT=5. For an operation with a required
`direction: index` field, use `compare(lhs, rhs, attrs.direction)`. Function
identity and parameter equality are part of value comparison.

The registry is part of the compiler. These functions are separate from metadata
builtins and do not call host functions. Adding a new registry entry gives it
argument and coordinate tracking; algebraic laws require separate implementation.

Existing `Const`, `Add`, binary `Product`, `Read` and sum `Reduce` syntax remains
supported. Lowercase `add`, `mul`, `sub` and `neg` lower to the existing arithmetic
representation. `Const` takes a signed integer literal. Numeric graph literals,
such as the `2` in a scalar `(power X 2)` pattern, also participate in semantic
checking. Constant payloads stored in operation attributes are not scalar values.

## Normalization

The compiler first normalizes each scalar argument. Polynomial expressions still
use sums of terms. A term's factors can now include scalar applications as well
as tensor reads and numeric literals:

```text
exp(Add(X, Const(0)))  -> exp(X)
min(min(X, Y), X)      -> min(X, Y)
not(not(P))           -> P
pow(X, Const(2))       -> Product(X, X)
Product(exp(X),exp(Y)) -> exp(Add(X,Y))
```

Unchanged scalar applications compare by function, parameters, dtype and arguments,
including the coordinates of every nested tensor read. Thus two `exp` expressions
can agree after their arguments normalize, while `exp(X)` and `log(X)` remain
different.

The `algebraic` policy enables shared associativity, commutativity and idempotence
properties where declared. Additional laws cover min/max absorption, Boolean
identities, selection, reflexive comparisons, small integer powers and selected
exp/log/sqrt/abs identities. Monotone functions can move through min/max. Functions
without an implemented law remain intact.

Domain conditions still matter. `log(exp(X))` simplifies under ideal real
arithmetic. `log(sqrt(X))` needs a positive argument; this is recognized for
structurally positive expressions such as `exp(Y)`, but not for arbitrary tensor
values. General division reassociation is not assumed. There is no general
tensor-value range analysis.

These laws use the existing ideal arithmetic policy. They do not establish IEEE
rounding, NaN, signed-zero or overflow equivalence. Under `strict`, generic scalar
congruence and coordinate tracking remain available; algebraic simplifications
are disabled. Scalar calls do not move through reductions without a justified
law. `Reduce` remains a summation binder; scalar min/max do not define min/max
reductions.
