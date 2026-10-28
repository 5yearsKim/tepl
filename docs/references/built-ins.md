# Built-in functions

TEPL builtins evaluate shapes, dtypes, and rule conditions. They operate on
metadata and ordinary values; they do not execute tensor operations.

Use an unprefixed call, such as `len(Dims)` or `is_numeric(T)`. The compiler
checks the function name, argument count, argument types, and expression context.

## Contexts and availability

| Context | Inputs and purpose |
| --- | --- |
| Rule `where` | Check a matched candidate using captured dimensions, dtype bindings, descriptor fields, and local values. The final expression must return `bool`. |
| Rule `derive` | Calculate replacement attribute descriptors using constructors and builtins. Each assignment must produce a descriptor. |
| Operation `shape` | Compute an output shape from operand shape lists and operation attributes. A variadic parameter receives a list of shapes. |
| Operation `dtype` | Compute an output dtype from operand dtype values and operation attributes. A variadic parameter receives a list of dtypes. |
| Operation `axes` | Declare axis bindings, output order, and extent constraints using operand and axis queries. |

Shape and dtype blocks share the metadata expression language. Function
availability still depends on the block:

The shape helpers are also available in dtype blocks when working with integer
lists. Their availability does not mean that a dtype block receives operand
shapes; its operand parameters remain dtype values.

### Signature notation

The following tables use descriptive types, rather than TEPL declarations:

- `Integer` is a checked signed 128-bit integer in shape and dtype programs.
- `Bool` and `DType` are metadata Boolean and dtype values.
- `T[]` is a homogeneous list. `T` may itself be a list, Boolean, integer, or
  dtype. All occurrences of `T` within a call must have the same type.
- `L` is a rule `u64[]`, containing unsigned 64-bit values.
- `I` is one shared rule integer type: either unsigned `u64` or signed `i64`.

Attribute schema roles `axis` and `extent` use the unsigned `u64` rule type;
`coordinate` uses signed `i64`. Only `axis` fields can request `auto` in a
replacement descriptor. `axes` remains the operation declaration for logical
axis structure; it is separate from the `axis` attribute role.

Rule builtins do not implicitly convert between signed and unsigned integers.
The numeric builtins have no floating-point overloads.
Use `i64(x)` or `u64(x)` for an explicit checked integer conversion. Both
accept either integer type and preserve the mathematical value. A conversion
fails if the value is outside the destination range; it never wraps or clamps.
In shape/dtype programs these conversions check the same 64-bit ranges even
though ordinary metadata arithmetic uses signed 128-bit integers.


Empty generic lists need an element type from context. `concat(s, [])` can use
the type of `s`, and `sum([])` requires integers. A standalone `len([])` cannot
infer the empty list's element type and is rejected.

## Function inventory

<!-- BEGIN BUILTIN CATALOG -->

### Expression function inventory

This table is generated from the compiler catalog. Feature groups do not grant
section access. The axes overloads are listed separately below.

| Function | Group | Metadata signature | Rule signature | Allowed sections |
| --- | --- | --- | --- | --- |
| `len` | list | `(T[]) -> Integer` | `(T[]) -> u64` | `shape`, `dtype`, `where`, `derive` |
| `range` | list | `(Integer) -> Integer[]` | `(u64) -> u64[]` | `shape`, `dtype`, `where`, `derive` |
| `concat` | list | `(T[], T[], ...) -> T[]` | `(T[], T[], ...) -> T[]` | `shape`, `dtype`, `where`, `derive` |
| `gather` | list | `(T[], Integer[]) -> T[]` | `(T[], u64[]) -> T[]` | `shape`, `dtype`, `where`, `derive` |
| `exclude` | list | `(T[], T[]) -> T[]` | `(T[], T[]) -> T[]` | `shape`, `dtype`, `where`, `derive` |
| `slice` | list | `(T[], Integer, Integer) -> T[]` | `(T[], u64, u64) -> T[]` | `shape`, `dtype`, `where`, `derive` |
| `replace` | list | `(T[], Integer, T) -> T[]` | `(T[], u64, T) -> T[]` | `shape`, `dtype`, `where`, `derive` |
| `all` | list | `(Bool[]) -> Bool` | `(bool[]) -> bool` | `shape`, `dtype`, `where`, `derive` |
| `any` | list | `(Bool[]) -> Bool` | `(bool[]) -> bool` | `shape`, `dtype`, `where`, `derive` |
| `contains` | list | `(T[], T) -> Bool` | `(T[], T) -> bool` | `shape`, `dtype`, `where`, `derive` |
| `is_disjoint` | list | `(T[], T[]) -> Bool` | `(T[], T[]) -> bool` | `shape`, `dtype`, `where`, `derive` |
| `filter` | list | `(LogicalAxis[], LogicalAxisSet) -> LogicalAxis[]` | `(LogicalAxis[], LogicalAxisSet) -> LogicalAxis[]` | `where`, `derive` |
| `subsequence` | list | `(LogicalAxis[], LogicalAxis[]) -> Bool` | `(LogicalAxis[], LogicalAxis[]) -> bool` | `where`, `derive` |
| `difference` | list | `(LogicalAxisSet, LogicalAxisSet) -> LogicalAxisSet` | `(LogicalAxisSet, LogicalAxisSet) -> LogicalAxisSet` | `where`, `derive` |
| `i64` | integer | `(Integer) -> Integer` | `(I) -> i64` | `shape`, `dtype`, `where`, `derive` |
| `u64` | integer | `(Integer) -> Integer` | `(I) -> u64` | `shape`, `dtype`, `where`, `derive` |
| `sum` | integer | `(Integer[]) -> Integer` | `(u64[]) -> u64` | `shape`, `dtype`, `where`, `derive` |
| `product` | integer | `(Integer[]) -> Integer` | `(u64[]) -> u64` | `shape`, `dtype`, `where`, `derive` |
| `min` | integer | `(Integer, Integer) -> Integer` | `(I, I) -> I` | `shape`, `dtype`, `where`, `derive` |
| `max` | integer | `(Integer, Integer) -> Integer` | `(I, I) -> I` | `shape`, `dtype`, `where`, `derive` |
| `floor_div` | integer | `(Integer, Integer) -> Integer` | `(I, I) -> I` | `shape`, `dtype`, `where`, `derive` |
| `ceil_div` | integer | `(Integer, Integer) -> Integer` | `(I, I) -> I` | `shape`, `dtype`, `where`, `derive` |
| `is_valid_axis_list` | shape | `(Integer[], Integer) -> Bool` | `(u64[], u64) -> bool` | `shape`, `dtype`, `where`, `derive` |
| `broadcast_shape` | shape | `(Integer[], Integer[]) -> Integer[]` | `(u64[], u64[]) -> u64[]` | `shape`, `dtype`, `where`, `derive` |
| `is_float` | dtype | `(DType) -> Bool` | `(dtype) -> bool` | `dtype`, `where`, `derive` |
| `is_integer` | dtype | `(DType) -> Bool` | `(dtype) -> bool` | `dtype`, `where`, `derive` |
| `is_signed_integer` | dtype | `(DType) -> Bool` | `(dtype) -> bool` | `dtype`, `where`, `derive` |
| `is_unsigned_integer` | dtype | `(DType) -> Bool` | `(dtype) -> bool` | `dtype`, `where`, `derive` |
| `is_numeric` | dtype | `(DType) -> Bool` | `(dtype) -> bool` | `dtype`, `where`, `derive` |

### Axes function inventory

These functions are callable in operation `axes` blocks. `Positions` is an
integer sequence, `Axes` is a sequence of axis identities, and `Operand` is an
operation operand. `S` is one shared sequence type: `Positions` or `Axes`.
An operand supplies its shape through `shape`, `rank`, or `all_axes`.

| Function | Axes signature |
| --- | --- |
| `len` | `(S) -> Integer` |
| `range` | `(Integer) -> Positions` |
| `concat` | `(S, S, ...) -> S` |
| `gather` | `(Positions, Positions) -> Positions` |
| `is_valid_axis_list` | `(Positions, Integer) -> Bool` |
| `ordered_complement` | `(Positions, Positions) -> Positions` |
| `permute` | `(Axes, Positions) -> Axes` |
| `all_axes` | `(Operand) -> Positions` |
| `rank` | `(Operand) -> Integer` |
| `shape` | `(Operand) -> Positions` |
| `place` | `(Axes, Positions, Axes) -> Axes` |
| `extent_add` | `(Positions, Positions) -> Positions` |
| `extent_sub` | `(Positions, Positions) -> Positions` |
| `extent_mul` | `(Positions, Positions) -> Positions` |
| `extent_floor_div` | `(Positions, Positions) -> Positions` |
| `extent_le` | `(Positions, Positions) -> Bool` |
| `scale_extents` | `(Positions, Integer) -> Positions` |
| `offset_extents` | `(Positions, Integer) -> Positions` |
| `is_valid_extents` | `(Positions) -> Bool` |
| `is_positive_extents` | `(Positions) -> Bool` |

<!-- END BUILTIN CATALOG -->

## List helpers

These functions are available in all four contexts.

| Function | Shape / dtype signature | Rule signature | Behavior |
| --- | --- | --- | --- |
| `len(xs)` | `T[] -> Integer` | `T[] -> u64` | Return the number of elements in the outer list. |
| `range(n)` | `Integer -> Integer[]` | `u64 -> L` | Construct `[0, 1, ..., n - 1]`. `range(0)` is empty. |
| `concat(a, b, ...)` | Two or more `T[] -> T[]` | Two or more `T[] -> T[]` | Join lists in argument order. Nested elements retain their boundaries. |
| `gather(xs, positions)` | `(T[], Integer[]) -> T[]` | `(T[], L) -> T[]` | Select entries in the supplied order. Repeated positions are allowed. |
| `exclude(xs, removed)` | `(T[], T[]) -> T[]` | `(T[], T[]) -> T[]` | Remove every element whose value occurs in `removed`. Preserve the order and duplicates of remaining elements. |
| `slice(xs, start, end)` | `(T[], Integer, Integer) -> T[]` | `(T[], u64, u64) -> T[]` | Copy the half-open interval `[start, end)`. |
| `replace(xs, position, value)` | `(T[], Integer, T) -> T[]` | `(T[], u64, T) -> T[]` | Return a copy with one existing element replaced. |
| `sum(xs)` | `Integer[] -> Integer` | `L -> u64` | Add elements with checked arithmetic. The empty sum is `0`. |
| `product(xs)` | `Integer[] -> Integer` | `L -> u64` | Multiply elements with checked arithmetic. The empty product is `1`. Any zero makes the product `0`. |
| `all(xs)` | `Bool[] -> Bool` | `bool[] -> bool` | Return true when every element is true. `all([])` is true. |
| `any(xs)` | `Bool[] -> Bool` | `bool[] -> bool` | Return true when at least one element is true. `any([])` is false. |
| `contains(xs, value)` | `(T[], T) -> Bool` | `(T[], T) -> bool` | Test whether the value occurs in the list. |
| `is_disjoint(a, b)` | `(T[], T[]) -> Bool` | `(T[], T[]) -> bool` | Test whether the lists have no values in common. |

Positions start at zero. `gather` and `replace` require each position to be
within the list. `slice` requires `0 <= start <= end <= len(xs)`; its bounds are
never clamped. Negative positions do not count backward from the end.

`range` creates its list eagerly. Its bound must be nonnegative and fit the
runtime's list-index domain; an excessive allocation can also fail. `concat`
requires at least two arguments, including when the lists are empty.

`exclude` removes **values**, not positions. For example, in a metadata program:

```text
gather([2, 3, 4], [2, 0, 2])       -> [4, 2, 4]
exclude([2, 3, 2, 4], [2])         -> [3, 4]
slice([2, 3, 4], 1, 3)            -> [3, 4]
replace([2, 3, 4], 1, 8)          -> [2, 8, 4]
concat([[2, 3]], [[4, 5]])         -> [[2, 3], [4, 5]]
```

## Integer helpers

These functions are available in rules, shape programs, and dtype programs.

| Function | Shape / dtype signature | Rule signature | Behavior |
| --- | --- | --- | --- |
| `min(a, b)` | `(Integer, Integer) -> Integer` | `(I, I) -> I` | Return the smaller integer. |
| `max(a, b)` | `(Integer, Integer) -> Integer` | `(I, I) -> I` | Return the larger integer. |
| `floor_div(a, b)` | `(Integer, Integer) -> Integer` | `(I, I) -> I` | Divide and round toward negative infinity. Require `b > 0`. |
| `ceil_div(a, b)` | `(Integer, Integer) -> Integer` | `(I, I) -> I` | Divide and round toward positive infinity. Require `b > 0`. |

Ordinary integer `/` truncates toward zero. The rounded helpers also handle
negative numerators:

```text
7 / 3              -> 2
floor_div(7, 3)    -> 2
ceil_div(7, 3)     -> 3

-7 / 3             -> -2
floor_div(-7, 3)   -> -3
ceil_div(-7, 3)    -> -2
```

## Axis and broadcasting helpers

Both functions are available in rules, shape programs, and dtype programs.

| Function | Shape / dtype signature | Rule signature | Behavior |
| --- | --- | --- | --- |
| `is_valid_axis_list(axes, rank)` | `(Integer[], Integer) -> Bool` | `(L, u64) -> bool` | Require unique axes, each in `[0, rank)`. Invalid axes or rank return false. |
| `broadcast_shape(a, b)` | `(Integer[], Integer[]) -> Integer[]` | `(L, L) -> L` | Compute a shape using right-aligned broadcasting. Incompatible dimensions fail. |

An axis list can be shorter than the rank. To validate a complete permutation,
also require `len(axes) == rank`. An empty axis list is valid for any
nonnegative, representable rank.

Broadcasting treats missing leading dimensions as `1`. Aligned dimensions must
be equal or one of them must be `1`. Zero dimensions are valid: `0` broadcasts
with `1` to `0`, but cannot broadcast with `2`. Negative dimensions fail.

```text
is_valid_axis_list([2, 0], 3)     -> true
is_valid_axis_list([1, 1], 3)     -> false
is_valid_axis_list([-1], 3)       -> false
broadcast_shape([2, 1, 4], [3, 4]) -> [2, 3, 4]
broadcast_shape([], [2, 3])       -> [2, 3]
broadcast_shape([0], [1])         -> [0]
```

`broadcast_shape` supplies metadata rules for dialects that define broadcasting.
Calling it does not insert a tensor broadcast or change an operation's semantics.

## Dtype predicates

Each predicate has signature `DType -> Bool` in a dtype program and
`dtype -> bool` in a rule. They are available in `dtype`, `where`, and `derive`,
and are unavailable in `shape`.

| Function | Returns true for |
| --- | --- |
| `is_float(t)` | `f16`, `bf16`, `f32`, `f64` |
| `is_signed_integer(t)` | `i8`, `i16`, `i32`, `i64` |
| `is_unsigned_integer(t)` | `u8`, `u16`, `u32`, `u64` |
| `is_integer(t)` | Any signed or unsigned integer dtype above. |
| `is_numeric(t)` | Any integer or floating-point dtype above. |

The dtype `bool` is neither integer nor numeric. Test it with `t == bool`.
Dtype equality compares exact element types: `f16` and `bf16` are distinct.
There is no general `dtype_of`, `promote`, or `can_cast` builtin.

## Axes functions

Operation `axes` blocks have their own signatures, listed in the inventory.
Functions available in rules or metadata are not automatically available there.
`len` and `concat` accept axis or position sequences, with one shared sequence
type per call. `concat` requires at least two arguments in every source context.
`gather` selects positions from a position sequence. `permute` performs the same
selection on axis identities, including the supplied order and repetitions;
operation legality may reject a repeated or incomplete output axis sequence.
`ordered_complement(a, b)` uses the value-removal behavior of `exclude(a, b)`
on position sequences.

`shape(operand)` returns its dimensions, `rank(operand)` returns their count,
and `all_axes(operand)` returns `[0, ..., rank - 1]`. Operand references are
accepted by these queries rather than implicitly treated as position sequences.
`place(left, positions, right)` places the left axis sequence at unique valid
positions in a result of length `len(left) + len(right)`, then fills remaining
positions from the right sequence. The position count must equal `len(left)`.

`extent_add`, `extent_sub`, `extent_mul`, and `extent_floor_div` perform checked
elementwise integer arithmetic on equal-length sequences. Rounded divisors must
be strictly positive. `extent_le` tests that every left element is less than or
equal to its corresponding right element and requires equal lengths too.
`scale_extents` multiplies every element by one integer; `offset_extents` adds
one integer to every element. `is_valid_extents` tests for nonnegative elements;
`is_positive_extents` tests for strictly positive elements. Both return true
for an empty sequence. Domain and binding validation enforce the final tensor
extent constraints separately.

Internal proof operations such as `axis_partition`, `inverse`, and
`matched_positions` cannot be called from source. Public axes functions can
lower to internal operations without exposing those operations elsewhere.
For implementation ownership and shared identities, see the
[developer architecture guide](../../src/shared/builtins/README.md).

## Using builtins in each context

### Rule conditions and derived attributes

Rule dimension sequences supply lists, and `dtype` declarations bind element
types through matched tensor declarations. Assuming `t.negate` is imported:

```javascript
rule cancel_negate_small {
    dtype T;
    X: T[Dims...]

    (t.negate (t.negate X)) => X

    where {
        is_numeric(T)
        && len(Dims) <= 4
        && product(Dims) <= 4096
    }
}
```

Rules support list literals, indexing, comprehensions, and local bindings.
Descriptor fields supply values directly, including nested lists. `derive`
constructs descriptors with schema-checked field expressions:

```tepl
derive {
    result.attrs = { ..source.attrs, permutation: gather(source.attrs.permutation, axes) };
}
```

A scalar or list value cannot replace a complete descriptor assignment.
See [attribute construction](../advanced/attributes_and_conditions.md).

### Shape programs

Shape parameters are operand shape lists. Inside an operation with a required
`permutation: axis[]` attribute, a transpose shape program can use:

```javascript
shape(s) {
    assert is_valid_axis_list(attrs.permutation, len(s));
    assert len(attrs.permutation) == len(s);
    yield gather(s, attrs.permutation);
}
```

Shape and dtype programs allow list literals, indexing, conditionals, and
comprehensions. For example, `all([n > 0 for n in s])` checks every dimension.
`all` and `any` consume an already evaluated Boolean list; they do not make
the list's construction lazy.

### Dtype programs

A variadic operand's dtype parameter is a list of dtype values:

```javascript
dtype(inputs...) {
    assert len(inputs) > 0;
    assert all([is_numeric(t) for t in inputs]);
    assert all([t == inputs[0] for t in inputs]);
    yield inputs[0];
}
```

Statements execute in order. The nonempty assertion protects the later indexing.
The final result of a dtype program must be one dtype; a shape program must
yield a list of integer dimensions.

## Name resolution and failures

Every call resolves to a builtin. Unknown names are errors. Application callbacks
and host-function template parameters are unsupported.

All occurrences of a call's generic element type must agree. For example,
`concat([f32], [i32])` is a valid dtype list, while `concat([f32], [1])` mixes
dtype and integer values and fails type checking. Optional or opaque attributes
cannot be inspected by metadata expressions; required `u64`, `i64`, `bool`,
and `dtype` attributes, including lists, can be read through `attrs.field`.

Integer arithmetic and reductions detect overflow. Division by zero, invalid
indices or ranges, nonpositive rounded divisors, and incompatible broadcasting
also fail. A product containing zero returns zero without overflowing its
intermediate multiplication. A yielded shape must contain nonnegative
dimensions that fit `u64`.

- In a rule, a builtin failure rejects the rewrite candidate before insertion.
  Pure conditions can prune search when their bindings are ready. Extended rule
  expressions run in the [greedy engine](../advanced/greedy_rewriting.md).
  Rust/C++ generation for these forms remains deferred and reports a diagnostic.
- In a shape or dtype program, a failed assertion or evaluated builtin makes
  inference invalid. Missing required tensor metadata can leave inference
  unknown. Assertions remain active in optimized builds.
- `&&`, `||`, and metadata conditionals evaluate only the required branches.
  Function arguments are evaluated once, in source order.

For tensor metadata concepts and operation examples, see the
[shape reference](shape.md) and [dtype reference](dtype.md).
The [builtin catalog](../../src/shared/builtins/catalog.def) defines the supported
names, signatures, and contexts.

## Logical-axis conditions

Named node properties expose logical axes without compiler node numbers. See
[the syntax and complete example](../advanced/attributes_and_conditions.md#semantic-axis-properties).
`node.axes.G.members` is a `LogicalAxisSet`; `node.axes.G.order` and
`node.axes.order` are `LogicalAxis[]`. Identities represent source-port
classes, independently of extents and numeric positions.

`filter(sequence, members)` preserves sequence order;
`difference(members, other)` returns a logical set;
`subsequence(required, actual)` tests relative order. Source logical-axis
properties lower to pure metadata expressions and do not impose an extent >= 2
restriction. Numeric node-query functions are internal.

Use `check --semantic-policy=algebraic --suggest-where --rule NAME` to search
for a verified sufficient source restriction. Every suggestion is proved under
the original conditions and checked for a legal witness; the bounded search
does not guarantee a weakest condition. Add a reviewed suggestion to source
before compiling. Runtime feasibility queries, including `axis_domain()` and
inferred RHS boundary queries, are unsupported.
