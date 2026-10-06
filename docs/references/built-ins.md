# Built-in functions

TEPL builtins evaluate shapes, dtypes, and rule conditions. They operate on
metadata and ordinary values; they do not execute tensor operations.

Use an unprefixed call, such as `len(Dims)` or `is_numeric(T)`. The compiler
checks the function name, argument count, argument types, and expression context.

## Contexts and availability

| Context | Inputs and purpose |
| --- | --- |
| Rule `where` | Check a matched candidate using captured dimensions, dtype bindings, descriptor fields, and host results. Each condition must return `bool`. |
| Rule `derive` | Compute arguments for host functions that construct replacement attribute descriptors. Each assignment must produce a descriptor. |
| Operation `shape` | Compute an output shape from operand shape lists and operation attributes. A variadic parameter receives a list of shapes. |
| Operation `dtype` | Compute an output dtype from operand dtype values and operation attributes. A variadic parameter receives a list of dtypes. |

Shape and dtype blocks share the metadata expression language. Function
availability still depends on the block:

| Functions | Rule `where` / `derive` | `shape` | `dtype` |
| --- | --- | --- | --- |
| List helpers, except `all` and `any` | Yes | Yes | Yes |
| `min`, `max`, `floor_div`, `ceil_div` | Yes | Yes | Yes |
| `all`, `any` | No | Yes | Yes |
| `is_valid_axis_list`, `broadcast_shape` | Yes | Yes | Yes |
| `is_float`, `is_integer`, `is_signed_integer`, `is_unsigned_integer`, `is_numeric` | Yes | No | Yes |

The shape helpers are also available in dtype blocks when working with integer
lists. Their availability does not mean that a dtype block receives operand
shapes; its operand parameters remain dtype values.

### Signature notation

The following tables use descriptive types, rather than TEPL declarations:

- `Integer` is a checked signed 128-bit integer in shape and dtype programs.
- `Bool` and `DType` are metadata Boolean and dtype values.
- `T[]` is a homogeneous list. `T` may itself be a list, Boolean, integer, or
  dtype. All occurrences of `T` within a call must have the same type.
- `L` is a rule `index_list`, containing unsigned 64-bit `index` values.
- `I` is one shared rule integer type: either unsigned `index` or signed `i64`.

Rule builtins do not implicitly convert between signed and unsigned integers.
The numeric builtins have no floating-point overloads.

Empty generic lists need an element type from context. `concat(s, [])` can use
the type of `s`, and `sum([])` requires integers. A standalone `len([])` cannot
infer the empty list's element type and is rejected.

## List helpers

These functions are available in all four contexts, except `all` and `any`,
which are available only in shape and dtype programs.

| Function | Shape / dtype signature | Rule signature | Behavior |
| --- | --- | --- | --- |
| `len(xs)` | `T[] -> Integer` | `L -> index` | Return the number of elements in the outer list. |
| `range(n)` | `Integer -> Integer[]` | `index -> L` | Construct `[0, 1, ..., n - 1]`. `range(0)` is empty. |
| `concat(a, b, ...)` | Two or more `T[] -> T[]` | Two or more `L -> L` | Join lists in argument order. Nested elements retain their boundaries. |
| `gather(xs, positions)` | `(T[], Integer[]) -> T[]` | `(L, L) -> L` | Select entries in the supplied order. Repeated positions are allowed. |
| `exclude(xs, removed)` | `(T[], T[]) -> T[]` | `(L, L) -> L` | Remove every element whose value occurs in `removed`. Preserve the order and duplicates of remaining elements. |
| `slice(xs, start, end)` | `(T[], Integer, Integer) -> T[]` | `(L, index, index) -> L` | Copy the half-open interval `[start, end)`. |
| `replace(xs, position, value)` | `(T[], Integer, T) -> T[]` | `(L, index, index) -> L` | Return a copy with one existing element replaced. |
| `sum(xs)` | `Integer[] -> Integer` | `L -> index` | Add elements with checked arithmetic. The empty sum is `0`. |
| `product(xs)` | `Integer[] -> Integer` | `L -> index` | Multiply elements with checked arithmetic. The empty product is `1`. Any zero makes the product `0`. |
| `all(xs)` | `Bool[] -> Bool` | Unavailable | Return true when every element is true. `all([])` is true. |
| `any(xs)` | `Bool[] -> Bool` | Unavailable | Return true when at least one element is true. `any([])` is false. |
| `contains(xs, value)` | `(T[], T) -> Bool` | `(L, index) -> bool` | Test whether the value occurs in the list. |
| `is_disjoint(a, b)` | `(T[], T[]) -> Bool` | `(L, L) -> bool` | Test whether the lists have no values in common. |

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
| `is_valid_axis_list(axes, rank)` | `(Integer[], Integer) -> Bool` | `(L, index) -> bool` | Require unique axes, each in `[0, rank)`. Invalid axes or rank return false. |
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
        is_numeric(T);
        len(Dims) <= 4;
        product(Dims) <= 4096;
    }
}
```

Rules can also receive `index_list` values from host functions. Rule expressions
have no list literals, list indexing, or comprehensions. Use the list helpers
to work with captured dimension sequences and host-provided lists.

Required scalar descriptor fields, such as `@capture.axis` or `@capture.to`,
can supply builtin arguments directly. Access list, optional, or opaque
descriptor fields through a host function.

`derive` assigns attribute descriptors. A builtin can compute an argument for a
descriptor-producing host function:

```javascript
derive {
    @out = $infer_attrs(@source, concat(Dims, range(2)));
}
```

This fragment assumes `@source` and `Dims` were captured on the left, and that
the application implements `$infer_attrs` with the required descriptor result.
A scalar or list builtin result cannot be the whole descriptor assignment.

### Shape programs

Shape parameters are operand shape lists. Inside an operation with a required
`permutation: index[]` attribute, a transpose shape program can use:

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

In rules, an unprefixed call resolves to a bound abstract-rule `fn` parameter
first, then to a builtin. `$name(...)` explicitly selects an application host
function, even when the name matches a builtin. Shape and dtype programs allow
builtins but do not allow `$` host calls.

All occurrences of a call's generic element type must agree. For example,
`concat([f32], [i32])` is a valid dtype list, while `concat([f32], [1])` mixes
dtype and integer values and fails type checking. Optional or opaque attributes
cannot be inspected by metadata expressions; required `index`, `i64`, `bool`,
and `dtype` attributes, including lists, can be read through `attrs.field`.

Integer arithmetic and reductions detect overflow. Division by zero, invalid
indices or ranges, nonpositive rounded divisors, and incompatible broadcasting
also fail. A product containing zero returns zero without overflowing its
intermediate multiplication. A yielded shape must contain nonnegative
dimensions that fit `u64`.

- In a rule, a builtin failure rejects the rewrite candidate before insertion.
  The leading builtin-only conditions in `where` can also prune matches during
  search. The first condition containing a host call ends that search prefix;
  it and later conditions, along with `derive`, run during application.
- In a shape or dtype program, a failed assertion or evaluated builtin makes
  inference invalid. Missing required tensor metadata can leave inference
  unknown. Assertions remain active in optimized builds.
- `&&`, `||`, and metadata conditionals evaluate only the required branches.
  Function arguments are evaluated once, in source order.

For tensor metadata concepts and operation examples, see the
[shape reference](shape.md) and [dtype reference](dtype.md).
The [builtin catalog](../../src/core/builtins/catalog.cc) defines the supported
names, signatures, and contexts.
