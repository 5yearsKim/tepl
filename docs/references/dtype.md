# Tensor dtypes

A **dtype** is the type of each tensor element. It is independent of the
tensor's [shape](shape.md): `f32[2, 3]` and `i32[2, 3]` have the same dimensions
but different element types.

TEPL supports `bool`, signed integers `i8`, `i16`, `i32`, `i64`, unsigned integers
`u8`, `u16`, `u32`, `u64`, and floating-point types `f16`, `bf16`, `f32`, `f64`.
Dtype equality compares exact types: `f16 != bf16` even though both use 16 bits.

## Operation dtype programs

An operation's `dtype(...)` block checks operand element types and chooses the
output element type. Parameters receive dtype values in operand order, not
tensors or strings. Their names can differ from the operand names.

A comparison can require equal numeric inputs and produce Boolean elements:

```javascript
op equal(lhs: tensor, rhs: tensor) -> tensor {
    dtype(a, b) {
        assert a == b;
        assert is_numeric(a);
        yield bool;
    }
}
```

For two `f32` inputs, `a` and `b` are both `f32`, and the result dtype is `bool`.
An `f32` input paired with an `i32` input fails the equality assertion.
This example defines dtype inference only; a shape block can separately require
matching dimensions and compute the output shape.

`let` creates a local value, `assert` checks a Boolean condition, and one final
`yield` must return a single dtype. Expressions can use dtype constants, equality,
Boolean logic, conditionals, lists, comprehensions, and supported attribute
fields. Arithmetic and ordering on dtype values are rejected.
See [built-in functions](built-ins.md#dtype-predicates) for dtype predicates.

### Fixed and variadic operands

Each fixed operand supplies one dtype parameter. A trailing variadic operand
supplies a list of dtypes, using a trailing `...` parameter. For example,
`dtype(first, rest...)` receives one dtype and a possibly empty dtype list.
The parameter structure must match the operation signature.

A zero-operand operation can fix its result with:

```javascript
dtype() {
    yield f32;
}
```

See the [variadic dtype example](built-ins.md#dtype-programs) for checking a
nonempty list of matching operand dtypes.

## Explicit conversion

A `dtype` attribute can choose an operation's output element type:

```javascript
op convert(input: tensor) -> tensor {
    attrs {
        to: dtype;
    }
    dtype(src) {
        assert is_numeric(src);
        assert is_numeric(attrs.to);
        yield attrs.to;
    }
    shape(s) {
        yield s;
    }
}
```

With an `f32` input and `to = i32`, this operation produces `i32` elements with
the same shape. The host defines the conversion's value semantics, including
rounding and overflow. Dtype inference establishes the type relationship.

Conversion requires an explicit operation. A dtype annotation does not insert a
cast, and TEPL does not implicitly promote mismatched input types. Each dialect
defines which conversions and mixed-type operations it supports.

## Dtype constraints in rules

Use a concrete annotation such as `X: f32[...]` to require one element type,
or declare a dtype variable to match a shared type. For the conversion operation
above, assuming its dialect is imported as `d`:

```javascript
rule remove_identity_convert {
    dtype T;
    X: T[...]

    (d.convert[@c] X) => X

    where {
        is_numeric(T);
        @c.to == T;
    }
}
```

`T` binds the matched tensor's dtype, and `@c.to` reads the captured conversion's
target type. This dialect defines same-type numeric conversion as identity.
Repeated `T` in other tensor declarations requires equal dtypes. A dtype variable
is local to its rule and must be bound by a matched tensor before use.

## Inference and rewrite checks

Dtype programs use the same [inference outcomes](shape.md#expressions-and-inference)
as shape programs. Omitting a dtype block provides no generated dtype inference;
the host may supply it. Unknown metadata is not a dtype value.

With tensor analysis enabled, replacement validation requires the root's dtype
and shape to match the original. Internal nodes may change dtype through explicit
conversions. Equal output metadata alone does not prove a rewrite correct:
intermediate conversions can lose information.

For complete operation definitions, see the
[dtype example dialect](../../examples/sample/dialects/dtype.tepl) and its
[rewrite rules](../../examples/sample/rules/dtype.tepl).
