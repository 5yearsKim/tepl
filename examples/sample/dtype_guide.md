# Operation dtype programs

Dtype programs, dtype-valued attributes, and rule dtype bindings are supported
by the parser, checker, and Rust and C++ backends.

An optional `dtype(...) { ... }` block validates operand element types and computes
the result element type. It does not execute tensor operations, insert casts, or
establish numerical equivalence between two graphs. Shape programs remain separate.

## Parameters are dtype values

The operation receives tensors. Its dtype block receives their element types in
operand declaration order:

```tepl
op add(lhs: tensor, rhs: tensor) -> tensor {
    dtype(a, b) {
        assert a == b;
        assert is_numeric(a);
        yield a;
    }
    shape(l, r) {
        assert l == r;
        yield l;
    }
}
```

For two `f32` tensors, `a` and `b` are both the dtype value `f32`. They are not
tensors or strings. Parameter names are local and need not match operand names.
Each fixed operand binds one dtype parameter; a trailing variadic operand binds
one `dtype[]` parameter. A zero-operand operation uses `dtype()`.

A block contains ordered `let` and `assert` statements followed by one final
`yield` of type `dtype`. Expressions support dtype constants, equality, Boolean
logic, conditionals, typed attribute access, and the existing list operations.
Each operation may declare at most one dtype block, alongside its optional
attribute schema and shape program. As in the examples, the shape block stays last;
attribute declarations are resolved for the entire operation.

Concrete dtype values initially remain `bool`, `i8`, `i16`, `i32`, `i64`, `u8`,
`u16`, `u32`, `u64`, `f16`, `bf16`, `f32`, and `f64`. Equality means exact element
type equality, not equal bit width or permission to convert.

## Builtins

| Function | Signature | Meaning for the initial dtype set |
| --- | --- | --- |
| `is_float` | `dtype -> bool` | `f16`, `bf16`, `f32`, or `f64` |
| `is_integer` | `dtype -> bool` | Signed or unsigned integer; excludes `bool` |
| `is_signed_integer` | `dtype -> bool` | `i8`, `i16`, `i32`, or `i64` |
| `is_unsigned_integer` | `dtype -> bool` | `u8`, `u16`, `u32`, or `u64` |
| `is_numeric` | `dtype -> bool` | Integer or floating-point; excludes `bool` |

These predicates are available in dtype blocks and rule `where` conditions.
There is no `dtype_of` builtin: block parameters and rule dtype bindings already
provide dtype values. Test Boolean types with `t == bool`, and particular formats
with expressions such as `t == f16 || t == bf16`.

List operations such as `len` and `all` also work in dtype programs. For example,
the tensor dialect's concatenation operation uses:

```tepl
dtype(inputs...) {
    assert len(inputs) > 0;
    assert all([t == inputs[0] for t in inputs]);
    yield inputs[0];
}
```

Assertions execute in source order. A failed assertion stops evaluation, so the
nonempty check protects the later indexing. Empty variadic input is invalid in
this example; another operation may define a result for an empty list explicitly.

## Common relationships

Preserve the sole operand's dtype:

```tepl
dtype(t) {
    yield t;
}
```

Restrict an operation to floating-point inputs:

```tepl
dtype(t) {
    assert is_float(t);
    yield t;
}
```

Compare equal numeric input types and produce a Boolean result:

```tepl
dtype(a, b) {
    assert a == b;
    assert is_numeric(a);
    yield bool;
}
```

For a select operation, constrain the condition separately from its values:

```tepl
dtype(condition, a, b) {
    assert condition == bool;
    assert a == b;
    yield a;
}
```

A zero-operand operation can fix its result with `dtype() { yield f32; }`.
Each operation declares its own dtype constraints and result expression.

## Explicit conversion and typed attributes

The attribute type `dtype` holds an element type directly:

```tepl
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

This example dialect permits conversions between its numeric types. The host
defines their value semantics, including rounding and overflow behavior, and
same-type conversion is an identity. The dtype block only establishes the input
and result type relationship. A dialect supporting fewer conversions needs an
additional dialect-specific validation policy.

Conversion is an explicit graph node. An annotation such as `X: f32[...]` only
restricts matching; it never converts `X`. Promotion rules are dialect-specific,
and no universal `promote`, `can_cast`, or lossless-conversion builtin is defined.
Specialized policies continue to use host inference and rule legality hooks.

## Rule dtype bindings

Declare a dtype variable explicitly and bind it through tensor declarations:

```tepl
rule commute_numeric {
    dtype T;
    X: T[N]
    Y: T[N]

    (add X Y) => (add Y X)

    where {
        is_numeric(T);
    }
}
```

Here `add` must be imported from a dialect whose semantics permit this rewrite.
Repeated `T` requires both captures to have the same dtype. Type variables are
local to the rule and must be bound by matched tensor declarations before use.
An omitted dtype remains unrestricted; concrete annotations remain supported.

Nonoptional scalar descriptor fields can be read in rule conditions. Optional,
list, and opaque descriptor fields require host access. For the numeric conversion
operation above, a same-type conversion can be removed:

```tepl
rule remove_identity_convert {
    dtype T;
    X: T[...]

    (convert[@c] X) => X

    where {
        is_numeric(T);
        @c.to == T;
    }
}
```

The complete declarations and rules are in [dialects/dtype.tepl](dialects/dtype.tepl)
and [rules/dtype.tepl](rules/dtype.tepl). Arbitrary nested conversions cannot be
collapsed merely because their endpoints have matching dtypes: intermediate
rounding may lose information. Such rewrites require explicit host legality checks.

## Inference and implementation boundaries

A dtype program returns a known result, invalid input, or unknown metadata.
A false assertion or an invalid evaluated operation rejects the input. Missing
metadata needed to evaluate the program produces `Unknown`; it is not a concrete
dtype value and cannot satisfy an equality check. Omitting the dtype block provides
no inference from that declaration; a host policy may supply it.

Shape and dtype programs describe separate metadata. Combined tensor validation
must reject known invalid metadata even if another component is unknown. Before
inserting a replacement, require its root shape and dtype to match the original;
internal nodes may have different dtypes connected by explicit conversions.

Accumulation precision, rounding, and supported algorithms remain operation
semantics or attributes. Quantized and custom element types will require a richer
dtype representation; their storage type alone does not describe their meaning.
Constants, dot, and convolution in the tensor dialect continue to rely on host
inference where their metadata depends on payloads, algorithms, or attributes.

In mixed-precision replacements, annotate graph literals explicitly when their
dtype differs from the root's dtype. The current runtime's root-based literal
default is not inference from the consuming operation.
