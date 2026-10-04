# Operation dtype policies

`dtype:` is optional. It describes output dtype inference, while a rule's
`X: f32[...]` annotation restricts what that rule can match. Shape definitions
compute shapes independently.

```tepl
dialect Example {
    op add(lhs: tensor, rhs: tensor) -> tensor {
        dtype: same_numeric;
        shape(l, r) {
            assert l == r;
            yield l;
        }
    }

    op sqrt(input: tensor) -> tensor { dtype: same_float; }
    op copy(input: tensor) -> tensor { dtype: same; }
    op concatenate(inputs: tensor...) -> tensor { dtype: same; }
    op equal(lhs: tensor, rhs: tensor) -> tensor { dtype: bool; }
    op scalar_zero() -> tensor {
        dtype: f32;
        shape() { yield []; }
    }

    // No dtype policy: no inferred output dtype from this declaration.
    op custom(input: tensor) -> tensor;
}
```

`same` requires equal operand dtypes and preserves that dtype. `same_numeric`
also excludes `bool`. `same_float` accepts only `f16`, `bf16`, `f32`, and `f64`.
These policies consider all actual operands, including variadic tails, and do
not promote mixed types. No operands or insufficient information means
`Unknown`; a known mismatch or disallowed dtype means invalid input.

A concrete dtype fixes only the output dtype. For example, `equal` above has a
Boolean result, but its declaration does not check whether its input dtypes
are compatible. Supported concrete dtypes are `bool`, `i8`, `i16`, `i32`, `i64`,
`u8`, `u16`, `u32`, `u64`, `f16`, `bf16`, `f32`, and `f64`.

At most one `dtype:` property is allowed, alongside an optional alias and
inline/shared attributes, in any order before the optional final shape block.
Omitting it is valid even when a shape block exists. Unknown policy names are
core errors with source locations. `dtype` remains an ordinary identifier
outside the property position.

The frontend and core retain and validate these declarations, and Rust codegen
emits their inference dispatch. Missing policies return `Unknown`. The example
tensor dialect annotates operations covered by these simple policies. Constants,
dot, and convolution leave `dtype:` unspecified; default tensor inference does
not guess their payload- or attribute-dependent behavior. An application can
provide a custom analysis and use `build_rewrite_with(...)` for explicit policies.
