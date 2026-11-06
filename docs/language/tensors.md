# Shapes and dtypes

Tensor declarations restrict which values a rule can match:

```tepl
X: f32[M, K]
Y: f32[K, N]
```

Both captures must be rank-two `f32` tensors. Reusing `K` requires those dimensions
to have equal sizes. A declaration does not reshape a tensor or insert a cast.

## Match different ranks

| Constraint | Matches |
| --- | --- |
| `X: f32[M, N]` | A rank-two `f32` tensor |
| `X: [Batch..., M, K]` | Any number of batch dimensions followed by `M, K` |
| `X: f32[...]` | Any rank with dtype `f32` |
| `X: [_, N]` | Rank two, with any first dimension |
| `X: f32[]` | An `f32` scalar |

Omit the dtype to leave it unrestricted. A shape can contain at most one
dimension sequence. Missing required metadata can prevent a constrained match.

## Define operation contracts

An operation can check operands and infer its result:

```tepl
op add(lhs: tensor, rhs: tensor) -> tensor {
    dtype(a, b) {
        assert a == b
        assert is_float(a)
        yield a
    }
    shape(a, b) {
        assert a == b
        yield a
    }
}
```

Inside a dialect, this defines addition with identical shapes and floating-point
dtypes, without broadcasting. The host implements the actual operation.
The `assert` statements reject incompatible inputs; `yield` supplies the
resulting dtype or shape.

Next: [attributes and conditions](attributes-and-conditions.md).

Reference: [shape syntax](../references/shape.md) and [dtype constraints](../references/dtype.md).
