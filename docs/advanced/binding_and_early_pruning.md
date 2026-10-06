# Binding and early pruning

A rewrite starts by matching its left-hand pattern. Each match binds tensor
captures, operation descriptors, and any declared dimension or dtype variables.
**Early pruning** rejects a search branch as soon as a required check fails.
This can avoid exploring the rest of a large pattern.

## Bindings come from matched values

```javascript
dtype T;
X: T[Batch..., M, K]
Y: T[Batch..., K, N]
```

Inside a rule, these declarations bind `T`, dimension sizes, and the `Batch`
sequence from matched tensor metadata. Repeated names require equal values.
For `X: f32[2, 3]`, `Batch` is empty, `M = 2`, and `K = 3`; a compatible `Y`
must also be `f32`, have no batch dimensions, and start with dimension `3`.

Declarations are checked when their tensor capture is bound. A mismatch or
unavailable required metadata rejects that branch. Bindings are local to each
branch, so a failed candidate does not affect other candidates.

## Which conditions run during search?

Assuming the tensor dialect is imported as `t`:

```javascript
rule cancel_small_negate {
    dtype T;
    X: T[Dims...]

    (t.negate (t.negate X)) => X

    where {
        is_numeric(T);
        len(Dims) <= 4;
        $allows_cancellation(X);
        product(Dims) <= 4096;
    }
}
```

| Check | When it runs |
| --- | --- |
| `X: T[Dims...]` | During binding, and rechecked during application. |
| `is_numeric(T)` and `len(Dims) <= 4` | During search once their bindings are ready, and rechecked during application. |
| `$allows_cancellation(X)` | During application. |
| `product(Dims) <= 4096` | During application because it follows a host condition. |

The leading conditions containing only builtins and operators can prune search.
They run in source order: if the next condition needs an unbound value, it waits,
and later conditions wait behind it.

The first condition containing a host call ends this early prefix. It and every
following condition run during application. A nested host call still establishes
this boundary, even if short-circuit evaluation skips calling it.

In this example, move `product(Dims) <= 4096` before the host condition if you
want it to prune search too. The rank and element-count limits are application
policy; they do not change the tensors or establish mathematical equivalence.

## Search and application are separate

Finding a structural match does not guarantee insertion. Application rechecks
the declarations and early conditions against current bindings, evaluates the
remaining conditions, and computes `derive` descriptors. With tensor analysis
enabled, it also validates replacement operations and the root's shape and dtype
before inserting nodes.

Keep inexpensive pure guards early when they express your intended policy.
Metadata answers must remain stable during a matching traversal. Host calls
remain fallible, and a rejected candidate leaves the replacement uninserted.

See [attributes and conditions](attributes_and_conditions.md) for descriptor
examples and [shape](../references/shape.md) and [dtype](../references/dtype.md)
for tensor constraints.
