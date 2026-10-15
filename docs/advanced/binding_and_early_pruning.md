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
        is_numeric(T)
        && len(Dims) <= 4
        && product(Dims) <= 4096
    }
}
```

| Check | When it runs |
| --- | --- |
| `X: T[Dims...]` | During binding, and rechecked during application. |
| Final `where` expression | During search once all its required bindings are ready, and rechecked during application. |

The final Boolean expression can prune search once its required bindings are
ready. Within that expression, `&&` and `||` evaluate from left to right and
short-circuit. Inherited guards remain separate checks in base-to-child order.

All rule expressions are pure. Guards wait for their required bindings.
Rank and element-count limits are application policy; they do
not establish mathematical equivalence.

Local bindings and extended list expressions are checked by the frontend. Their
code generation and execution are deferred and produce explicit diagnostics.

## Search and application are separate

Finding a structural match does not guarantee insertion. Application rechecks
the declarations and early conditions against current bindings, evaluates the
remaining conditions, and computes `derive` descriptors. With tensor analysis
enabled, it also validates replacement operations and the root's shape and dtype
before inserting nodes.

Keep inexpensive pure guards early when they express your intended policy.
Metadata answers must remain stable during a matching traversal. Failed expression evaluation rejects the candidate before insertion.

See [attributes and conditions](attributes_and_conditions.md) for descriptor
examples and [shape](../references/shape.md) and [dtype](../references/dtype.md)
for tensor constraints.
