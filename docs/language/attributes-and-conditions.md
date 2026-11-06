# Attributes and conditions

Attributes configure an operation: a transpose needs a permutation, while a
slice needs starts, sizes, and strides. They are separate from a tensor's shape
and dtype. The examples below use a tensor dialect imported as `t`.

## Capture attributes

Name an operation occurrence to read its attributes:

```tepl
(node: t.transpose X)
```

`X` captures the operand. `node.attrs` captures the complete descriptor;
`node.attrs.permutation` reads one field. Each matched operation has its own
attributes, even when several operations use the same symbol.

## Restrict a match

The tutorial removes a transpose only when it is an identity:

```tepl
rule remove_identity {
    X: [Dims...]
    (node: t.transpose X) => X
    where node.attrs.permutation == range(len(Dims))
}
```

`where` evaluates a Boolean condition. False skips the match. Use a block when a
condition needs local calculations:

```tepl
where {
    let identity = range(len(Dims))
    yield node.attrs.permutation == identity
}
```

## Build replacement attributes

Use `derive` to construct a replacement descriptor:

```tepl
rule compose_transposes_explicit {
    (outer: t.transpose (inner: t.transpose X))
    => (result: t.transpose X)
    derive {
        result.attrs = {
            permutation: gather(inner.attrs.permutation, outer.attrs.permutation)
        }
    }
}
```

An inline `[source.attrs]` reuses a complete captured descriptor on a replacement
operation with the same schema. `{ ..source.attrs, field: value }` copies it and
overrides a field. `where` restricts applicability; `derive` computes metadata.
Neither block calls application host functions.

You now have the basic tools for writing rules: patterns, tensor constraints,
attribute captures, conditions, and explicit replacements.

Continue to Advanced with [semantic verification](semantics.md).
For detailed syntax, see [rule expressions and descriptors](../references/rule-expressions.md).
