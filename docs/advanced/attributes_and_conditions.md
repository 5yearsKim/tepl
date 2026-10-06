# Attributes, conditions, and derivations

Operation **attributes** describe configuration such as a transpose permutation
or contraction axes. They belong to the operation, separately from its operands
and tensor metadata. A **descriptor** holds the operation's complete attributes.

The examples below assume the [tensor dialect](../../examples/sample/dialects/tensor.tepl)
is imported as `t`.

## Capture and reuse attributes

Use `@name` to capture a descriptor on the left and reuse it on the right:

```javascript
rule transpose_negate {
    X: f32[Dims...]

    (t.transpose[@perm] (t.negate X))
        => (t.negate (t.transpose[@perm] X))
}
```

`@perm` captures all transpose attributes, including its permutation list.
The replacement transpose receives that same descriptor. It does not guess a
permutation from the input shape.

Repeated captures of the same descriptor name require matching attributes.
A replacement operation must accept the captured descriptor's schema.

## Check conditions and derive new attributes

`where` decides whether a candidate is allowed. `derive` constructs descriptors
needed by the replacement:

```javascript
rule compose_transposes {
    X: f32[Dims...]

    (t.transpose[@outer] (t.transpose[@inner] X))
        => (t.transpose[@combined] X)

    where {
        len(Dims) <= 4;
        $supports_composition(@outer, @inner);
    }

    derive {
        @combined = $compose_transposes(@outer, @inner);
    }
}
```

This rule applies only to inputs of rank at most four that the host accepts.
`$supports_composition` checks the host's policy. `$compose_transposes` returns
a transpose descriptor with the composed permutation. For inner permutation
`p` and outer permutation `q`, that permutation is `[p[q[i]] for each axis i]`.
These `$` functions are implemented by the application.

- Every `where` condition must be true. A failed host call also rejects the
  candidate; in Rust, callbacks report failure with `None`.
- A `derive` assignment must produce a descriptor. A scalar or list result
  alone is insufficient, although builtins can compute host-function arguments.
- Required scalar fields can be read directly, such as `@capture.axis`.
  List, optional, and opaque fields require host access in rule expressions.

Derived descriptors are validated before insertion. With tensor analysis enabled,
replacement metadata is checked too. Shape and dtype agreement does not prove
numerical equivalence; the dialect and host must justify the transformation.

See [binding and early pruning](binding_and_early_pruning.md) for evaluation
timing, [built-in functions](../references/built-ins.md) for expression support,
and the [LoRA rule](../../examples/sample/rules/lora.tepl) for a larger derivation.
