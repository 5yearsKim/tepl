# Automatic axis attributes

A graph rewrite often changes an operation's axis attributes. TEPL can construct
eligible replacement attributes from declared semantics, so the rule can focus
on the transformation.

## Compose transposes explicitly

With the [tutorial dialect](../../examples/projects/transpose/dialects/tensor.tepl)
imported as `t`, you can write:

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

`gather(p, q)` selects `p[q[i]]` for each position `i`. Applying `[1, 0]` twice
therefore produces `[0, 1]`.

## Let TEPL construct it

The same transformation can omit the replacement attributes:

```tepl
rule compose_transposes {
    (t.transpose (t.transpose X)) => (t.transpose X)
}
```

TEPL uses the dialect's axis and value contracts to construct the permutation
and verify the replacement. This happens when checking or compiling the rule.
The resulting metadata expression runs on each match; the runtime does not
repeat proof search.

```sh
bazel-bin/tepl check examples/projects/transpose --rule compose_transposes
```

The report shows the inferred constructor as well as the proof result.
The same completed rule can run in the TEPL engine, egg, or egg-c.

## Know what can be inferred

Only required non-optional `axis` and `axis[]` fields are eligible. Supply
extents, coordinates, and general fields yourself unless they have defaults or
are optional. Constructors can combine supplied fields and explicit `auto`:

```tepl
result.attrs = { dimensions: auto, new_extents: source.attrs.new_extents }
```

This illustrates a broadcast descriptor with an inferred axis mapping and
explicit new dimension sizes. Inference preserves supplied fields. A missing
proof is a diagnostic, not permission to guess an attribute.

See [attribute construction](../references/rule-expressions.md#infer-or-construct-replacement-attributes)
for constructor syntax and [metadata reasoning](../references/metadata-reasoning.md)
for the supported reasoning.
