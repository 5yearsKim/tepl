# What is TEPL?

TEPL is a declarative language for tensor graph rewrites. A rule describes
**what to match and what to build**, with tensor constraints and operation
attributes alongside the transformation.

For example, with a tensor dialect imported as `t`:

```tepl
rule compose_transposes {
    (t.transpose (t.transpose X)) => (t.transpose X)
}
```

The replacement needs a permutation that combines the two original transposes.
When the dialect declares axis and value semantics, TEPL can construct that
permutation and verify the transformation in its supported reasoning fragment.
You can also supply attributes explicitly.

## Three building blocks

| Declaration | Purpose |
| --- | --- |
| **Dialect** | Defines operations, attributes, and metadata or semantic contracts. |
| **Rule** | Describes a matched pattern, replacement, and applicability conditions. |
| **Graph** | Describes a concrete computation, including inputs and its output. |

A dialect gives operations their vocabulary and declared meaning. Rules transform
computations in that vocabulary. A graph supplies something to rewrite.

## What TEPL checks

TEPL checks names, types, operation signatures, and attribute construction.
Runtime metadata checks constrain matches and validate replacement shapes and
dtypes. Operations with semantic contracts also support compiler verification
of value and axis relationships under the selected arithmetic policy.

These are different guarantees: matching shapes alone does not establish
mathematical equivalence. See [semantic verification](../language/semantics.md).
The host's operation implementation must agree with its declared contracts.

## Choose how to run it

- **TEPL engine:** rewrite a concrete graph directly and export readable TEPL.
- **E-graph integration:** generate Rust for egg or C++ for egg-c, retain
  equivalent alternatives, and extract a result using your cost model.

The engine transforms graphs; it does not execute tensor kernels. E-graph hosts
choose their search limits and extraction costs.

Next: [install TEPL](install.md), then [run your first rewrite](first-rewrite.md).
