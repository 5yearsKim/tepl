# Semantic example project

This is the canonical example dialect for axis/value contracts, semantic
verification, explicit metadata constructors, and field-level `auto` synthesis.
`tepl.toml` marks the root; rule files share `dialects/tensor.tepl`.

| Rules | Demonstrates |
| --- | --- |
| [reductions.tepl](rules/reductions.tepl), [reverse.tepl](rules/reverse.tepl) | Permutation/reduction/reversal relationships |
| [basic.tepl](rules/basic.tepl), [algebraic.tepl](rules/algebraic.tepl) | Guarded contraction rewrites and reassociation |
| [multiply.tepl](rules/multiply.tepl), [scalars.tepl](rules/scalars.tepl) | Aligned scalar operations and arithmetic policies |
| [constructors.tepl](rules/constructors.tepl), [explicit.tepl](rules/explicit.tepl) | Typed descriptor construction and reuse |
| [broadcast.tepl](rules/broadcast.tepl) | Automatic axis mappings with supplied new extents |
| [convolution.tepl](rules/convolution.tepl), [slice.tepl](rules/slice.tepl) | Explicit calculated-read metadata and verification |

Run from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples/semantic/rules/algebraic.tepl \
  --semantic-policy algebraic --rule distribute
bazel-bin/tepl check examples/semantic/rules/slice.tepl --rule compose_slice
bazel-bin/tepl generate examples/semantic --semantic-policy algebraic \
  --out /tmp/app/src/generated
```

Use `--target cpp` for C++ headers. Both targets evaluate completed constructors
through their shared runtimes. Distribution/reassociation require an appropriate
arithmetic policy; metadata preservation alone does not establish value equality.

Only axis fields are automatically constructed. Coordinate, extent, and general
fields must be supplied, defaulted, or optional. Synthesis preserves supplied
fields and never invents hidden applicability guards. An unproved rule is
rejected without claiming that every attribute assignment is impossible.

For general explanations, use:

- [Public metadata reasoning](../../docs/references/metadata-reasoning.md) and
  [attributes/conditions](../../docs/advanced/attributes_and_conditions.md).
- [Semantic verification design](../../design/semantic-verification.md): contracts,
  normalization, acceptance, implementation ownership, and tests.
- [Attribute synthesis design](../../design/attribute-synthesis.md): strategies,
  ordered admission, suggestions, and the executable boundary.
- [Research](../../research/attribute-synthesis/README.md): mathematical arguments
  and their admitted domains, separately from production implementation coverage.

The example source files are the authority for current guards and constructors.
