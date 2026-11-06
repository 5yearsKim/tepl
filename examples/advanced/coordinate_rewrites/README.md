# Coordinate rewrites

Compose two slices into one. The rule adds their starting offsets, retains the
outer sizes, and restricts both slices to unit strides. The dialect describes
calculated reads and their bounds.

## Run

```sh
bazel-bin/tepl check examples/advanced/coordinate_rewrites
bazel-bin/tepl rewrite examples/advanced/coordinate_rewrites --trace -o /tmp/slices.tepl
```

The result is one slice of `X` with:

```text
starts  = [2, 1]
sizes   = [2, 2]
strides = [1, 1]
```

The result shape is `[2, 2]`. The rule supplies coordinate fields explicitly;
automatic axis inference does not construct slice offsets.

Read [metadata and coordinate reasoning](../../../docs/references/metadata-reasoning.md).
