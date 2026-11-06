# Transpose simplification

Compose two transposes and remove the resulting identity transpose. This project
is shared by the [first rewrite tutorial](../../../docs/getting-started/first-rewrite.md)
and the [Rust integration tutorial](../../../docs/runtime/egraphs/rust/run.md).

```text
transpose/
├── tepl.toml
├── dialects/tensor.tepl
├── rules/transpose.tepl
└── graphs/transpose.tepl
```

## Run

From the repository root:

```sh
bazel-bin/tepl check examples/projects/transpose
bazel-bin/tepl rewrite examples/projects/transpose --trace -o /tmp/result.tepl
bazel-bin/tepl check /tmp/result.tepl
```

The result yields the original input `X`. The compiler infers the composed
permutation and verifies both rules under the default strict arithmetic policy.

Compare [explicit and inferred attributes](../../advanced/axis_inference/README.md),
or start with [double negation](../../basic/double_negation.tepl) for basic syntax.
