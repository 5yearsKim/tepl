# Generated TEPL rewrites with egg

This lab runs the transpose rules from `examples/rewrite/`. Generated code lives
in `src/ir/` and depends on the shared `tepl-runtime` crate. Application code
supplies graph inputs, selects rules, and extracts a result.

From the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl generate examples/rewrite --out labs/rust-egg/src/ir
cargo test --manifest-path labs/rust-egg/Cargo.toml --locked --all-targets
cargo run --manifest-path labs/rust-egg/Cargo.toml --example transpose
```

The example composes two transpose permutations, removes an identity transpose,
and extracts a smaller expression. Input shape and dtype are registered by the
generated graph constructor.

`src/ir/` is ignored by Git. Regenerate it after changing the TEPL program or
emitters. Reusable execution code belongs in `runtime/rust/`, and application
code belongs in this lab.

See the [runtime guide](../../runtime/rust/README.md) for node construction,
metadata facts, and checked runner integration. `tools/test_codegen.sh` tests
source and JSON generation, synthesized attributes, and the complete builtin
catalog against the executable evaluator.
