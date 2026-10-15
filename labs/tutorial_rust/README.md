# TEPL Rust tutorial

Both demos use generated rule functions and the shared Rust runtime. Conditions
such as `@dot.axis == 0` are evaluated directly by generated Rust.

From the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl generate labs/tutorial_rust/pattern_basic \
  --out labs/tutorial_rust/src/tepl_pattern_basic
bazel-bin/tepl generate labs/tutorial_rust/pattern_analyze \
  --out labs/tutorial_rust/src/tepl_pattern_analyze
cargo run --manifest-path labs/tutorial_rust/Cargo.toml --bin pattern_basic
cargo run --manifest-path labs/tutorial_rust/Cargo.toml --bin pattern_analyze
```

`pattern_basic` explores structural addition and dot rewrites, then extracts a
minimum-depth expression. Its operations have no metadata contracts, so their
output metadata remains unknown.

`pattern_analyze` defines shape and dtype contracts and declares input metadata.
It factors a shared matrix-product operand, checks output metadata preservation,
and extracts a smaller expression.

Generated modules are ignored by Git. Reusable runtime helpers live in
[`runtime/rust/`](../../runtime/rust/README.md).
