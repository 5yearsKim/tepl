# Example project

- `dialects/tensor.tepl`: tensor operations and attribute schemas.
- `dialects/scalar.tepl`: a minimal second dialect with `add` and `negate`.
- `rules/`: all concrete rules and reusable abstract templates.
- `rules/lowering.tepl`: checked rank-zero TensorLang → Scalar lowering.

From the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples
bazel-bin/tepl generate examples --out generated/examples
cargo check --manifest-path generated/examples/Cargo.toml
```

Generation recursively discovers `.tepl` files under both directories, resolves
imports relative to each file, and emits each dialect once. Nested rule files
keep their relative module path. Rules with the same name in different files
remain distinct. Use unique dialect names across the project; aliases select
names in TEPL source but do not create extra Rust dialect modules.

Single-file generation is also supported:
`bazel-bin/tepl generate examples/rules/simple.tepl --out generated/simple`.
