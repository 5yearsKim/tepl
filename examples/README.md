# Example project

- `dialects/tensor.tepl`: tensor operations and attribute schemas.
- `dialects/scalar.tepl`: a minimal second dialect with `add` and `negate`.
- `rules/`: all concrete rules and reusable abstract templates.
- `rules/lowering.tepl`: checked rank-zero TensorLang → Scalar lowering.

From the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples
bazel-bin/tepl generate examples --out my_app/src/generated
```

Generation emits only module contents. Add `pub mod generated;` to your
application and egg 0.11 to its Cargo dependencies. The output directory can
have any name and live at any depth in the application's module tree.
Rust output is formatted by default; use `--no-format` to skip rustfmt.

Generation recursively discovers `.tepl` files under both directories, resolves
imports relative to each file, and emits each dialect once. Nested rule files
keep their relative module path. Rules with the same name in different files
remain distinct. Use unique dialect names across the project; aliases select
names in TEPL source but do not create extra Rust dialect modules.

Single-file generation is also supported:
`bazel-bin/tepl generate examples/rules/simple.tepl --out generated/simple`.

The checked-in lab IR is generated from this project:

```sh
./tools/regenerate_lab.sh
./tools/regenerate_lab.sh --check
```

Edit dialects and rules here, regenerate, then run
`cargo test --manifest-path labs/rust-egg/Cargo.toml` to exercise their generated
code with the handwritten hosts under `labs/rust-egg/src/host/`.
