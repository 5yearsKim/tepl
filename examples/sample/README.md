# Sample project

This project demonstrates typed operations, shape/dtype contracts, pure rule
expressions, descriptor construction, and concrete graphs. `tepl.toml` marks the
project root; bare imports such as `"dialects/tensor.tepl"` resolve from here.

| Location | Demonstrates |
| --- | --- |
| `dialects/tensor.tepl` | Tensor operation subset, schemas, shape/dtype contracts |
| `dialects/scalar.tepl` | A second dialect for rank-zero lowering |
| `dialects/dtype.tepl` | Conversion, comparison, selection, fixed result dtypes |
| `rules/attributes.tepl` | Captured attributes, locals, comprehensions, descriptor updates |
| `rules/dtype.tepl`, `rules/lowering.tepl` | Dtype bindings and tensor-to-scalar lowering |
| `rules/lora.tepl` | Explicit axis guards and descriptor reuse |
| [graphs/](graphs/README.md) | Inputs, sharing, attributes, literals, named results |

```sh
bazel build //:tepl
bazel-bin/tepl check examples/sample
bazel-bin/tepl generate examples/sample --out /tmp/rust-app/src/generated
bazel-bin/tepl generate examples/sample --target cpp \
  --cpp-namespace sample_ir --out /tmp/cpp-app/generated
```

Both backends execute completed locals, lists, indexing, conditionals,
comprehensions, and typed descriptor constructors. Add their separate
[Rust](../../runtime/rust/README.md) or [C++](../../runtime/cpp/README.md) runtime.
Single-file generation and nested source module paths are also supported.

For syntax and contracts, use the public
[shape](../../docs/references/shape.md), [dtype](../../docs/references/dtype.md),
[builtin](../../docs/references/built-ins.md), and
[attribute-construction](../../docs/advanced/attributes_and_conditions.md)
references. Rank-zero tensors use `[]`; shape/dtype checks do not independently
justify floating-point reassociation or distribution.

The dialect is an example operation subset, not a complete StableHLO importer.
Use [semantic examples](../semantic/README.md) when working with value proofs or
automatic axis attributes. Shared fixture validation is described in
[tests/codegen](../../tests/codegen/README.md).
