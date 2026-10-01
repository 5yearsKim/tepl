# Tensor IR and rules with egg

The library contains a tensor IR and a reusable rewrite adapter. The
`ir::patterns` module separates pattern data, matching, metadata lookup, and
application. LHS `TensorPattern::bind` captures the matched e-class while
checking its nested operation pattern; later `TensorPattern::Var` and
`TensorExpr::Var` references reuse that value.

`src/ir/rules/` contains one Rust module per TEPL source file:
`lora.rs`, `simple.rs`, and `binders.rs`. The last module contains both rules
declared in `examples/binders.tepl`. Each module constructs the corresponding
rewrite through a `rule_<name>` function and, when needed, declares the host
functions it calls. Callers supply host implementations and tensor metadata.
These rule modules are manually
maintained reference output for future generation. Their fixtures and behavior
checks live in `tests/lora.rs`, `tests/simple.rs`, and `tests/binders.rs`.

The operation signatures and attribute schemas are now declared in
[`examples/dialects/tensor.tepl`](../../examples/dialects/tensor.tepl), which
the TEPL LoRA example imports. The Rust IR is still maintained manually.
`src/ir/dialects/tensor_lang.rs` contains the TensorLang operation definitions,
attributes, node type, and `egg::Language` implementation as one reference
output for the future generator. `src/ir/mod.rs` re-exports the public API; its
tests live in `tests/ir.rs`. TEPL `string` fields use Rust `String`, and
`index` fields use `usize`.

The TEPL compiler currently discovers function calls in `where` and `derive`
and emits a standalone Rust host interface. For `examples/lora.tepl`:

```sh
bazel build //:tepl
bazel-bin/tepl host-template examples/lora.tepl > labs/rust-egg/tests/support/generated_host.rs
bazel-bin/tepl host-template examples/lora.tepl --impl > my_functions.rs
cargo test --manifest-path labs/rust-egg/Cargo.toml
```

`tests/support/generated_host.rs` records an example of that standalone
interface. The reference `rules/lora.rs` declares the same methods directly
so its callers can implement `rules::lora::HostFunctions`. Once TEPL generates
whole rule modules, the interface and rewrite will come from the same source.
The `--impl` output is a starting template; fill in its `todo!()` bodies once
and keep that implementation when regenerating the interface.
Functions called in `where` return `Option<bool>`. Functions called in
`derive` return `Option<InferredTensor>`, which holds attributes and the
output's `TensorInfo`. `None` rejects that match. The output description lets
a later derive call use a named RHS intermediate before insertion into egg.

The integration tests use manually written rule modules and test host
implementations. Lowering full TEPL rules into those Rust patterns and
callbacks is a separate compiler stage. The current template generator infers
arguments from named tensors, dimensions, attributes, binders, and literals.
It diagnoses calls whose argument types cannot yet be inferred, including
untyped scalar declarations and nested function arguments.
