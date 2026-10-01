# Tensor patterns with egg

The library contains a tensor IR and a reusable rewrite adapter. The
`tensor_pattern` module separates pattern data, matching, metadata lookup, and
application. `tests/lora_pattern.rs` shows a LoRA rule built from this API.
The operation signatures and attribute schemas are now declared in
[`examples/dialects/tensor.tepl`](../../examples/dialects/tensor.tepl), which
the TEPL LoRA example imports. The Rust IR is still maintained manually;
generating it from the dialect is a later compiler stage.

The TEPL compiler discovers function calls in `where` and `derive` and emits a
Rust host interface. For `examples/lora.tepl`:

```sh
bazel build //:tepl
bazel-bin/tepl host-template examples/lora.tepl > labs/rust-egg/tests/support/generated_host.rs
bazel-bin/tepl host-template examples/lora.tepl --impl > my_functions.rs
cargo test --manifest-path labs/rust-egg/Cargo.toml
```

Keep the generated interface and your `my_functions.rs` implementation in
separate files. The `--impl` output is a starting template; fill in its
`todo!()` bodies once and keep that file when regenerating the interface.
Functions called in `where` return `Option<bool>`. Functions called in
`derive` return `Option<InferredTensor>`, which holds attributes and the
output's `TensorInfo`. `None` rejects that match. The output description lets
a later derive call use a named RHS intermediate before insertion into egg.

The integration test uses the generated interface and a hand-written LoRA
pattern and callback. Lowering full TEPL rules into those Rust patterns and
callbacks is a separate compiler stage. The current template generator infers
arguments from named tensors, dimensions, attributes, binders, and literals.
It diagnoses calls whose argument types cannot yet be inferred, including
untyped scalar declarations and nested function arguments.
