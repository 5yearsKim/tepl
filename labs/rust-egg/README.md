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
so its callers can implement `rules::lora::lora::Functions`. Each rule is
exposed as a module containing its own `Functions` trait, `pattern`, and
`build_rewrite` constructor, so rules can be used independently even when
several share a source module.
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

## Run the LoRA saturation example

```sh
cargo run --manifest-path labs/rust-egg/Cargo.toml --example lora_saturation
cargo test --manifest-path labs/rust-egg/Cargo.toml --test lora_saturation
```

The example starts from `X @ (W + A @ B)` with shapes `X=[2,4,64]`,
`W=[2,64,32]`, `A=[2,64,4]`, and `B=[2,4,32]`. It runs the LoRA and add
commutativity rules until egg reports saturation, then extracts the expression
with the lowest estimated arithmetic cost. The expected LoRA form is
`X @ W + (X @ A) @ B`. With these shapes the estimate is 69,632 operations
for the original and 39,168 for the extracted form. The example also evaluates
both expressions with deterministic integer tensors and checks that the results
are equal. Reassociation is enabled here under exact integer arithmetic;
floating-point reassociation requires a separate numerical policy.

The command prints the e-classes and each iteration's rule applications. It
writes `target/lora/before.dot` and `target/lora/after.dot` relative to this
crate. If Graphviz is installed, render the final graph with:

```sh
dot -Tsvg labs/rust-egg/target/lora/after.dot -o labs/rust-egg/target/lora/after.svg
```

In the final graph, the root e-class contains both a `dot` node (the input)
and an `add` node (the LoRA form). The test also checks a reversed addition,
where commutativity exposes the LoRA match in a later iteration, and verifies
that invalid shapes or contraction axes and rejected reassociation do not add
the LoRA form. This is a small arithmetic-cost experiment, not a runtime
benchmark; caching the merged weights would change the relative cost.
