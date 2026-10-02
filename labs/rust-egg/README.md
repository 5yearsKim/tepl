# Tensor IR and rules with egg

The library contains a tensor IR and a reusable rewrite adapter. The
`ir::patterns` module separates pattern data, matching, metadata lookup, and
application. LHS `TensorPattern::bind` captures the matched e-class while
checking its nested operation pattern; later `TensorPattern::Var` and
`TensorExpr::Var` references reuse that value.

`src/ir/rules/` contains one Rust module per TEPL source file:
`lora.rs`, `simple.rs`, `basic.rs`, `binders.rs`, and `inherited.rs`.
`basic.rs` contains the consolidated dtype and literal examples;
`binders.rs` contains both rules declared in `examples/binders.tepl`. Each rule lives in a `rule_<name>` module
with a `build_rewrite` constructor and, when needed, a host `Functions` trait.
`ir::rules` re-exports the basic, binder, LoRA, and simple rule modules directly.
`ir::rules::basic` also exposes the consolidated basic rules; inherited rules
use their own namespace to avoid name collisions. Callers supply host implementations and tensor metadata:

```rust
use rust_egg::ir::rules::{rule_commute_add, rule_shared_expression};

let rewrite = rule_commute_add::build_rewrite::<()>().unwrap();
let pattern = rule_shared_expression::pattern();
```

These rule modules are manually
maintained reference output for future generation. Their fixtures and behavior
checks live in `tests/lora.rs`, `tests/simple.rs`, and `tests/binders.rs`.

`TensorLang::literal("1.0", DType::F32)` creates a zero-operand numeric node;
`TensorPattern::literal` and `TensorExpr::literal` match and construct those
nodes. Accepted spellings are signed or unsigned integers and decimals with
digits on both sides of the point. Dtype and exact spelling define identity:
`1:i32`, `1:f32`, `1.0:f32`, and `1.00:f32` are distinct. Integer and boolean
ranges are checked without machine floating conversion. Floating spelling and
signed zero are preserved; hosts define float rounding and representability.
Graph literals denote rank-zero tensors with explicit dtype. Hosts supply
operation semantics, metadata, and broadcasting rules.
`tests/literals.rs` exercises the reference rules from
[`examples/basic.tepl`](../../examples/basic.tepl), including matching,
RHS insertion, and rejection of invalid spellings.

`src/ir/rules/inherited.rs` is the reference output for compile-time expansion
of `examples/inherited.tepl` using the templates in `examples/abstract.tepl`.
Its seven concrete rules contain specialized operations and host function names;
the vector instances retain both inherited checks and additional constraints.
They are available under `ir::rules::inherited` to avoid colliding with the
`commute_add` rule from `simple.tepl`:

```rust
use rust_egg::ir::rules::inherited::rule_commute_mul;

let rewrite = rule_commute_mul::build_rewrite::<()>().unwrap();
```

Behavior checks live in `tests/inherited.rs`. These files describe the expected
expanded output. The C++ core analyzer supports template expansion into checked
IR; Rust code generation remains future work.

The operation signatures and attribute schemas are now declared in
[`examples/dialects/tensor.tepl`](../../examples/dialects/tensor.tepl), which
the TEPL LoRA example imports. The Rust IR is still maintained manually.
`src/ir/dialects/tensor_lang.rs` contains the TensorLang operation definitions,
attributes, node type, and `egg::Language` implementation as one reference
output for the future generator. `src/ir/mod.rs` re-exports the public API; its
tests live in `tests/ir.rs`. TEPL `string` fields use Rust `String`, and
`index` fields use `usize`.

The TEPL compiler checks host signatures and rules through `core::Program`.
Rust code generation remains future work. Inspect the checked LoRA IR and run
the reference runtime tests with:

```sh
bazel build //:tepl
bazel-bin/tepl check examples/lora.tepl
cargo test --manifest-path labs/rust-egg/Cargo.toml
```

`tests/support/generated_host.rs` preserves a standalone interface fixture from
the retired AST-based generator, compiled by `tests/lora.rs`. The reference
`rules/lora.rs` declares the same methods directly so its callers can implement
`rules::rule_lora::Functions`. Each rule is exposed as a module containing its
own `Functions` trait, `pattern`, and `build_rewrite` constructor, so rules can
be used independently even when several share a source module.
Functions called in `where` return `Option<bool>`. Functions called in
`derive`, including `infer_dot`, return `Option<OpAttrs>`: only the descriptor
used to construct the operation. `None` rejects the match.

RHS `TensorExpr` is a tree of operations, literals, and references to LHS
captures. Bindings are supported only in LHS `TensorPattern`; RHS `let` is
rejected by the TEPL grammar. Both `where` and `derive` use the LHS environment.
Descriptor derivation has no RHS binding environment or dependency scheduler.
`OutputInference` separately verifies intermediate output shape and dtype before
insertion, using resolved descriptors and child metadata.

LoRA constructs its descriptors in source order with `infer_dot(X, W, outer)`,
`infer_dot(X, A, outer)`, and `infer_lora_out(X, A, B, outer, inner)`. The last
host function chooses the final dot descriptor directly from matched inputs.
Its result contains attributes only. Construct the reference rule with:

```rust
let rewrite = rule_lora::build_checked_rewrite(metadata, inference, functions)?;
```

All host checks and descriptor derivations complete first. The checked runtime
validates the RHS structure, infers every output, and requires final shape/dtype
equality with the matched root before recursively inserting nodes. Missing
metadata, unsupported dtype combinations, and invalid descriptors reject the
match without leaving partial RHS nodes. Static RHS arity and exact-attribute
errors are diagnosed when constructing the rewrite. E-class analysis should
share the same operation semantics. The saturation example demonstrates this.

`TensorInfo` contains `shape: Vec<usize>` and `dtype: DType`, with no unknown or
default dtype. Metadata lookup returns `None` unless all e-class alternatives
have compatible tensor descriptions. `TensorBindings` rejects conflicting
symbol/constant registrations while keeping existing types unchanged.
The legacy `tensor_rewrite` and `rule_lora::build_rewrite` paths rely on their
callers to prove complete replacement validity and output compatibility.

The integration tests use manually written rule modules and test host
implementations. Lowering full TEPL rules into those Rust patterns and
callbacks is a separate compiler stage. Core infers host argument and result
types from captures, declared dimensions, descriptors, literals, and expression
contexts, including nested calls. Both `where` and `derive` use the LHS
environment. `scalar` captures are rank-zero tensors passed as `&TensorInfo` in
the reference runtime; numbers in host expressions remain scalar host values.

[`examples/basic.tepl`](../../examples/basic.tepl) and `rules/basic.rs` cover
f32 vector constraints, shape-only declarations with a `same_dtype` host
predicate, typed scalar tensors, and integer/float literals. All five reference
rules use `tensor_rewrite_checked`. The overlapping floating-literal examples
are represented by one `commute_float_literal` rule. Signed-literal spellings
remain covered by tests without a separate example rule.
`tests/dtypes.rs` compiles the preserved basic host-interface fixture in
`tests/support/generated_basic_host.rs` and verifies acceptance, rejection,
output compatibility, literal identity, and absence of partial RHS insertion.
The fixture remains checked in until code generation from core is implemented.

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
