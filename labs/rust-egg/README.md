# Generated tensor IR and rules with egg

`src/ir/` is generated entirely from the TEPL project under `examples/`.
`src/host/` contains handwritten tensor semantics, input bindings, shape analysis,
node helpers, and LoRA host functions. Keep host changes there; change dialects
and rules in their TEPL source, then regenerate.

From the repository root:

```sh
./tools/regenerate_lab.sh
./tools/regenerate_lab.sh --check
cargo test --manifest-path labs/rust-egg/Cargo.toml
```

The script builds the compiler, generates module contents with
`--out labs/rust-egg/src/ir`, and leaves Cargo configuration and host code
in place. `.tepl-generated-files` tracks generated paths so deleting or renaming
sources also removes obsolete output. `--check` compares without writing and
fails if output is stale. An existing compiler can be supplied through
`TEPL_COMPILER=/path/to/tepl`.
Rust output is formatted by default; the script requires rustfmt on `PATH`.

`ir` is this lab's chosen module name. Generated code uses relative imports;
other applications can choose `generated`, `any_name`, or a nested module path.
The compiler emits no Cargo configuration or `src/` wrapper.

The generated structure is:

```text
src/ir/
  dialects/{mod.rs, tensor_lang.rs, scalar.rs}
  op_node.rs
  types.rs
  pattern/{mod.rs, pattern.rs, matcher.rs, rewrite.rs, context.rs, shape.rs}
  rules/{mod.rs, basic.rs, binders.rs, inherited.rs, lora.rs, simple.rs,
         scalar.rs, lowering.rs}
```

Each dialect owns short `Op` and `OpAttrs` enums. `op_node.rs` combines them
into one node language for egg, keeping identically named operations distinct:

```rust
use rust_egg::ir::{OpNode, dialects::{scalar, tensor_lang}};
let node = OpNode::new(scalar::Op::Add, scalar::OpAttrs::None, vec![x, y])?;
```

The typed constructor pairs operations with attributes from their own dialect.
Arity and operation-specific schemas are checked at runtime. Rule files retain
separate modules and qualified rewrite names. Imported abstract templates expand
in core; their concrete instances belong to the instance file's module.

Each rule exposes `pattern()`, `expression()`, a `Functions` trait, and
`build_rewrite(metadata, inference, functions)`. For example:

```rust
use rust_egg::ir::rules::lora::rule_lora;
let rewrite = rule_lora::build_rewrite(metadata, inference, functions)?;
```

Hosts supply `TensorMetadata`, `OutputInference`, and implementations of the
rule's referenced host functions. Pass `()` for functions when a rule has no
host calls. Generated attribute fields and host trait methods preserve their
TEPL names, escaping Rust keywords with `r#`, such as `r#type`. Names that Rust
cannot escape are diagnosed during generation. Shapes and index attributes use
`u64`; Rust collection positions use
`usize`. `TensorInfo` has a concrete shape and dtype, and unavailable or
incompatible e-class metadata returns `None`.

The runtime matches structure and attribute witnesses, checks declared types,
runs `where` checks, and derives descriptors in source order from LHS values.
It validates and infers every RHS output, requiring final shape and dtype to
match the root before inserting any nodes. Missing metadata, failed host calls,
invalid descriptors, and incompatible outputs reject the rewrite. Host legality
functions remain responsible for numerical equivalence.

Bare graph literals use the shared `Literal` operation. An omitted dtype
matches any dtype; RHS inference resolves the concrete dtype before insertion.
Literal spelling is exact, including decimal formatting and signed zero.
Explicit dtype annotations are preserved. See
[the runtime contract](../../runtime/rust/README.md) and
[the generator architecture](../../src/codegen/README.md) for details.

The tests exercise the generated modules directly: structural matching,
attribute witnesses, binders, literals, dtype and shape restrictions, abstract
instances, cross-dialect lowering, checked replacement, and LoRA saturation.
Structural-only fixtures provide explicit metadata and inference; semantic
fixtures use the same host inference and analysis as the example.

## Run the LoRA saturation example

```sh
cargo run --manifest-path labs/rust-egg/Cargo.toml --example lora_saturation
cargo test --manifest-path labs/rust-egg/Cargo.toml --test lora_saturation
```

The example starts from `X @ (W + A @ B)` with shapes `X=[2,4,64]`,
`W=[2,64,32]`, `A=[2,64,4]`, and `B=[2,4,32]`. It runs LoRA and add
commutativity until saturation, then extracts the expression with the lowest
estimated arithmetic cost. The expected form is `X @ W + (X @ A) @ B`.
With these shapes, the estimate falls from 69,632 to 39,168 operations.
The example evaluates both expressions with deterministic integer tensors and
checks equal results. Floating-point reassociation requires its own numerical
policy.

The command prints e-classes and per-iteration rule applications and writes
`target/lora/before.dot` and `target/lora/after.dot` relative to this crate.
With Graphviz installed:

```sh
dot -Tsvg labs/rust-egg/target/lora/after.dot -o labs/rust-egg/target/lora/after.svg
```

The saturation tests also cover reversed addition, invalid shapes or contraction
axes, and rejected reassociation. Arithmetic cost here is an experiment rather
than a runtime benchmark; caching merged weights changes the relative cost.
