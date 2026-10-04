# Tensor IR reference and rules with egg

`src/ir/` is an executable reference for intended compiler output. It provides
`TensorAnalysis`, input bindings, shape and dtype inference, and checked rules.
`src/host/` contains application-specific node helpers and LoRA legality functions.
See [IR file ownership and contracts](src/ir/README.md) for copied versus emitted
code and supported policies.

## Run and validate

From the repository root:

```sh
cargo test --manifest-path labs/rust-egg/Cargo.toml --all-targets
cargo run --manifest-path labs/rust-egg/Cargo.toml --example lora_saturation
```

The compiler generates the complete `src/ir/` module: operation inference,
reusable analysis support, dialects, nodes, and default/explicit rule builders.
Regenerate and verify it with:

```sh
./tools/regenerate_lab.sh
./tools/regenerate_lab.sh --check
```

The output is a module directory for an existing Rust edition 2024 crate with
egg 0.11. It can use any enclosing module name, including a nested path. The
compiler creates no Cargo configuration. `.tepl-generated-files` tracks the
files owned by current generation, and rustfmt formats output by default.

## Using the provided analysis

Register input types before building the graph:

```rust
use egg::EGraph;
use rust_egg::ir::analysis::{TensorAnalysis, TensorBindingTable, TensorInfo};
use rust_egg::ir::{DType, OpNode};
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::rules::simple::rule_commute_add;

let mut inputs = TensorBindingTable::default();
for name in ["X", "Y"] {
    inputs.register_symbol(name, TensorInfo {
        shape: vec![32, 64],
        dtype: DType::F32,
    }).unwrap();
}
let mut graph = EGraph::new(TensorAnalysis::new(inputs));
let x = graph.add(OpNode::input("X"));
let y = graph.add(OpNode::input("Y"));
let sum = graph.add(OpNode::new(
    tensor_lang::Op::Add, tensor_lang::OpAttrs::None, vec![x, y],
).unwrap());
assert_eq!(graph[graph.find(sum)].data.info().unwrap().shape, vec![32, 64]);

let rewrite = rule_commute_add::build_rewrite(()).unwrap();
let runner = egg::Runner::<_, TensorAnalysis>::new(TensorAnalysis::default())
    .with_egraph(graph)
    .run(&[rewrite]);
```

One `TensorAnalysis` holds the input table for each graph. Its `make` reads input
or operand metadata and calls shared tensor inference; its `merge` combines
`TensorAnalysisData` and reports changes to egg. The application supplies input
metadata, constructs nodes, and selects rewrites. It writes no metadata adapter
or analysis implementation.

Facts expose shape and dtype only when every alternative is known and agrees.
Unknown alternatives preserve known evidence but block metadata access; later
conflicts are still detected. Invalid operands take precedence over unknown
ones. Facts accumulate rather than retract, so supplying missing input types
later requires a fresh graph. Invalid facts do not retain error messages.

## Operation inference

All 24 TEPL shape definitions in the example dialects have generated evaluators,
including variadic concatenate, reduce, general dot, convolution, and all-to-all.
Rust statements follow their TEPL blocks and use `shape_builtins`. Shapes stay
`Vec<u64>` at the interface; computations widen to checked `i128` and validate
final dimensions against `u64`. Missing blocks return `Unknown`; failed
assertions, bad signatures, invalid indices, and arithmetic errors return
`Invalid` in both debug and release builds.

```rust
use rust_egg::ir::analysis::{Inference, infer_shape, infer_dtype, infer_tensor};

// infer_shape(op, &[&[u64]], attrs) -> Inference<Vec<u64>>
// infer_dtype(op, &[DType], attrs)  -> Inference<DType>
// infer_tensor(op, &[TensorInfo], attrs) -> Inference<TensorInfo>
```

Dtype inference follows optional TEPL policies: `same`, `same_numeric`,
`same_float`, or a fixed concrete dtype. Missing definitions return `Unknown`;
common policies with zero operands also return `Unknown`. Graph literals read
explicit dtypes. Dot, convolution, and constants have no example dtype policy,
so default combined inference remains unknown for them. Constants also have no
shape definition; their payload does not implicitly supply metadata. All-gather
and reduce-scatter need process-grid metadata and have no shape block.

The LoRA demo explicitly supplies `host::LoraAnalysis` and
`infer_lora_tensor_output` for its dot policy. Opaque regions, mesh groups, and
constant byte encodings remain application responsibilities.

## Checked rewrites

Each rule exposes `pattern()`, `expression()`, a `Functions` trait, and the
default builder:

```rust
use rust_egg::ir::rules::simple::rule_commute_add;

let rewrite = rule_commute_add::build_rewrite(()).unwrap();
```

The default builder reads metadata from `TensorAnalysis` and validates RHS
operations with the same inference. Users pass only implementations of the
rule's explicit custom functions, or `()` when there are none.
`build_rewrite_with(metadata, inference, functions)` provides explicit hooks
for custom analyses and tests that inject missing or incompatible metadata.
For example, LoRA selects `LoraAnalysis`, `lora_tensor_info`, and
`infer_lora_tensor_output` explicitly through that builder.

The matching runtime finds structural and attribute witnesses, checks tensor
restrictions and `where` conditions, and derives descriptors in source order.
It infers every RHS operation and requires final shape/dtype equality with the
matched root before inserting nodes. Missing facts, failed host functions, bad
descriptors, and incompatible outputs reject application without partial RHS
insertion. Early shape pruning during structural search remains future work.
Host legality functions remain responsible for numerical equivalence.

Each dialect owns short `Op`/`OpAttrs` enums; `op_node.rs` combines them into one
egg language. Typed constructors pair operations with the proper attributes,
while dynamic constructors validate arity and schemas. Rule modules retain
source ownership, and inherited templates expand to concrete rules in core.

Bare literals use the shared `Literal` operation. Unannotated LHS literals match
any dtype; RHS inference resolves dtype before insertion. Spelling and explicit
dtype annotations are preserved. Input nodes use `OpNode::input(name)`. See
[the runtime contract](../../templates/rust/README.md) for literal and matching
behavior. The compiler's current API is documented separately in
[the generator architecture](../../src/codegen/README.md).

## LoRA saturation example

The example begins with `X @ (W + A @ B)` and input shapes `X=[2,4,64]`,
`W=[2,64,32]`, `A=[2,64,4]`, and `B=[2,4,32]`. LoRA and add commutativity reach
saturation, then extraction selects `X @ W + (X @ A) @ B`. Estimated arithmetic
cost falls from 69,632 to 39,168 operations. Deterministic integer evaluation
checks equal values; floating-point reassociation needs its own host policy.

The example prints e-classes and rule applications, and writes
`target/lora/before.dot` and `target/lora/after.dot` inside this crate. Graphviz
can render them:

```sh
dot -Tsvg labs/rust-egg/target/lora/after.dot -o labs/rust-egg/target/lora/after.svg
```

Tests cover reversed addition, rejected reassociation, incompatible shapes,
unsupported axes, shape/dtype inference, fact merge laws, inherited rules,
binders, attribute witnesses, literal handling, and atomic RHS rejection.
