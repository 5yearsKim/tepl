# Generated tensor IR and rules with egg

`src/ir/` contains generated code from the TEPL project under `examples/`, plus
the handwritten `analysis/` reference for shape/dtype inference and direct egg
analysis. Import `TensorAnalysis`, `TensorBindingTable`, and `TensorInfo` from
`rust_egg::ir::analysis`; applications supply input metadata and construct the
graph. `src/host/` contains only application-specific node helpers and LoRA host
functions. Change dialects and rules in their TEPL source, then regenerate.

The compiler emits the shared `analysis/shape_builtins.rs` runtime, but does not
emit the operation shape or dtype evaluators yet. Regeneration
replaces the sample's `pub mod analysis;` registration in `src/ir/mod.rs` with an
inline module exposing only `shape_builtins`; restore the external module
registration to use the sample. Until generator support is added, `--check`
reports this intentional difference. The handwritten analysis sample files
remain outside the generated-file manifest and are preserved.

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
  analysis/shape_builtins.rs                       # copied runtime helpers
  analysis/mod.rs                                # public analysis API
  analysis/{shape.rs, dtype.rs, tensor.rs}         # operation inference reference
  analysis/{tensor_analysis.rs, tensor_analysis_data.rs, bindings.rs}      # direct egg analysis reference
  analysis/inference.rs                          # inference result type
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

Dialect operations and attributes follow `examples/dialects/`, including
`exponential` (`exp` alias), `broadcast_in_dim`, two-operand `reduce`, and full
dot, convolution, collective, and constant descriptors. The compiler parses
shape blocks into a structured AST and checks them in core; evaluator generation
remains deferred. The sample in `ir/analysis/shape.rs`
manually demonstrates evaluators for add, multiply, negate, reshape, transpose,
scalar operations, and general dot contraction. Other operations return Unknown
and may be resolved by the host. Variadic shape assertions such as nonempty
concatenation remain deferred.

`OpNode::input(name)` creates a runtime input leaf outside TensorLang. The lab's
`symbol` helper wraps it. Constants take `types::Elements` (element type, shape,
and canonical payload bytes), and reductions take a `types::Region` plus explicit
input and initialization operands. Regions and mesh group specifications are
opaque canonical host encodings; hosts validate their contents. Precision values,
optional dot algorithms, signed padding, and collective metadata are preserved
in node identity and matching. No obsolete TensorLang operation aliases remain.

The tests exercise the generated modules directly: structural matching,
attribute witnesses, binders, literals, dtype and shape restrictions, abstract
instances, cross-dialect lowering, checked replacement, and LoRA saturation.
Structural-only fixtures provide explicit metadata and inference; semantic
fixtures use the same `ir::analysis` implementation as the example.

## Direct tensor analysis

The ownership boundary is explicit:

- `ir/analysis/shape.rs`: operation shape evaluators with no host or egg dependency.
- `ir/analysis/dtype.rs`: operation dtype evaluators with no host or egg dependency.
- `ir/analysis/shape_builtins.rs`: internal checked primitives maintained in
  `runtime/rust/src/analysis/shape_builtins.rs` and copied by generation.
- `ir/analysis/inference.rs`: the small `Inference<T>` result type.
- `ir/analysis/mod.rs`: the public API for inference and ready-to-use analysis.
- `ir/analysis/bindings.rs`: the input name-to-metadata table; reads runtime input
  nodes directly without depending on host helpers.
- `ir/analysis/tensor_analysis_data.rs`: conservative `TensorAnalysisData` and their e-class merge policy.
- `ir/analysis/tensor.rs`: `infer_tensor` combines shape and dtype results and
  reads typed constant payloads. `infer_tensor_output` returns the same result as
  an `Option` for rewrite builders.
- `ir/analysis/tensor_analysis.rs`: the concrete `TensorAnalysis` implementation of
  `egg::Analysis<OpNode>` and the ready-made `tensor_info` metadata callback.
- `host/lora.rs`: application-specific LoRA conditions and specialized dot helpers.
- `host/nodes.rs`: convenience constructors and accessors used by the examples.

All files under `ir/analysis` depend only on other IR modules and egg, never on
`host/`. Applications do not implement analysis, fact merging, or metadata
adapters. The existing rewrite builder still takes the provided `tensor_info`
and `infer_tensor_output` functions; this relocation does not change its API.

The pure API takes an operation, operand shapes or dtypes, and attributes:

```rust
use rust_egg::ir::analysis::{infer_shape, infer_dtype, Inference};

// infer_shape(op, &[&[u64]], attrs) -> Inference<Vec<u64>>
// infer_dtype(op, &[DType], attrs)  -> Inference<DType>
```

Neither function accesses an e-graph or host configuration. `TensorAnalysis`
reads operand metadata and calls `infer_tensor`, which combines both results into
`TensorInfo`. The analysis stores this information in `TensorAnalysisData`.

The handwritten dtype reference makes the sample contract explicit: add,
multiply, negate, and default-precision dot preserve matching non-bool operand
types; reshape and transpose preserve any operand type; literals use their
explicit dtype. Dot with non-default precision or an algorithm override returns
`Unknown`, as do operations without a dtype evaluator. Malformed calls to known
evaluators return `Invalid`. These dtype rules are not yet encoded in TEPL or
emitted by the compiler; defining and generating them remains future work.

Shape evaluators return `ShapeResult<Vec<u64>>` and use `?` to propagate helper
errors. `infer_shape` dispatches operations and converts these results to
`Known` or `Invalid`, reserving `Unknown` for missing definitions. All 19 builtins
listed in `examples/shape_guide.md` are available, plus `ensure`, including list
operations, Boolean and integer reductions, broadcasting, and signed rounded
division. Gathering allows repeated indices; operations such as transpose
separately assert axis uniqueness. The helpers check overflow and indexing in
both debug and release builds. See [the runtime builtin contract](../../runtime/rust/README.md)
for calling conventions and integer types. Compiling TEPL shape blocks into
evaluators remains future work.

`TensorAnalysis` holds the input bindings directly. There is
one analysis object per graph, while every e-class stores `TensorAnalysisData` directly:

```rust
pub struct TensorAnalysis {
    pub symbols: TensorBindingTable,
}

impl egg::Analysis<OpNode> for TensorAnalysis {
    type Data = TensorAnalysisData;
    // make reads child facts and calls shared tensor inference.
    // merge combines facts and tells egg whether they changed.
}
```

Register input metadata before constructing the graph:

```rust
use egg::EGraph;
use rust_egg::ir::analysis::{TensorAnalysis, TensorBindingTable, TensorInfo};
use rust_egg::ir::{DType, OpNode};

let mut symbols = TensorBindingTable::default();
symbols.register_symbol("X", TensorInfo {
    shape: vec![4],
    dtype: DType::F32,
}).unwrap();
let mut graph = EGraph::new(TensorAnalysis::new(symbols));
let x = graph.add(OpNode::input("X"));
assert_eq!(graph[graph.find(x)].data.info().unwrap().shape, vec![4]);
```

`make` gets input facts from the host or computes an operation's output from its
children. Invalid operands take precedence over unknown operands. `merge`
delegates to `TensorAnalysisData::merge`, allowing egg to propagate changes to parents
during rebuilding. `ir/analysis` owns the facts, inference, and egg integration;
the application supplies input metadata and selects its rewrites.

Generated rewrites read metadata through the ordinary `tensor_info` callback.
The ordinary `infer_tensor_output` function checks a proposed RHS before
insertion, using the same inference as the analysis:

```rust
use rust_egg::ir::analysis::{TensorAnalysis, infer_tensor_output, tensor_info};
use rust_egg::ir::rules::simple::rule_commute_add;

let rewrite = rule_commute_add::build_rewrite(
    tensor_info,
    infer_tensor_output,
    (),
).unwrap();
let runner = egg::Runner::<_, TensorAnalysis>::new(TensorAnalysis::default())
    .with_egraph(graph)
    .run(&[rewrite]);
```

There is no semantics trait, configurable fallback, or inference adapter object.
To read the main flow, start with `ir/analysis/tensor_analysis.rs`, then `tensor.rs`, then
`tensor_analysis_data.rs`: read input/child metadata, infer one operation, and store or merge
its facts. Constants are handled explicitly in `tensor.rs`; unsupported operation
inference remains `Unknown`. Generated invalid results remain `Invalid`.
Both Unknown and Invalid reject RHS validation. The sample fact store records
invalid evidence but does not retain diagnostic messages per e-class.

Tensor facts expose metadata only when all alternatives are known and agree on
shape and dtype. Unknown alternatives block metadata without erasing known
evidence, so later conflicts are detected regardless of merge order. Facts only
accumulate: registering missing input types later requires a fresh graph. This
sample does not recompute or retract unknown facts.

`tests/analysis.rs` exercises the direct analysis with a generated rewrite,
propagation of missing and conflicting operand metadata, conservative merge
laws, unknown/invalid inference, pure dtype rules, checked shape evaluation,
and agreement between analysis and RHS inference.

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
