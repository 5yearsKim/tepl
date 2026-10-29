# Tensor analysis

Generated Rust programs provide a concrete `TensorAnalysis` backed by the shared
`tepl-runtime` crate. Custom analysis adapters are deferred. Applications supply
input metadata; TEPL operation definitions supply shape and dtype contracts.
The [C++ runtime](../../runtime/cpp/README.md) uses the same fact and conflict model.

## Build a graph

```rust
use generated::{DType, EGraph, OpNode, TensorAnalysis, TensorFacts};
use generated::dialects::tensor_lang::{Op, TransposeAttrs};

let analysis = TensorAnalysis::new([
    ("X".into(), TensorFacts::known(vec![2, 3], DType::F32)),
])?;
let mut graph = EGraph::new(analysis);
let x = graph.add(OpNode::input("X"));
let transposed = graph.add(OpNode::operation(
    Op::Transpose,
    Some(TransposeAttrs { permutation: vec![1, 0] }.into()),
    vec![x],
)?);
graph.analysis.check()?;
```

Register metadata before inserting an input node. Generated graph constructors
perform registration automatically.

## Facts and conflicts

Each e-class stores independent shape and dtype facts:

- `Unknown`: no information is available.
- `Known(value)`: the value is established.
- `Conflict`: equivalent expressions disagree.

Merging unknown and known facts preserves the known value. Equal known values
remain known. Different known values produce a conflict. Rebuilding propagates
new information through dependent nodes.

Operation evaluators execute compiled shape and dtype programs. An axes contract
provides shape inference when there is no explicit shape program.
Axis identity is independent of extent: two dimensions of size two may represent
different axes. The e-class stores the resulting shape, not those temporary
identities.

## Run with diagnostics

```rust
let rules = generated::rules::all()?;
let runner = tepl_runtime::runner::runner(graph)?.with_iter_limit(20);
let result = tepl_runtime::runner::run(runner, &rules)?;
```

Conditions with unavailable required metadata skip a match. Evaluation errors
reject candidates and record rule/source diagnostics in
`result.egraph.analysis.rejections()`. Conflicting metadata and resource limits
stop optimization; egg cannot undo a union after discovering a conflict.
Replacements are prepared and checked before insertion, and known output
metadata mismatches reject the candidate.

If integrating directly with egg, call `graph.analysis.check()` after insertion,
rewriting, and rebuilding. The runtime runner performs error propagation for you.
