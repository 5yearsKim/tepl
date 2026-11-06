# Supply tensor metadata

Generated graph constructors register input shapes and dtypes automatically.
For manual construction, register facts before inserting each input node.
This example uses the same generated module as [run generated rules](run.md).

## Build a graph manually

Inside your application's `main` function:

```rust
use generated::{DType, EGraph, OpNode, TensorAnalysis, TensorFacts};
use generated::dialects::tensor::{Op, TransposeAttrs};

let analysis = TensorAnalysis::new([
    ("X".into(), TensorFacts::known(vec![2, 3], DType::F32)),
])?;
let mut graph = EGraph::new(analysis);
let x = graph.add(OpNode::input("X"));
let root = graph.add(OpNode::operation(
    Op::Transpose,
    Some(TransposeAttrs { permutation: vec![1, 0] }.into()),
    vec![x],
)?);
graph.analysis.check()?;
println!("{:?}", graph[root].data);
```

The output shape is `[3, 2]` and dtype is `F32`. The tutorial dialect's `axes`
contract supplies shape inference and its `dtype` contract preserves the input
dtype. A dialect can also define an explicit `shape` program.

## Run with checked analysis

```rust
let rules = generated::rules::all()?;
let runner = tepl_runtime::runner::runner(graph)?.with_iter_limit(20);
let result = tepl_runtime::runner::run(runner, &rules)?;
```

Shape and dtype are independent facts: `Unknown`, `Known(value)`, or `Conflict`.
Missing required facts skip a match. Known replacement shape/dtype changes reject
a candidate before insertion; conflicts stop checked execution.

Inspect `result.egraph.analysis.rejections()` for candidate evaluation failures.
The shared [analysis guide](../analysis.md) explains merging and rewrite safety.
