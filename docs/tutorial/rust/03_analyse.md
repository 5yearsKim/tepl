# Add tensor analysis

Generated programs use a concrete `TensorAnalysis` to propagate shape and dtype
facts through egg. Define operation contracts in TEPL and supply input metadata.

## Define operation metadata

```javascript
dialect D {
    op transpose(input: tensor) -> tensor {
        attrs { permutation: axis[]; }
        dtype(t) { yield t; }
        shape(s) {
            assert is_valid_axis_list(attrs.permutation, len(s));
            assert len(attrs.permutation) == len(s);
            yield gather(s, attrs.permutation);
        }
    }
}
```

For semantic operations, an axes contract can calculate output shape instead.
The compiler also uses scalar/index semantics to verify rewrites and synthesize
attributes. Generated Rust evaluates the completed metadata expressions.

## Supply input facts

Generated TEPL graph constructors register declared input shapes and dtypes.
For manual graph construction:

```rust
use generated::{DType, EGraph, OpNode, TensorAnalysis, TensorFacts};
use generated::dialects::d::{Op, TransposeAttrs};

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

The output shape is `[3, 2]` and dtype is `F32`. Register input metadata before
inserting its node.

## Read partial facts

Shape and dtype are independent. Each is `Unknown`, `Known(value)`, or `Conflict`.
An operation without a contract leaves the corresponding result unknown.
Rebuilding propagates newly established facts through dependent nodes.

Rules skip matches when required metadata is unavailable. A known RHS shape or
dtype mismatch rejects a candidate before insertion. Conflicting facts mean
that equivalent expressions disagree; optimization stops with a diagnostic.
Use the [checked runner](02_run.md#run-egg) to receive these errors.

The [analysis guide](../../advanced/custom_analysis.md) explains merge behavior.
The [tutorial lab](../../../labs/tutorial_rust/README.md) demonstrates factoring
a matrix product while preserving output metadata.
