# Custom e-class analysis

An **e-class analysis** maintains facts about equivalent expressions. TEPL's
generated `TensorAnalysis` tracks shape and dtype. A custom host analysis can
also track constants, value ranges, or other application facts, and provide
inference for operations that need host knowledge.

This page describes the Rust integration. The [Rust tutorial](../tutorial/rust/03_analyse.md)
covers using the generated tensor analysis directly.

## Facts belong to the whole e-class

An e-class can contain several operations representing the same value. Its
metadata must account for all alternatives, including after classes merge.
Selecting the metadata of one arbitrary node is insufficient.

The generated `TensorAnalysisData` retains three kinds of evidence:

| Evidence | Behavior |
| --- | --- |
| Known shape and dtype | Exposed only when all alternatives are known and agree. |
| Unknown metadata | Retains known evidence but prevents exposing a complete tensor type. |
| Invalid metadata or conflicting known types | Marks the class invalid. |

For example, merging evidence for `f32[2, 3]` with evidence for `f32[3, 2]`
creates a conflict. An unknown alternative does not erase either known type.
Marking the conflict cannot undo an incorrect e-class union; rewrites must
validate replacements before merging them.

## Connect a host analysis to generated rules

Implement `egg::Analysis<OpNode>` to compute and merge your facts:

- `make` computes a node's facts from its operation, attributes, and child facts.
- `merge` combines evidence when e-classes unite and reports changes to egg.
  Combining evidence must be independent of order and repeated merges.

Then implement the generated `RewriteAnalysis` interface. This adapter assumes
`MyAnalysis` already implements `egg::Analysis<OpNode>` with
`type Data = TensorAnalysisData`:

```rust
use egg::{EGraph, Id};
use generated::{Op, OpAttrs, OpNode, RewriteAnalysis, TensorInfo};
use generated::analysis::{TensorAnalysisData, infer_tensor_output};

impl RewriteAnalysis for MyAnalysis {
    const HAS_TENSOR_INFO: bool = true;

    fn tensor_info(graph: &EGraph<OpNode, Self>, id: Id) -> Option<TensorInfo> {
        graph[graph.find(id)].data.info().cloned()
    }

    fn infer_output(
        &self,
        op: Op,
        operands: &[TensorInfo],
        attrs: &OpAttrs,
    ) -> Option<TensorInfo> {
        infer_tensor_output(op, operands, attrs)
    }
}
```

`tensor_info` supplies agreed metadata for captures. `infer_output` validates
new replacement operations; the example delegates to generated inference.
If an operation needs host inference, use the same policy in this method and
in your analysis's `make`. For `make`, `analysis::infer_tensor` preserves the
known, unknown, and invalid outcomes instead of reducing them to `Option`.

You can reuse `TensorAnalysisData::merge` for tensor evidence. If your e-class
data also stores custom facts, embed the tensor evidence in that data and update
the adapter to read it. Custom facts need their own sound merge policy and host
integration to use them in rewrite guards; they do not become TEPL fields
automatically.

## Missing information is not permission

With `HAS_TENSOR_INFO = true`, returning `None` for a capture or replacement
rejects candidates that need that information. It does not switch to structural
rewriting. The unit analysis `()` is the separate structural mode used in the
first tutorial.

Register input metadata before insertion. The generated tensor evidence
accumulates unknown and conflicting facts; build a fresh graph when previously
missing input metadata changes. Stable metadata answers are required during
matching.

See the [generated analysis implementation](../../templates/rust/src/analysis/tensor_analysis.rs)
for `make` and `merge`, and the
[rewrite interface](../../templates/rust/src/rewriting/context.rs) for the
adapter contract and optional literal inference hook.
