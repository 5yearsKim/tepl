# Tensor analysis and rewrite safety

An e-class represents one value through several equivalent expressions. Its
analysis facts must describe that shared value consistently.

TEPL's generated Rust and C++ runtimes provide a concrete `TensorAnalysis`.
The host supplies input metadata; operation contracts infer facts for results.
Custom analysis adapters are deferred.

## Follow a fact through the graph

For the transpose tutorial:

```text
X                       f32[2, 3]
transpose[1, 0](X)       f32[3, 2]
transpose[1, 0](above)   f32[2, 3]
```

The first and last expressions may share an e-class after applying the valid
rewrite. Their metadata agrees. Equal sizes alone would not justify that union:
a square matrix and its transpose usually contain different values.

## Merge facts

Shape and dtype are tracked independently. Each fact has three states:

| Existing fact | Incoming fact | Merged fact |
| --- | --- | --- |
| `Unknown` | `Known(v)` | `Known(v)` |
| `Known(v)` | `Known(v)` | `Known(v)` |
| `Known(a)` | `Known(b)`, where `a != b` | `Conflict` |
| `Conflict` | Any | `Conflict` |

Merging two unknown facts leaves the result unknown. Merge order does not change
these results. Rebuilding propagates established facts through dependent nodes.

`Unknown` is missing information. `Conflict` means expressions treated as equal
disagree about their metadata. Checked execution stops on a conflict. An e-graph
cannot undo a bad union just because analysis discovers the problem later.

## Check before inserting

For each candidate, the runtime checks capture constraints and conditions,
evaluates attribute derivations, and validates the complete replacement.
Known root shape or dtype changes reject the candidate before insertion.
Required metadata that is unavailable skips the match.

Evaluation failures reject candidates and appear in
`graph.analysis.rejections()` with rule/source diagnostics. Conflicts and resource
limits stop the checked runner. A failed replacement leaves no partial RHS nodes.

These checks complement [compiler semantic verification](../../language/semantics.md).
They do not establish arbitrary value equivalence or provide general tensor-value
range analysis.

## Supply inputs first

Generated graph constructors register declared input facts before inserting nodes.
For manual construction, register the metadata yourself before adding each input.
Use the checked runtime runner to propagate analysis errors; direct egg users
must call `graph.analysis.check()` after insertion, rewriting, and rebuilding.

Continue with [Rust input metadata](rust/analysis.md) or
[C++ input metadata](cpp/analysis.md). For matching details, see
[binding and rule evaluation](binding.md).
