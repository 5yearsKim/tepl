# Broadcast–transpose inference update

The [companion rule](../../tepl/rules/experiments/broadcast_transpose_broadcast.tepl)
expresses the same general-rank insertion-only broadcast/transpose graph as the
frozen `catalog_broadcast_transpose_broadcast` task. It computes the supplied
intermediate shape by placing the input and inserted extents in their original
positions, then permutes and filters that shape. It does not add conditions or
supply the RHS axis mapping. The original frozen source, classification, and
results remain unchanged.

The verifier recognizes two general sequence facts: a unique-match sum over a
proven permutation computes its inverse (or selects from an equal-length
parallel list); and selecting the inserted part of a permuted interleaving
orders it by its permuted positions. The facts require established permutation,
uniqueness, and length obligations. The independent proof checker also checks
comprehension variable scope and source metadata definedness before accepting
supplied expressions. These changes do not add general support for arbitrary
symbolic comprehensions, including the original shape calculation's nested
conditional expression.

A fresh, separate run of all 112 previously in-scope cases, using this companion
in place of that one frozen task, is recorded in
[broadcast_update.json](../results/broadcast_update.json). Its result is:

| Task | Verified success | Unresolved |
|---|---:|---:|
| Axis inference | 31/32 | 1 |
| No RHS attributes | 61/75 | 14 |
| Supplied non-axis RHS attributes | 4/5 | 1 |
| Total | 96/112 | 16 |

The only case whose outcome changed from the frozen run is the companion
broadcast–transpose case. Scalar division remains unresolved. The separate
negative control remains unproved. The new result is not a rerun of the exact
frozen source expression: that original expression still exceeds the metadata
checker fragment, so do not overwrite or relabel the historical 30/32 result.
The run has its own compiler and dependency hashes in the JSON. The arithmetic
policy and success criteria are the same as the [axis-only protocol](axis_only_scope.md).

The concrete execution test checks the compiled companion on scalar, zero-size,
permuted and unsorted-axis inputs and compares dense tensor results. It also
checks that an incorrectly ordered RHS extent list is rejected. The symbolic
unit tests reject unproved permutation and length premises, altered sums,
unbound variables, shadowing, and unknown operations inside comprehensions.

Reproduce from the repository root:

```sh
bazel build //:tepl
python3 labs/xla_rewrite/coverage/scripts/run_broadcast_update.py
bazel test //tests:ordered_synthesis_test //tests:semantic_test //tests:scalar_test //tests:semantic_cli_test //tests:axis_auto_test --test_output=errors
python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p test_rule_execution.py
```
