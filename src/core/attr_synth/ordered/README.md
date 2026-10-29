# Ordered synthesis implementation

This directory owns contraction/permutation recognition, ordering requirements,
and descriptor construction. It implements the original ordered specialization;
broader signed/weighted mathematical state-table procedures are research work.

| Files/API | Responsibility |
| --- | --- |
| `contract.*`, `recognize(Definition)` | Conservative structural recognition |
| `admit(Plan, Mode)` | Semantic/tree and attribute-policy checks |
| `compile(Plan, Mode)` | Completed instructions and field adapters |
| `deriveRequirements(...)` | Required live classes and subsequences |
| `construct(...)` | Bottom-up positional descriptor construction |
| `concrete.*` | Plan adapters for native evaluation and certificates |

Candidates require independent verification under the original conditions.
Admission failure allows general synthesis to continue; it does not prove
infeasibility. Logical identity uses physical-port equivalence, not equal extents.

The [synthesis design](../../../../design/attribute-synthesis.md) describes
integration, concrete-domain restrictions, evaluation, and validation. The
[research index](../../../../research/attribute-synthesis/README.md) identifies
current proofs and their relationship to implementation.

```sh
bazel test //tests:ordered_synthesis_test //tests:attribute_reasoning_test \
  //tests:where_suggestions_cli_test //tests:named_nodes_test
```
