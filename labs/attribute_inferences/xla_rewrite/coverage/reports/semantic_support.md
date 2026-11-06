# Semantic support inventory

This is an additional inventory of the 209 concrete cases in the frozen
axis-only evaluation. A subsequent [separate run](expanded_semantics.md) tested
the eleven newly included cases with a different compiler; the tables below
preserve the inventory and the original frozen results. Cases and schemas are different
units: the cases represent 199 of the 239 cataloged schemas. The negative
control is separate. Semantic support here means the operations have local
value-semantic declarations, or an existing checked unit-view adapter, under
the case's stated restrictions. This inventory is a screening criterion; it
does not itself establish that the verifier can prove each case or that every
source-side domain condition has been audited.

## Semantic inventory and RHS tasks

| RHS task | Semantically modeled candidates | Previously evaluated | Newly included, not evaluated |
|---|---:|---:|---:|
| At least one original RHS axis field | 38 | 32 | 6 |
| Supplied non-axis RHS attributes only | 9 | 5 | 4 |
| No RHS attributes | 76 | 75 | 1 |
| Total | 123 | 112 | 11 |

Of the other 86 cases, the inventory has not established semantic support:
one or more required operations lack a semantic declaration, or a checked
adapter has not been established for the case. This is not a proof that no
restricted implementation or future adapter could support them. Four of the
123 use existing checked unit-only reshape adapters even though general
reshape lacks a semantic declaration.

The 11 newly included cases all use semantically declared operations but were
excluded by the axis-only protocol: ten involve static slice and one uses the
restricted convolution declaration. That convolution declaration models valid,
ungrouped, fixed-layout cross-correlation, not general HLO convolution. The
case-specific domain and full semantic applicability must still be audited.

## Additional axis tasks

The new axis cases are `slice_reverse`, `expr_slice_reverse_general`,
`expr_convolution_scale_filter_bias`, `catalog_broadcast_slice_broadcast_only`,
`catalog_broadcast_slice_broadcast`, and `catalog_dot_slice_output`. Their
axis fields were inspected with the existing role-directed `AxisTask` parser;
their subsequent outcomes are in the separate run.

## Verification-only outcomes

| RHS task | Verified | Unresolved | Not tested | Total |
|---|---:|---:|---:|---:|
| Supplied non-axis attributes only | 4 | 1 | 4 | 9 |
| No RHS attributes | 61 | 14 | 1 | 76 |

The tested results are those of the frozen run in [attribute_synthesis.md](attribute_synthesis.md).
Success requires semantic proof and executable compilation and validation; the
unresolved cases are not infeasibility results. Original source conditions,
semantic restrictions, and arithmetic policy continue to apply. The new cases'
separate run is reported in [expanded_semantics.md](expanded_semantics.md).

Inventory procedure: exclude the negative control; for each recorded case,
check its `required_operations` against [hlo_support.json](../data/hlo_support.json),
then retain the four already-evaluated cases with checked unit-view adapters.
Use the frozen [classification](../data/synthesis_classification.json) and
[results](../results/synthesis.json) for existing tasks and outcomes; inspect
new cases with `AxisTask` from [axis_fields.py](../scripts/axis_fields.py).
