# XLA specification size: matched axis-only rules

A fresh run checks all **32** axis-inference cases from the frozen axis-only cohort. **30** inferred rules verify, compile, and validate; **28** cases pass the same checks for both matched variants and form the primary size cohort.

For these paired rules (69 axis fields), lexical tokens decrease from **2,946 to 2,709** (**8.0%**). The median per-rule reduction is **5.5%**. This measures specification size, not usability, runtime speed, or equivalence to unrestricted XLA behavior.

The matched baseline uses named fields in both variants. In the **26** cases where the original compact handwritten rules also verify, the original rules have **2,380 tokens**, compared with **2,505** for the current inferred form. The inferred form changes this by **+5.3%**. Inference shortens field expressions, but the current migration expands compact descriptor copies; the controlled reduction is not a claim that it shortens the original files.

## Paired specification size

| Representation | Layout lines | Lexical tokens |
|---|---:|---:|
| Corresponding XLA implementation excerpts | Pending excerpt audit | Pending excerpt audit |
| TEPL with explicit RHS axis fields | 375 | 2,946 |
| TEPL with inferred RHS axis fields | 375 | 2,709 |

Layout lines use the same brace/semicolon/arrow line-break policy for both TEPL variants. They exclude comments and blank lines. Lexical tokens, rather than lines, are the primary metric. Both totals include the complete rule: constraints, patterns, guards, descriptor syntax, and derivations. Imports and shared dialect definitions are reported separately. See [the protocol](../protocol.md).

## Where the savings come from

Every rule token belongs to exactly one component; the differences sum to the whole-rule reduction. Positive savings mean fewer tokens in the inferred variant.

| Component | Explicit tokens | Inferred tokens | Saved |
|---|---:|---:|---:|
| Axis field expressions | 306 | 69 | 237 |
| Local helper computations | 0 | 0 | 0 |
| Supplied non-axis field expressions | 94 | 94 | 0 |
| Descriptor labels, field names, and punctuation | 483 | 483 | 0 |
| Fixed normalization adapter assignments | 12 | 12 | 0 |
| Tensor shape/dtype constraints | 215 | 215 | 0 |
| LHS pattern | 369 | 369 | 0 |
| RHS pattern | 311 | 311 | 0 |
| Conditions and where locals | 932 | 932 | 0 |
| Rule declaration and outer syntax | 140 | 140 | 0 |
| Derive block syntax | 84 | 84 | 0 |
| **Total** | **2,946** | **2,709** | **237** |

### Axis-expression detail

This table counts only field-value expressions. Any removed helper statements appear in the component table, not again here.

| Handwritten expression form | Fields | Explicit tokens | Auto tokens | Saved | Share of total savings |
|---|---:|---:|---:|---:|---:|
| field copy | 33 | 165 | 33 | 132 | 55.7% |
| empty list | 15 | 30 | 15 | 15 | 6.3% |
| literal | 19 | 65 | 19 | 46 | 19.4% |
| helper reference | 0 | 0 | 0 | 0 | 0.0% |
| computed expression | 2 | 46 | 2 | 44 | 18.6% |

## Breakdown by family

| Family | Paired cases | Explicit tokens | Inferred tokens | Saved (%) |
|---|---:|---:|---:|---:|
| broadcast | 8 | 739 | 710 | 3.9% |
| dot | 11 | 1,577 | 1,444 | 8.4% |
| indexing | 3 | 225 | 182 | 19.1% |
| reduction | 5 | 357 | 338 | 5.3% |
| transpose-reshape | 1 | 48 | 35 | 27.1% |

## Sensitivity checks

| Cohort | Cases | Explicit tokens | Inferred tokens | Saved (%) |
|---|---:|---:|---:|---:|
| All 32 attempted pairs (includes unproved rules) | 32 | 3,514 | 3,205 | 8.8% |
| Paired verified rules | 28 | 2,946 | 2,709 | 8.0% |
| Not marked scalar-forced | 26 | 2,798 | 2,569 | 8.2% |
| All targets marked scalar-forced | 2 | 148 | 140 | 5.4% |
| Without unit-view normalization | 26 | 2,738 | 2,505 | 8.5% |
| With unit-view normalization | 2 | 208 | 204 | 1.9% |

The frozen classifier marks a field forced only for a rank-zero broadcast's empty mapping. It does not establish that other tasks have multiple possible axis assignments. A general uniqueness or intended-optimization audit is a separate experiment.

### Comparison with the original source

This table uses only cases where all three TEPL variants verify, compile, and validate. It retains the original descriptor-copy idioms rather than expanding them away.

| Representation | Cases | Layout lines | Lexical tokens |
|---|---:|---:|---:|
| Original handwritten explicit rules | 26 | 268 | 2,380 |
| Matched named-field explicit rules | 26 | 347 | 2,738 |
| Current inferred rules | 26 | 347 | 2,505 |

The original handwritten files have **2,564 rule tokens** over the primary cohort; **26/28** of those original rules pass current proof/compile/validation. Their difference from the matched explicit baseline reflects descriptor expansion, normalization, and dead-helper cleanup; it is not attributed to inference. All per-rule original sizes and outcomes are retained in the result JSON.

## Per-case results

O/E/I mean original handwritten / matched explicit / inferred. Pass means proof, compilation, and JSON validation all succeed. Savings for excluded pairs are descriptive only.

| Case | O/E/I pass | Primary cohort | Explicit tokens | Inferred tokens | Saved |
|---|---|---|---:|---:|---:|
| transpose_compose | yes/yes/yes | yes | 48 | 35 | 13 |
| broadcast_add | yes/yes/yes | yes | 81 | 77 | 4 |
| broadcast_multiply | yes/yes/yes | yes | 81 | 77 | 4 |
| scalar_broadcast_reverse | yes/yes/yes | yes | 58 | 54 | 4 |
| scalar_broadcast_transpose | yes/yes/yes | yes | 90 | 86 | 4 |
| reduce_negate | yes/yes/yes | yes | 40 | 36 | 4 |
| reduce_compose | yes/yes/yes | yes | 70 | 66 | 4 |
| reduce_transpose | yes/yes/yes | yes | 68 | 66 | 2 |
| reverse_compose_disjoint | yes/yes/yes | yes | 68 | 64 | 4 |
| reverse_add_scalar | yes/yes/yes | yes | 85 | 77 | 8 |
| dot_associate | yes/yes/yes | yes | 209 | 177 | 32 |
| dot_transpose_output | yes/yes/yes | yes | 121 | 115 | 6 |
| dot_move_reverse | yes/yes/yes | yes | 144 | 124 | 20 |
| reduce_compose_leading | yes/yes/yes | yes | 70 | 66 | 4 |
| broadcast_compose_append | yes/yes/yes | yes | 101 | 97 | 4 |
| dot_swap_operands | yes/yes/yes | yes | 123 | 113 | 10 |
| dot_no_contraction | yes/yes/yes | yes | 114 | 110 | 4 |
| dot_reduce_product | yes/yes/yes | yes | 87 | 85 | 2 |
| dot_reduce_batch | yes/yes/yes | yes | 115 | 109 | 6 |
| expr_reverse_compose_general | yes/yes/yes | yes | 72 | 41 | 31 |
| catalog_broadcast_divide_broadcast | no/no/no | no | 79 | 75 | 4 |
| catalog_broadcast_multiply_constant_into_broadcast | yes/yes/yes | yes | 120 | 115 | 5 |
| catalog_broadcast_transpose | no/no/yes | no | 94 | 45 | 49 |
| catalog_broadcast_inserted_unit_dims | no/yes/yes | yes | 113 | 111 | 2 |
| catalog_broadcast_drop_unit_dims | no/yes/yes | yes | 95 | 93 | 2 |
| catalog_broadcast_transpose_broadcast | no/no/no | no | 267 | 267 | 0 |
| catalog_indexing_reverse_inserted_unit_dims | no/no/yes | no | 128 | 109 | 19 |
| catalog_dot_transpose_both_operands | yes/yes/yes | yes | 177 | 157 | 20 |
| catalog_dot_move_broadcast | yes/yes/yes | yes | 150 | 144 | 6 |
| catalog_dot_reduce_output_free | yes/yes/yes | yes | 146 | 136 | 10 |
| catalog_dot_scalar_multipliers | yes/yes/yes | yes | 191 | 174 | 17 |
| catalog_reduction_broadcast_original_axes | yes/yes/yes | yes | 109 | 104 | 5 |

### Unproved matched variants

An unproved explicit baseline is not evidence that its handwritten assignment is wrong; it can expose a metadata-definedness or proof-support limit. These cases are excluded from the primary paired claim.

| Case | Variant | Reason |
|---|---|---|
| catalog_broadcast_divide_broadcast | explicit | normalized coefficients, tensor accesses, or reduction domains differ |
| catalog_broadcast_divide_broadcast | inferred | normalized coefficients, tensor accesses, or reduction domains differ |
| catalog_broadcast_transpose | explicit | metadata definedness: comprehension is outside the pointwise metadata fragment |
| catalog_broadcast_transpose_broadcast | explicit | metadata definedness: comprehension is outside the pointwise metadata fragment |
| catalog_broadcast_transpose_broadcast | inferred | metadata definedness: comprehension is outside the pointwise metadata fragment |
| catalog_indexing_reverse_inserted_unit_dims | explicit | The current checker cannot establish the handwritten attribute's legality from the LHS and conditions. |

## XLA excerpt audit

The catalog's enclosing source ranges contain **2,154 nonblank code lines** and **16,713 lexical tokens** for this cohort after file/range deduplication. These contexts include unrelated branches and source-policy checks. They are not corresponding implementation excerpts and must not be used as an XLA-to-TEPL compression ratio.

**0/28** paired cases have reviewed excerpt boundaries. The three-way table stays incomplete until every paired case is reviewed. The [excerpt manifest](../data/xla_excerpts.json) records contexts, selected ranges, and review notes. Count overlapping excerpt lines once globally, and separately document shared helpers and policy checks.

## Shared setup and evidence

| Shared dialect file (whole file; not a minimal dependency slice) | Code lines | Tokens |
|---|---:|---:|
| `axis_views.tepl` | 17 | 135 |
| `hlo.tepl` | 423 | 3,875 |

Per-file imports add **615 explicit / 615 inferred tokens** before deduplication. Shared dialects are identical across the paired variants and are not multiplied by the number of rules. The negative control fails semantic verification as required.

Run UTC: `2026-10-29T19:20:50.378614+00:00`. Compiler SHA-256: `96258055f9087add6e7857b0767c69e6c5191a303feddd54d7c011ddbbc57912`. Repository revision: `906b6578b73c3e6c7631ede30025ad5b713c2398`; input hashes include the experiment's working-tree sources. Arithmetic policy: `algebraic`; 30 seconds per command; 4 workers.

[Raw measurements](../results/measurement.json) include complete measured paired sources, per-field expressions, disjoint component counts, compiler commands and diagnostics, artifact hashes, and input hashes. Historical inference results remain unchanged. This is a source-size experiment on a fixed cohort; it does not establish that the generated rules preserve every XLA numerical, layout, or profitability policy.
