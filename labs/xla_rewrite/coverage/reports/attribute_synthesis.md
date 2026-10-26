# XLA Attribute Synthesis Evaluation

TEPL automatically infers RHS attributes for **32/32 (100.0%)** of the attribute-bearing cases whose explicit baseline already verifies. This includes **10/10 verified theorem-fragment cases**. Across all supported attribute-bearing cases, including cases with unproved explicit baselines, auto succeeds on **36/43 (83.7%)**, spanning **30 catalog schemas**.

The experiment infers RHS attributes for **36 cases** in total. Full auto succeeds on **36 cases**; partial auto succeeds on **11**, including **0 partial-only successes**. Overlapping full and partial results count once.

Combining verified explicit rules, inferred rules, and attribute-free identities gives **99/119 (83.2%) supported cases** with a semantic proof. **4 cases gain a verified implementation through inference even though their explicit metadata formulas remain unproved.**

## Experiment setup

The evaluation follows three steps: migrate the existing corpus, freeze theorem-based categories, then measure explicit, full-auto, and partial-auto variants. It uses the ideal `algebraic` policy. A successful auto variant has an explicit semantic proof, the expected number of inferred descriptors, successful executable compilation, and validated JSON. Equations, input declarations, and written conditions are preserved. No suggested restrictions are added.

The corpus contains **239 catalog schemas**, **209 concrete cases representing 199 schemas**, and one negative control. Concrete cases retain the scopes recorded in the existing manifest: some specialize ranks, arithmetic operations, layouts, or constant representations. Schema coverage means at least one measured case, not every XLA variant.

## Step 1 Syntax migration

**136 legacy semantic cases and their 136 structural counterparts were migrated.** Node labels and `.attrs` replace `@` bindings; original equations, conditions, attribute sharing, and metadata formulas are preserved. Original text and hashes are retained in [migration snapshots](../baselines/pre_syntax/) and the [migration record](../data/syntax_migration.json).

All **209 structural cases compile and their executable JSON validates**. The semantic baseline verifies **95 cases**, retaining **86/86 previously verified cases**. The deliberately incorrect subtraction-to-addition control is rejected.

## Step 2 Classification before testing

[synthesis_classification.json](../data/synthesis_classification.json) is frozen before inference runs. Classification uses declarations and graph structure, never proof outcomes. Metadata functions such as `gather(...)` are distinguished from HLO graph operations.

| Category | Catalog schemas | Concrete cases | Cases with RHS descriptors | Attribute-free cases |
|---|---:|---:|---:|---:|
| 1 Theorem fragment | 12 | 13 | 10 | 3 |
| 2 Supported beyond theorem | 107 | 106 | 33 | 73 |
| 3 Unsupported | 120 | 90 | 68 | 22 |

Category 1 follows the theorem's contraction, permutation, unary-sum, and separated-addition fragment, including read-once components and matching distinct input sets. The theorem's comparison domain assumes extents at least two and ideal arithmetic; this experiment adds no extent restriction to the original TEPL conditions. Accepted rules may therefore cover a broader domain. The current ordered complete path covers contraction/permutation; unary-sum and additive cases may use general verification. These measurements report coverage, not an exhaustive completeness evaluation.

Category 2 has supported local semantics but lies outside those theorem assumptions. Category 3 requires missing semantics or other unsupported capabilities and is excluded from auto attempts. Forty catalog schemas have no concrete case; their porting state is recorded separately, and they contribute no measured successes. Classification describes the scoped algebraic evaluation model, not a complete HLO importer.

## Step 3 Explicit full auto and partial auto

Full auto removes all RHS descriptor definitions. Partial auto retains explicit descriptors on some nodes and infers the rest. Both modes are measured for cases with multiple RHS descriptors. Partial subsets are tried in descending number of inferred nodes, stopping after the first successful cardinality. Partial success means at least one whole descriptor was inferred; field-level pins are not used.

| Category | Attribute cases | Explicit verified | Full auto | Partial auto including overlap | Partial only | Full or partial success |
|---|---:|---:|---:|---:|---:|---:|
| 1 Theorem fragment | 10 | 10 | 10 | 4 | 0 | 10/10 (100.0%) |
| 2 Supported beyond theorem | 33 | 22 | 26 | 7 | 0 | 26/33 (78.8%) |
| Total | 43 | 32 | 36 | 11 | 0 | 36/43 (83.7%) |

The **63/76 (82.9%) attribute-free cases** that verify are useful semantic verification coverage and are counted separately from attribute inference.

### Comparison on the verified explicit baseline

This comparison holds the availability of an explicit semantic proof fixed, isolating how much attribute-writing work can be automated on already verified cases. The overall table above retains all supported attribute-bearing cases, including those with an unproved explicit baseline.

| Category | Explicit-verified attribute cases | Auto successes on those same cases |
|---|---:|---:|
| 1 Theorem fragment | 10 | 10/10 (100.0%) |
| 2 Supported beyond theorem | 22 | 22/22 (100.0%) |
| Total | 32 | 32/32 (100.0%) |

### Practical interpretation

Theorem-fragment cases show how much the current implementation automates under the original XLA-case conditions. Beyond-theorem successes demonstrate practical reach of candidate construction plus independent verification. A failed proof is recorded as unproved; it does not establish that the rewrite is false or that no suitable attributes exist. All reported auto successes pass the same compiler acceptance checks used for ordinary rules.

### Cases improved by inference

- [catalog_broadcast_transpose](../../tepl/rules/full_auto/catalog_broadcast_transpose.tepl): 1 RHS descriptor(s) inferred; the original explicit formula was `not_proved`.
- [catalog_broadcast_transpose_broadcast](../../tepl/rules/full_auto/catalog_broadcast_transpose_broadcast.tepl): 1 RHS descriptor(s) inferred; the original explicit formula was `not_proved`.
- [catalog_broadcast_slice_broadcast_only](../../tepl/rules/full_auto/catalog_broadcast_slice_broadcast_only.tepl): 1 RHS descriptor(s) inferred; the original explicit formula was `not_proved`.
- [catalog_indexing_slice_singleton_stride](../../tepl/rules/full_auto/catalog_indexing_slice_singleton_stride.tepl): 1 RHS descriptor(s) inferred; the original explicit formula was `not_proved`.

### Successful inferred cases

| Case | Category | Full auto | Partial auto | Inferred descriptors in best variant | Original baseline scope |
|---|---:|---|---|---:|---|
| [transpose_compose](../../tepl/rules/full_auto/transpose_compose.tepl) | 1 | yes | — | 1/1 | One written operand order of the catalog equation, under the algebraic arithmetic model; source numerical policy is not executed. |
| [broadcast_add](../../tepl/rules/full_auto/broadcast_add.tepl) | 2 | yes | — | 1/1 | Only binary addition, equal base shapes, identical broadcast maps and added extents; excludes other elementwise functions and scalar expansion. |
| [broadcast_multiply](../../tepl/rules/full_auto/broadcast_multiply.tepl) | 2 | yes | — | 1/1 | Only binary multiplication, equal base shapes, identical maps and added extents. |
| [scalar_broadcast_reverse](../../tepl/rules/full_auto/scalar_broadcast_reverse.tepl) | 2 | yes | — | 1/1 | Only the reverse member of the parameterized indexing schema; input broadcast map is empty. |
| [scalar_broadcast_transpose](../../tepl/rules/full_auto/scalar_broadcast_transpose.tepl) | 2 | yes | — | 1/1 | Only a rank-two transpose with permutation [1,0] and scalar broadcast. |
| [reduce_negate](../../tepl/rules/full_auto/reduce_negate.tepl) | 2 | yes | — | 1/1 | Only sum with implicit zero initializer, matching the catalog arithmetic case. |
| [reduce_compose](../../tepl/rules/full_auto/reduce_compose.tepl) | 1 | yes | — | 1/1 | Rank-three sum, inner axis 1 then outer axis 1; symbolic extents. Does not cover arbitrary combiners/initializers. |
| [reduce_transpose](../../tepl/rules/full_auto/reduce_transpose.tepl) | 1 | yes | — | 1/1 | Rank-two transpose [1,0], sum over output axis 0, implicit zero initializer. |
| [reverse_compose_disjoint](../../tepl/rules/full_auto/reverse_compose_disjoint.tepl) | 2 | yes | — | 1/1 | Rank-two disjoint lists [0] and [1]; probes a case beyond identical-list cancellation. |
| [reverse_add_scalar](../../tepl/rules/full_auto/reverse_add_scalar.tepl) | 2 | yes | yes | 2/2 | Only addition and a scalar broadcast; tests the compound rule directly, not a two-rule sequence. |
| [slice_compose](../../tepl/rules/full_auto/slice_compose.tepl) | 2 | yes | — | 1/1 | Rank-one, positive-size, unit-stride slices with an explicit computed start. |
| [dot_associate](../../tepl/rules/full_auto/dot_associate.tepl) | 1 | yes | yes | 2/2 | Ordinary rank-two matrix products without batch axes; every attribute is reused explicitly. |
| [dot_transpose_output](../../tepl/rules/full_auto/dot_transpose_output.tepl) | 1 | yes | — | 1/1 | Rank-two matrix product, output transpose [1,0]; explicit swapped contraction attributes. |
| [dot_move_reverse](../../tepl/rules/full_auto/dot_move_reverse.tepl) | 2 | yes | yes | 2/2 | Rank-one inner product; reversal of its sole contracting axis; no free/batch axes. |
| [reduce_compose_leading](../../tepl/rules/full_auto/reduce_compose_leading.tepl) | 1 | yes | — | 1/1 | Rank-three sum with inner axis 0 then outer axis 0; implicit zero initializer. Compared with the nonleading-axis case. |
| [slice_compose_zero_starts](../../tepl/rules/full_auto/slice_compose_zero_starts.tepl) | 2 | yes | — | 1/1 | Rank-one, positive-size, unit-stride slices with both starts zero. Copies outer attributes; avoids computed constructors. |
| [broadcast_compose_append](../../tepl/rules/full_auto/broadcast_compose_append.tepl) | 2 | yes | — | 1/1 | Arbitrary input rank, broadcasts only append new dimensions; concatenates new extents explicitly. |
| [dot_swap_operands](../../tepl/rules/full_auto/dot_swap_operands.tepl) | 1 | yes | yes | 2/2 | Rank-two matrix product. Proves the identity for any captured X/Y; does not implement parameter-like policy screening. |
| [dot_no_contraction](../../tepl/rules/full_auto/dot_no_contraction.tepl) | 2 | yes | yes | 2/2 | Rank-one operands, no batches or contractions; same-type ideal arithmetic, no conversions. |
| [dot_reduce_product](../../tepl/rules/full_auto/dot_reduce_product.tepl) | 2 | yes | — | 1/1 | Rank-one inner product to multiply then sum; same-type ideal arithmetic, no widened accumulation. |
| [dot_reduce_batch](../../tepl/rules/full_auto/dot_reduce_batch.tepl) | 1 | yes | — | 1/1 | Rank-one batched product reduced over its sole batch axis; no remaining output axes. |
| [expr_reverse_compose_general](../../tepl/rules/full_auto/expr_reverse_compose_general.tepl) | 2 | yes | — | 1/1 | General symmetric difference computed with existing list builtins; no auto. |
| [expr_slice_compose_general](../../tepl/rules/full_auto/expr_slice_compose_general.tepl) | 2 | yes | — | 1/1 | Arbitrary-rank unit-stride composition; explicit metadata arithmetic and comprehensions. |
| [catalog_elementwise_compare_selected_mask](../../tepl/rules/full_auto/catalog_elementwise_compare_selected_mask.tepl) | 2 | yes | — | 1/1 | Scalar literal instance; source conditions retained in the catalog. Local HLO select requires the predicate and both branches to have the same shape. |
| [catalog_elementwise_compare_max](../../tepl/rules/full_auto/catalog_elementwise_compare_max.tepl) | 2 | yes | — | 1/1 | Written operand/opcode variant under the catalog conditions. |
| [catalog_elementwise_compare_min](../../tepl/rules/full_auto/catalog_elementwise_compare_min.tepl) | 2 | yes | — | 1/1 | Written operand/opcode variant under the catalog conditions. |
| [catalog_broadcast_multiply_constant_into_broadcast](../../tepl/rules/full_auto/catalog_broadcast_multiply_constant_into_broadcast.tepl) | 2 | yes | yes | 2/2 | Scalar factor C is a captured scalar; literal-only profitability does not affect the identity. |
| [catalog_broadcast_transpose](../../tepl/rules/full_auto/catalog_broadcast_transpose.tepl) | 2 | yes | — | 1/1 | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_broadcast_transpose_broadcast](../../tepl/rules/full_auto/catalog_broadcast_transpose_broadcast.tepl) | 2 | yes | — | 1/1 | General insertion-only broadcast/permutation equation; source sorted-map/layout policy remains in the catalog. |
| [catalog_broadcast_slice_broadcast_only](../../tepl/rules/full_auto/catalog_broadcast_slice_broadcast_only.tepl) | 2 | yes | — | 1/1 | Rank-one X maps to axis zero; only newly created axes are sliced. |
| [catalog_indexing_slice_singleton_stride](../../tepl/rules/full_auto/catalog_indexing_slice_singleton_stride.tepl) | 2 | yes | — | 1/1 | Arbitrary tensor shapes under the catalog arithmetic/applicability assumptions. |
| [catalog_dot_transpose_both_operands](../../tepl/rules/full_auto/catalog_dot_transpose_both_operands.tepl) | 1 | yes | yes | 2/2 | Rank-two matrix transposes, no batches. |
| [catalog_dot_move_broadcast](../../tepl/rules/full_auto/catalog_dot_move_broadcast.tepl) | 2 | yes | yes | 2/2 | One broadcast-created contraction axis; replacement dot is an outer product. |
| [catalog_dot_reduce_output_free](../../tepl/rules/full_auto/catalog_dot_reduce_output_free.tepl) | 1 | yes | yes | 2/2 | Reduce the left free axis of a rank-two dot; implicit sum initializer zero. |
| [catalog_dot_scalar_multipliers](../../tepl/rules/full_auto/catalog_dot_scalar_multipliers.tepl) | 2 | yes | yes | 2/2 | Scalar factor pulled from a rank-two f32 dot; source constant/profitability screening is outside the algebraic identity. |
| [catalog_reduction_broadcast_original_axes](../../tepl/rules/full_auto/catalog_reduction_broadcast_original_axes.tepl) | 2 | yes | yes | 2/2 | Sum specialization; rank-one base and one broadcast-created axis. |

## Shared axis reasoning improvement

The compiler reuses coordinate-sequence expansion in set-membership reasoning. Registered selections and permutations retain their original coordinate witnesses. This historical comparison ends before the general inference changes below.

The comparison uses unchanged equations, applicability conditions, classifications and acceptance criteria. No operation names, rule names or XLA case IDs participate in these compiler mechanisms.

[Pre-change measurements](../baselines/pre_axis_views/) preserve the previous compiler results.

| Measurement | Before | After |
|---|---:|---:|
| Explicit semantic proofs | 86 | 89 |
| Unique auto successes | 23/43 (53.5%) | 26/43 (60.5%) |
| Theorem-fragment auto successes | 7/10 | 10/10 |
| Previous semantic proofs retained | — | 86/86 |
| Previous auto successes retained | — | 23/23 |

Newly inferred cases:

- [reduce_compose](../../tepl/rules/full_auto/reduce_compose.tepl): full_auto.
- [reduce_compose_leading](../../tepl/rules/full_auto/reduce_compose_leading.tepl): full_auto.
- [catalog_dot_transpose_both_operands](../../tepl/rules/full_auto/catalog_dot_transpose_both_operands.tepl): full_auto.

## General inference improvement

Scalar parameter correspondence reuses unambiguous parameters from normalized source semantics. Constructors are checked against declared scalar attribute types. Coordinate correspondence combines additive dependencies and propagates whole-operand consumer interfaces. Metadata normalization merges adjacent ordered ranges, reuses selected element facts, and compares symbolic sequences by ordered blocks; arbitrary permutations cannot establish order. Coordinate complements preserve their extent maps. Reduction reflection changes only bound coordinates in equal finite domains, substituting the entire summand under algebraic addition.

The comparison uses unchanged equations, applicability conditions, classifications and acceptance criteria. No operation names, rule names or XLA case IDs participate in these compiler mechanisms.

[Pre-change measurements](../baselines/pre_general_inference/) preserve the previous compiler results.

| Measurement | Before | After |
|---|---:|---:|
| Explicit semantic proofs | 89 | 95 |
| Unique auto successes | 26/43 (60.5%) | 36/43 (83.7%) |
| Theorem-fragment auto successes | 10/10 | 10/10 |
| Previous semantic proofs retained | — | 89/89 |
| Previous auto successes retained | — | 26/26 |

Newly inferred cases:

- [reverse_add_scalar](../../tepl/rules/full_auto/reverse_add_scalar.tepl): full_auto.
- [dot_move_reverse](../../tepl/rules/full_auto/dot_move_reverse.tepl): full_auto.
- [broadcast_compose_append](../../tepl/rules/full_auto/broadcast_compose_append.tepl): full_auto.
- [dot_no_contraction](../../tepl/rules/full_auto/dot_no_contraction.tepl): full_auto.
- [catalog_elementwise_compare_selected_mask](../../tepl/rules/full_auto/catalog_elementwise_compare_selected_mask.tepl): full_auto.
- [catalog_elementwise_compare_max](../../tepl/rules/full_auto/catalog_elementwise_compare_max.tepl): full_auto.
- [catalog_elementwise_compare_min](../../tepl/rules/full_auto/catalog_elementwise_compare_min.tepl): full_auto.
- [catalog_broadcast_multiply_constant_into_broadcast](../../tepl/rules/full_auto/catalog_broadcast_multiply_constant_into_broadcast.tepl): full_auto.
- [catalog_broadcast_slice_broadcast_only](../../tepl/rules/full_auto/catalog_broadcast_slice_broadcast_only.tepl): full_auto.
- [catalog_dot_scalar_multipliers](../../tepl/rules/full_auto/catalog_dot_scalar_multipliers.tepl): full_auto.

## Reproduction and artifacts

```sh
bazel build //:tepl
python3 labs/xla_rewrite/coverage/scripts/migrate_syntax.py --check
python3 labs/xla_rewrite/coverage/scripts/synthesis_cases.py
python3 labs/xla_rewrite/coverage/scripts/run_checks.py
python3 labs/xla_rewrite/coverage/scripts/expression_coverage.py --run
python3 labs/xla_rewrite/coverage/scripts/run_synthesis.py
python3 labs/xla_rewrite/coverage/scripts/synthesis_report.py
python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_*.py'
```

[synthesis.json](../results/synthesis.json) contains every attempt, compiler diagnostics, constructor counts, proof output, artifact hashes, and source/input hashes. [synthesis_cases.md](synthesis_cases.md) lists all concrete outcomes, including unsupported and attribute-free cases. Generated [full-auto](../../tepl/rules/full_auto/) and [partial-auto](../../tepl/rules/partial_auto/) files preserve the source conditions.

- Run UTC: `2026-10-26T13:58:33.124070+00:00`.
- Compiler revision: `054ab237707c46540663b55ae698ef6b16d0399d`; working-tree source hashes are recorded.
- Timeout: 30 seconds per compiler command; 4 independent case workers.
- Arithmetic model: ideal algebraic semantics. The measurements do not claim IEEE floating-point or full backend equivalence.
