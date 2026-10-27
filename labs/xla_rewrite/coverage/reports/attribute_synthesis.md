# XLA Attribute Synthesis Evaluation

TEPL automatically infers RHS attributes for **32/32 (100.0%)** of the attribute-bearing cases whose explicit baseline already verifies. This includes **10/10 verified theorem-fragment cases**. Across all supported attribute-bearing cases, including cases with unproved explicit baselines, auto succeeds on **36/43 (83.7%)**, spanning **30 catalog schemas**.

The experiment infers RHS attributes for **36 cases** in total. Full auto succeeds on **36 cases**; partial auto succeeds on **11**, including **0 partial-only successes**. Overlapping full and partial results count once.

Combining verified explicit rules, inferred rules, and attribute-free identities gives **98/119 (82.4%) supported cases** with a semantic proof. **4 cases gain a verified implementation through inference even though their explicit metadata formulas remain unproved.**

## Experiment setup

The evaluation follows three steps: migrate the existing corpus, freeze theorem-based categories, then measure explicit, full-auto, and partial-auto variants. It uses the ideal `algebraic` policy. A successful auto variant has an explicit semantic proof, the expected number of inferred descriptors, successful executable compilation, and validated JSON. Equations, input declarations, and written conditions are preserved. No suggested restrictions are added.

The corpus contains **239 catalog schemas**, **209 concrete cases representing 199 schemas**, and one negative control. Concrete cases retain the scopes recorded in the existing manifest: some specialize ranks, arithmetic operations, layouts, or constant representations. Schema coverage means at least one measured case, not every XLA variant.

The [syntax migration record](../data/syntax_migration.json) and [snapshots](../baselines/pre_syntax/) preserve original text and hashes.

## Classification

[synthesis_classification.json](../data/synthesis_classification.json) is frozen before inference runs. Classification uses declarations and graph structure, never proof outcomes. Metadata functions such as `gather(...)` are distinguished from HLO graph operations.

| Category | Catalog schemas | Concrete cases | Cases with RHS descriptors | Attribute-free cases |
|---|---:|---:|---:|---:|
| 1 Theorem fragment | 12 | 13 | 10 | 3 |
| 2 Supported beyond theorem | 107 | 106 | 33 | 73 |
| 3 Unsupported | 120 | 90 | 68 | 22 |

Category 1 follows the theorem's contraction, permutation, unary-sum, and separated-addition fragment, including read-once components and matching distinct input sets. The theorem's comparison domain assumes extents at least two and ideal arithmetic; this experiment adds no extent restriction to the original TEPL conditions. Accepted rules may therefore cover a broader domain. The current ordered complete path covers contraction/permutation; unary-sum and additive cases may use general verification. These measurements report coverage, not an exhaustive completeness evaluation.

Category 2 has supported local semantics but lies outside those theorem assumptions. Category 3 requires missing semantics or other unsupported capabilities and is excluded from auto attempts. Forty catalog schemas have no concrete case; their porting state is recorded separately, and they contribute no measured successes. Classification describes the scoped algebraic evaluation model, not a complete HLO importer.

## Explicit, full-auto and partial-auto results

Full auto removes all RHS descriptor definitions. Partial auto retains explicit descriptors on some nodes and infers the rest. Both modes are measured for cases with multiple RHS descriptors. Partial subsets are tried in descending number of inferred nodes, stopping after the first successful cardinality. Partial success means at least one whole descriptor was inferred; field-level pins are not used.

| Category | Attribute cases | Explicit verified | Full auto | Partial auto including overlap | Partial only | Full or partial success |
|---|---:|---:|---:|---:|---:|---:|
| 1 Theorem fragment | 10 | 10 | 10 | 4 | 0 | 10/10 (100.0%) |
| 2 Supported beyond theorem | 33 | 22 | 26 | 7 | 0 | 26/33 (78.8%) |
| Total | 43 | 32 | 36 | 11 | 0 | 36/43 (83.7%) |

The **62/76 (81.6%) attribute-free cases** that verify are useful semantic verification coverage and are counted separately from attribute inference.

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

## Compiler improvements

Shared axis reasoning preserves coordinate witnesses through selections and permutations. General inference adds scalar-parameter correspondence, coordinate propagation, ordered metadata normalization and reduction reflection. These mechanisms use no operation names, rule names or XLA case IDs.

Equations, conditions, classifications and acceptance criteria are unchanged. The table compares each stage with its immediate predecessor.

| Stage | Explicit proofs | Auto successes | Theorem-fragment auto | Previous proofs retained | Previous auto retained |
|---|---:|---:|---:|---:|---:|
| [Before axis reasoning](../baselines/pre_axis_views/) | 86 | 23/43 (53.5%) | 7/10 | — | — |
| [After axis reasoning](../baselines/pre_general_inference/) | 89 | 26/43 (60.5%) | 10/10 | 86/86 | 23/23 |
| After general inference | 94 | 36/43 (83.7%) | 10/10 | 88/89 | 26/26 |

Cases losing acceptance between stages: catalog_indexing_reverse_trivial.

## Reproduction and artifacts

Follow the [reproduction commands](../README.md#reproduce) to rerun the evaluation.

[synthesis.json](../results/synthesis.json) contains every attempt, compiler diagnostics, constructor counts, proof output, artifact hashes, and source/input hashes. [synthesis_cases.md](synthesis_cases.md) lists all concrete outcomes, including unsupported and attribute-free cases. Generated [full-auto](../../tepl/rules/full_auto/) and [partial-auto](../../tepl/rules/partial_auto/) files preserve the source conditions.

- Run UTC: `2026-10-26T20:39:44.091717+00:00`.
- Compiler revision: `17d48f05087b540cae87648fed3b43d9f14c2166`; working-tree source hashes are recorded.
- Timeout: 30 seconds per compiler command; 4 independent case workers.
- Arithmetic model: ideal algebraic semantics. The measurements do not claim IEEE floating-point or full backend equivalence.
