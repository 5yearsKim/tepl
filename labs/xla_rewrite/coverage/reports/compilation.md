# XLA Rewrite Coverage in TEPL

This report evaluates whether XLA's algebraic rewrites can be written using TEPL dialect declarations, graph patterns, conditions and explicit RHS attributes. Automatic value-equivalence checking is reported separately.

## Result

The language assessment covers **239 catalog schemas**:

- **192** have algebraic structures and metadata constructions expressible through ordinary rules and dialect declarations.
- **33** require analysis facts or tensor-literal handling not currently exposed to rules. These are extension requirements, not supported general schemas. Some concrete cases can instead use readable constant attributes, literal patterns or dtype conditions; see the [host-dependency audit](host_dependency_audit.md).
- **12** are expressible as concrete arities, extents or single-result cases; their fully general schemas need additional adaptation.
- **2** reuse a separately existing graph value and need graph lookup or nonlocal matching support that is not currently available.

These are manual expressibility assessments. They identify which language mechanisms can represent a schema. Concrete implementations are measured below.

The explicit baseline now includes a designated rule for each of the **192 originally expressible schemas**. **192/192 designated rules compile**. Each rule has an explicit catalog ID, RHS metadata and scope. Some implementations select a fixed rank, opcode or padding domain; their compilation establishes those written cases.

Including retained pilot variants, compilation accepted **209 of 209 explicit cases**, spanning **199 catalog schemas**. Each case supplies its RHS attributes. The full semantic sweep proved **94 of 209 cases**, covering **87 catalog schemas**; see [automatic verification](verification.md).

Both experiments use the self-contained HLO project configured by [tepl.toml](../../tepl/tepl.toml). Compilation uses [HLO declarations without value semantics](../../tepl/dialects/hlo_structural.tepl); verification uses the same rule bodies with the local [semantic HLO profile](../../tepl/dialects/hlo.tepl). Every compilation case has a semantic counterpart; missing semantics and unproved cases are reported separately. Imports resolve from the project root.

## What counts as coverage

A missing built-in operation or semantic lowering does not prevent an algebraic rule from being expressed. For example, a dialect can declare divide, exp, reshape, pad or dynamic slice; a rule can match and rebuild those operations without the semantic normalizer evaluating them.

The evaluation follows the catalog's algebraic scope. XLA visitor plumbing, graph-mutation APIs, backend profitability and target-specific numerical policy do not reduce language coverage. Conditions that determine when the algebraic equation applies remain part of the rule specification. Host calls and host `fn` parameters are unsupported; source facts or literal handling that need those capabilities are recorded as extension requirements, not assumed to be available.

Three outcomes are recorded independently:

1. **Expressible:** the rule, its metadata construction and relevant source requirements have a language-level representation.
2. **Compiled:** a concrete explicit rule passes `tepl compile` and its executable JSON passes `tepl validate`. Some cases specialize ranks, literals or arities; their scope is recorded in the [case manifest](../data/compilation_cases.json). Compilation checks syntax, types and construction, and does not establish that every source condition has been ported.
3. **Verified:** the optional semantic checking path additionally proves the written value/coordinate equivalence under its arithmetic policy.

## Coverage by category

| Category | Catalog | Rule + dialect | Needs extension | Concrete cases / partial | Nonlocal | Baseline compiled / targeted | Compiled cases | Schemas with a compiled case | Schemas with a verified case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elementwise | 99 | 77 | 22 | 0 | 0 | 77/77 | 84 | 83 | 62 |
| indexing | 60 | 43 | 10 | 7 | 0 | 43/43 | 49 | 44 | 4 |
| broadcast | 20 | 18 | 1 | 1 | 0 | 18/18 | 20 | 18 | 4 |
| transpose-reshape | 9 | 8 | 0 | 1 | 0 | 8/8 | 8 | 8 | 2 |
| dot | 18 | 17 | 0 | 1 | 0 | 17/17 | 17 | 17 | 11 |
| reduction | 26 | 22 | 0 | 2 | 2 | 22/22 | 24 | 22 | 4 |
| convolution | 7 | 7 | 0 | 0 | 0 | 7/7 | 7 | 7 | 0 |

## Examples demonstrated by compilation

The pilot includes polynomial rules and explicit dot attributes, division/remainder identities, logical operations, exp/log, min/max, complex projections, reshape composition, concatenate, dynamic update identities, padding, scatter/gather, reduction windows, and convolution attribute transformations.

It also includes general metadata constructions for reverse symmetric difference, unit-stride slice composition, reflected slice starts and lifted reduction dimensions. These use existing lists, indexing, comprehensions and descriptor constructors.

Per-case scope, catalog IDs and source files are recorded in the [compilation manifest](../data/compilation_cases.json); [compilation.json](../results/compilation.json) records every outcome.

## Execution checks

[test_rule_execution.py](../tests/test_rule_execution.py) runs eight representative compiled rewrites and compares their inputs and outputs using independent dense array meanings. The cases cover reverse composition, slice composition, signed padding converted to a crop, symbolic replication counts, pointwise convolution to dot, convolution operand swapping with reversal, a splat exponent matched through attributes, and unsigned absolute value.

The tests require the selected rule to fire. They then compile the exported replacement graph and compare its shape and values with the original graph. These concrete checks supplement the compilation measurements; universal proofs remain in the separate verification report.

### Dialect and implementation scope

Dynamic slices/updates use scalar start operands; fixed-rank cases spell out those operands. Reduction regions are forwarded as handles. The extended convolution schema exposes dimension numbers, grouping, window parameters, precision and output shape.

Padding arithmetic uses a nonnegative-edge padding declaration where it combines with index metadata. A separate signed-pad rule demonstrates negative padding. Window and extended convolution cases use nonnegative edge metadata. Constant construction supplies payload, element type and shape explicitly.

The [case manifest](../data/compilation_cases.json) records each baseline's scope. Fixed-rank/opcode/padding instances demonstrate a catalog schema without asserting that every conditional variant is implemented.

## Evidence and reproduction

[language_assessments.json](../data/language_assessments.json) retains all per-schema assessments and integration requirements. Representative gaps include nonnegativity analysis, opaque constant payloads, variable arities and nonlocal graph-value reuse; see the [host-dependency audit](host_dependency_audit.md).

TEPL revision: `17d48f05087b540cae87648fed3b43d9f14c2166`. XLA revision: `e02127ba9c9c48912b0e6776fe5bfe3843d81042`. [compilation.json](../results/compilation.json) records input, dialect, catalog and runner hashes. Follow the [reproduction commands](../README.md#reproduce) to rerun the evaluation.
