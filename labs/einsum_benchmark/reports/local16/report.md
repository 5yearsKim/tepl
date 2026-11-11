# 16-input contraction rewrites: coverage and inferred axes

One connected 16-input subnetwork is selected per original benchmark instance (seed 42). Original shapes and dtypes and cut-boundary indices are preserved. Both supplied paths are restricted to the selection. Each path has 15 binary contractions, plus any reductions and a final permutation.

| Category | Sampled | Represented | Verified / nonidentical tasks | Inferred axis-list fields | Inferred axis positions | Positions/rule median [min, max] |
|---|---:|---:|---:|---:|---:|---:|
| Graphical models | 10 | 10 | 18/20 | 1,118 | 898 | 48 [41, 70] |
| Tensor network language models | 25 | 14 | 28/28 | 1,708 | 1,432 | 51 [47, 56] |
| Model counting | 50 | 50 | 100/100 | 6,108 | 8,852 | 78.5 [50, 211] |
| Quantum computing | 32 | 0 | — | 0 | 0 | — |
| Random | 16 | 9 | 18/18 | 1,112 | 904 | 43 [34, 70] |
| Structural | 21 | 21 | 36/36 | 2,196 | 2,034 | 57 [32, 80] |
| Weighted model counting | 14 | 13 | 26/26 | 1,586 | 1,808 | 66 [43, 100] |
| Total | 168 | 117 | 226/228 | 13,828 | 15,928 | 63 [32, 211] |

## Metric definitions

An **inferred axis-list field** is one omitted RHS attribute completed by TEPL: each `dot_general` has four fields, each `reduce_sum` one, and each `transpose` one. An empty inferred list counts as one completed field and zero axis positions.

An **inferred axis position** is one integer entry in such a field. For example, `lhs_contracting_dimensions=[1,3]` and `rhs_contracting_dimensions=[0,2]` contribute two fields and four positions. A rank-4 permutation contributes one field and four positions. The same logical axis appearing in multiple fields/nodes counts multiple times. These are attribute-entry occurrences, not unique logical axes or tensor elements.

Counts are measured from the actual rewritten concrete graph after independent verification, not from the hidden target witness. Each direction is counted separately. The per-rule CSV also includes maximum/mean input rank, input-port count, distinct logical indices, indices shared by at least three inputs (including output indices), and RHS depth.

## Inferred positions by role

| Category | Contracting | Batching | Reduction | Permutation |
|---|---:|---:|---:|---:|
| Graphical models | 164 | 448 | 20 | 266 |
| Tensor network language models | 780 | 276 | 0 | 376 |
| Model counting | 272 | 7,092 | 10 | 1,478 |
| Quantum computing | 0 | 0 | 0 | 0 |
| Random | 496 | 96 | 14 | 298 |
| Structural | 1,420 | 0 | 0 | 614 |
| Weighted model counting | 84 | 1,348 | 0 | 376 |
| Total | 3,216 | 9,260 | 44 | 3,408 |

## Coverage and protocol

117/168 local cases are represented. 6 structurally identical directed pairs are excluded. 2 nonidentical directions are excluded for singleton extents and remain in the task denominator. Unsupported dtypes and trace/diagonal handling are adapter exclusions, not proofs of impossibility in arbitrary TEPL dialects.

Guard policy `domain`, algebraic arithmetic, 10 s per subprocess, and 4 parallel cases. Rule declarations retain rank, dtype and shared extents; written guards require extents >=2. Actual LHS descriptors and generated feasibility determine applicability. No work budget was increased. Any valid once-applied result at the intentional step limit must still pass independent checks.

CLI compile/rewrite times are diagnostic process times, not isolated synthesis or constructor latency. This experiment does not claim full-network coverage, optimal local paths, numerical execution speedups, or arbitrary-rank generalization.

[Per-rule axis counts](axis_counts.csv), [category totals](categories.csv), [summary](summary.json), [raw results](results.json), [run provenance](provenance.json). Full generated artifacts: `labs/einsum_benchmark/results/local16`.
