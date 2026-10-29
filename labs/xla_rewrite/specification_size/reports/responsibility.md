# XLA and TEPL specification size by responsibility

The primary table uses **25 common cases** with reviewed XLA branch mappings and successful proof, compilation, and JSON validation for the original explicit, omitted-field, and compact inferred TEPL variants. XLA source is measured from the pinned snapshot; XLA itself is not compiled or executed in this experiment.

| Responsibility | XLA | Explicit TEPL | Compact inferred TEPL | Saved from XLA |
|---|---:|---:|---:|---:|
| Matching | 1,259 | 324 | 310 | 949 |
| Legality checks | 1,055 | 985 | 985 | 70 |
| Attribute reasoning and syntax | 4,583 | 554 | 191 | 4,392 |
| Replacement construction | 1,538 | 236 | 218 | 1,320 |
| Graph-management code | 753 | 125 | 125 | 628 |
| **Total** | **9,188** | **2,224** | **1,829** | **7,359** |

Compact inferred TEPL has **7,359 fewer lexical tokens (80.1%)** than the selected XLA transformation excerpts and **395 fewer (17.8%)** than the original explicit TEPL rules.

These are descriptive source-footprint differences. XLA branches often handle more ranks, operand orders, and cases than the TEPL specializations. The table does not establish equivalent support breadth or assign the entire reduction causally to the language. Source policy and logging are excluded rather than credited as savings; library implementations and shared dialect setup are also outside this rule/excerpt table. The five responsibilities partition every included token exactly once.

## What omission and compaction contribute

All rows below use the same primary cohort. The original idiomatic baseline keeps descriptor copies and spreads. The expanded baseline controls their expansion cost.

| Representation | Tokens |
|---|---:|
| Original explicit TEPL | 2,224 |
| Expanded explicit fields | 2,547 |
| Explicit field-level auto | 2,331 |
| Omitted axis fields; labels and assignments retained | 2,107 |
| Omitted fields plus unused-label/empty-block cleanup | 1,829 |

Expanded expressions → `auto` saves **216** tokens. Omitting inferable fields saves another **224**. Removing unused labels, empty assignments, and empty derive blocks saves another **278**. Expansion itself costs **323** tokens relative to the idiomatic original; it must be subtracted when claiming net practical savings.

## Coverage and exclusions

All **32** preregistered axis cases were attempted. Omitted-field inference succeeds on **30**; compact inference succeeds on **30**. **26** have accepted original explicit counterparts. The inherited negative control fails semantic verification.

| Case | Explicit / omitted / compact accepted | In main table | Audit note |
|---|---|---|---|
| transpose_compose | yes / yes / yes | yes | Operand binding and nested-transpose branch only; omit identity/scatter/bitcast and dot branches. |
| broadcast_add | yes / yes / yes | yes | Shared elementwise sinking algorithm includes scalar expansion and generic elementwise/user handling beyond the two TEPL binary/equal-map specializations. Configuration dispatch is policy. |
| broadcast_multiply | yes / yes / yes | yes | Shared elementwise sinking algorithm includes scalar expansion and generic elementwise/user handling beyond the two TEPL binary/equal-map specializations. Configuration dispatch is policy. |
| scalar_broadcast_reverse | yes / yes / yes | yes | Common broadcast operand and descriptor capture, counted once. Local permutation predicate; includes the additional sort case. Subset predicate is short-circuited for the selected reverse/transpose cases and is not expanded. Scalar broadcast through a permutation: retain user traversal and mutation; no iota, broadcast-composition, or unrelated reshape branches. |
| scalar_broadcast_transpose | yes / yes / yes | yes | Common broadcast operand and descriptor capture, counted once. Local permutation predicate; includes the additional sort case. Subset predicate is short-circuited for the selected reverse/transpose cases and is not expanded. Scalar broadcast through a permutation: retain user traversal and mutation; no iota, broadcast-composition, or unrelated reshape branches. |
| reduce_negate | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. Local scalar-zero predicate and its effective-scalar predicate; scalar type switching is retained. General primitive/literal library routines are not expanded. Only negate/sum commutation. Floating-type and zero-init checks remain legality; scalar/combiner assumptions are built into the TEPL sum dialect. |
| reduce_compose | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. MergeReduces helper plus actual matching/combiner-equivalence caller. Both retained TEPL reduction-composition cases share this code. |
| reduce_transpose | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. Reduction-through-transpose branch includes general output-transpose correction; TEPL case is rank two/vector output. Fanout/unconditional-rewrite option is policy. |
| reverse_compose_disjoint | yes / yes / yes | yes | Reverse operand capture; layout sensitivity is policy. Trivial-dimension elimination is another rewrite and omitted. Nested-reverse symmetric difference, including empty-result deletion; shared by fixed and general TEPL cases. |
| reverse_add_scalar | yes / yes / yes | yes | Reverse operand capture; layout sensitivity is policy. Trivial-dimension elimination is another rewrite and omitted. Generic binary elementwise commutation with a scalar broadcast, both operand orders. TEPL selects addition with the scalar on one side; side-effect/opcode restrictions are retained. |
| dot_associate | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Complete local ConstructToDotMaps and ConstructFromDotMaps axis-map algorithms, once globally, including both input maps and inverse maps. General nested-dot reassociation, including batch/contract maps and optional output transpose; source flop heuristic is excluded. TEPL is ordinary matrix multiplication only. No unrelated HandleDot branches included. |
| dot_transpose_output | yes / yes / yes | yes | Canonical transpose(dot) branch, appropriate for the rank-two TEPL case; exclude the separate arbitrary noncanonical-dot branch, and parameter/precision preference policy. Parameter-like profitability/layout preference predicate, kept in excluded-policy accounting. Precision-configuration propagation, outside the ideal-algebraic TEPL profile; excluded rather than credited as a saving. |
| dot_move_reverse | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Shared reverse/broadcast operand selection and dimension maps. Selector also mentions pad, but the pad algorithm is excluded. Reverse-specific map partition and construction; shape-ratio threshold and threshold multiplier are source policy. Common new-dot construction and actual visitor replacement. Flop threshold and logging excluded; pad-specific source body is not selected. |
| reduce_compose_leading | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. MergeReduces helper plus actual matching/combiner-equivalence caller. Both retained TEPL reduction-composition cases share this code. |
| broadcast_compose_append | yes / yes / yes | yes | Common broadcast operand and descriptor capture, counted once. Nested broadcast descriptor composition and replacement. |
| dot_swap_operands | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Operand and output-permutation swap helper and call site. Parameter placement gate and precision policy excluded; general batching permutation retained. Parameter-like profitability/layout preference predicate, kept in excluded-policy accounting. |
| dot_no_contraction | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Local batch-major/contracting-minor normalization, shared by no-contraction and strength-reduction helpers; general transpose path retained. No-contracting-dimension multiply rewrite and dispatch. General broadcasting retained; preferred-type conversions and feature/precision gates excluded. |
| dot_reduce_product | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Local batch-major/contracting-minor normalization, shared by no-contraction and strength-reduction helpers; general transpose path retained. Complete local predicate used by dot-to-reduce dispatch. Product/reduction construction, rank-general outer-dimension broadcasting, and caller; scalar f32 dot specialization in TEPL. Preferred-type/precision and CPU strength-reduction policy excluded. Local zero-init sum-reduction builder. General scalar-add computation construction is treated as a primitive library dependency, like TEPL sum semantics. |
| dot_reduce_batch | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. Convert reduced batch dimensions to contracting dimensions, including optional remaining reduction; TEPL fixes ranks and reduces one batch axis. Feature flags excluded. |
| expr_reverse_compose_general | yes / yes / yes | yes | Reverse operand capture; layout sensitivity is policy. Trivial-dimension elimination is another rewrite and omitted. Nested-reverse symmetric difference, including empty-result deletion; shared by fixed and general TEPL cases. |
| catalog_broadcast_divide_broadcast | no / no / no | no | Original explicit counterpart is unproved in this cohort; excluded from the three-way common-success table. No XLA excerpt size is claimed. |
| catalog_broadcast_multiply_constant_into_broadcast | yes / yes / yes | yes | Only multiply-constant-into-broadcast rewrite, including original literal/nonconstant matching. TEPL captures a scalar C and proves the more general algebraic equation; numerical fast-math gating is excluded, not counted as a saving. |
| catalog_broadcast_transpose | no / yes / yes | no | Original explicit counterpart is unproved in this cohort; excluded from the three-way common-success table. No XLA excerpt size is claimed. |
| catalog_broadcast_inserted_unit_dims | no / yes / yes | no | Original explicit counterpart is unproved in this cohort; excluded from the three-way common-success table. No XLA excerpt size is claimed. |
| catalog_broadcast_drop_unit_dims | no / yes / yes | no | Original explicit counterpart is unproved in this cohort; excluded from the three-way common-success table. No XLA excerpt size is claimed. |
| catalog_broadcast_transpose_broadcast | no / no / no | no | Original explicit counterpart is unproved in this cohort; excluded from the three-way common-success table. No XLA excerpt size is claimed. |
| catalog_indexing_reverse_inserted_unit_dims | no / yes / yes | no | Original explicit counterpart is unproved in this cohort; excluded from the three-way common-success table. No XLA excerpt size is claimed. |
| catalog_dot_transpose_both_operands | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Both-input transpose removal helper and dispatch, retaining general canonical batch handling and both operand orders. Rank-two TEPL specialization; precision helper separately excluded. Precision-configuration propagation, outside the ideal-algebraic TEPL profile; excluded rather than credited as a saving. |
| catalog_dot_move_broadcast | yes / yes / yes | yes | Shared HandleDot entry/capture only, not other dot rewrites or sharding checks. Layout policy excluded. Shared reverse/broadcast operand selection and dimension maps. Selector also mentions pad, but the pad algorithm is excluded. Broadcast-specific contraction map rewriting and reduction construction; optional residual broadcast retained. Threshold adjustment is policy. Common new-dot construction and actual visitor replacement. Flop threshold and logging excluded; pad-specific source body is not selected. |
| catalog_dot_reduce_output_free | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. Reduce-free-output-dimension into dot operands, including complete local dot maps via shared helper; original flop/profitability gate excluded. TEPL selects a rank-two free dimension. Complete local ConstructToDotMaps and ConstructFromDotMaps axis-map algorithms, once globally, including both input maps and inverse maps. Local scalar-zero predicate and its effective-scalar predicate; scalar type switching is retained. General primitive/literal library routines are not expanded. |
| catalog_dot_scalar_multipliers | yes / yes / yes | no | Source-mapping mismatch: ScalarMultiplyReduction (704–853) is called at 11039 from HandleConvolution; there is no call in HandleDot in this pinned snapshot. Its valid TEPL dot identity is retained but cannot support a corresponding XLA dot-excerpt count. |
| catalog_reduction_broadcast_original_axes | yes / yes / yes | yes | Reduce operand, init, output shape, dimensions and combiner capture. Layout-sensitive dispatch is separate policy; unrelated tuple/trivial-reduce rewrites omitted. Only reduction of original base-tensor axes through broadcast: mapping legality, surviving axes, and replacement. Replication-count reduction is a different branch and excluded. |

## XLA attribution and limits

File/line ranges are deduplicated before counting. The selected blocks also contain **1,147 policy tokens** and **107 logging tokens**, recorded separately and excluded from the primary total. Annotations use each line's main responsibility; control syntax belongs to the block it controls. Different reasonable assignments can change individual rows, though they do not change the included-source total.

Transformation-specific local helpers are included once. General XLA HLO/matcher/shape APIs and TEPL compiler/runtime/builtins are library dependencies, not expanded into these source counts. Whole shared TEPL dialect sizes appear in the [field-auto report](specification_size.md#shared-setup-and-evidence). An independent boundary/category review is recommended before publication.

The algebraic policy does not reproduce all IEEE behavior or backend profitability policies. Successful inference constructs a valid witness, which can differ from the handwritten or intended optimization. No claim about runtime speed or authoring effort follows from token counts.

## Evidence

[Protocol](../responsibility_protocol.md), [annotated source listing](xla_excerpts.txt), [span ledger](../data/responsibility_spans.json), [measurements](../results/responsibility.json), and [LaTeX table](responsibility.tex).

Run UTC: `2026-10-29T19:27:35.397712+00:00`. Compiler SHA-256: `96258055f9087add6e7857b0767c69e6c5191a303feddd54d7c011ddbbc57912`. 4 workers; 30 seconds per compiler command. Source, tool, protocol, and baseline hashes are included in the measurement JSON.
