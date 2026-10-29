# Responsibility and omission experiment

This study extends the [matched-field protocol](protocol.md) with implicit axis
fields, compact source, and reviewed XLA branch excerpts. Its primary metric is
the number of handwritten lexical tokens under the same scanner.

## Representations and acceptance

Attempt all 32 fixed axis-inference cases. Retain the original idiomatic explicit
rule, expanded explicit fields, and expanded field-level `auto` forms from a
fresh matched-field run using the same compiler. Add two representations:

1. **Implicit fields:** omit required axis fields whose value is `auto` and
   whose schema has no default. Retain labels and empty assignments. An explicit
   `auto` overriding a default must remain explicit.
2. **Compact inferred:** also remove empty attribute assignments, empty `derive`
   blocks, and operation labels with no remaining reference in the rule.

Spreads, explicit values, defaults, and optional values keep their precedence.
Supplied non-axis fields and fixed normalization fields remain supplied. Do not
remove guards, change tensor constraints, introduce assumptions, select a new
operation, or rewrite an attribute formula. Assert unchanged graph structure and
guard tokens. Labels can disappear only when no other rule token refers to them.

Both new forms must independently prove with `additional-guard: true`, compile,
validate, contain no unresolved holes, and report the same requested automatic
descriptor count. The common primary cohort requires all five TEPL forms to
pass and a reviewed corresponding XLA mapping. Keep every failure and source
mapping exclusion in the report. Reuse the matched run's negative control only
when its source dependencies and compiler hash match this run.

The explicit column in the responsibility table is the original idiomatic
source, not the expanded control. All ablation rows use the same primary cohort.
Report expansion cost as well as savings from expression inference, omitted
fields, and unused-label/empty-block cleanup.

## Source audit and five responsibilities

`data/responsibility_spans.json` is a hand-reviewed input ledger. Each block
selects exact inclusive source line spans in the pinned XLA snapshot, with one
dominant responsibility per span. Case mappings identify shared blocks and
record concrete TEPL scope restrictions. Function declarations, local helpers,
caller gates, branch handling, and excluded code are selected explicitly;
whole handler ranges from the catalog are not used as automatic excerpts.

| Responsibility | XLA | TEPL |
|---|---|---|
| Matching | Input/descriptor capture, opcode and structural matching, user search | Whole LHS graph pattern |
| Legality checks | Mathematical/shape/type applicability checks and local predicates | Tensor constraints and complete `where` block |
| Attribute reasoning and syntax | Dimension/shape calculations, copies, permutations, and local map algorithms | RHS inline descriptors and complete `derive` block |
| Replacement construction | Replacement operand choice and HLO creation | RHS graph pattern excluding inline descriptors |
| Graph-management code | Entry/return plumbing, replacement APIs, mutation, change tracking, and derived metadata propagation | Rule declaration, arrow, and outer block syntax |

On a mixed-purpose C++ line, use the main responsibility and retain the source
line in the annotated listing. Control braces belong to their annotated block.
These choices are descriptive and can affect individual rows. They do not
measure a unique causal contribution of each compiler feature.

Backend/layout/profitability/precision policy and logging receive separate
labels and are excluded from the five-row total. Their cost cannot be credited
as a saving when TEPL does not reproduce their behavior. Unrelated rewrite
branches are not selected. Treat general HLO constructors, shape APIs, matchers,
and primitive library routines as dependencies, as with TEPL's compiler/runtime
and builtins. Include transformation-specific local map and normalization
algorithms once; the ledger explains those boundaries.

Tokenize the complete original C++ source before selecting token-start lines.
Thus comments crossing a selected range do not become code. Deduplicate each
source-file/line token globally across selected blocks and cases. Conflicting
category assignments to an overlapping source line are errors. Do not sum
per-case XLA sizes when they share helper bodies or generic rewrite code.
Every primary token must belong to exactly one of the five categories, and
the responsibility columns must sum to their whole-source totals.

## Interpretation and reproducibility

XLA branches are often rank-general or handle both operand orders, whereas
several TEPL cases specialize ranks, literals, types, or operation families.
Record those differences rather than implying that the table measures full
feature parity. The table is a source-footprint comparison for selected
XLA-derived cases, not a causal estimate of DSL benefit independent of scope.
Do not claim full XLA numerical/backend equivalence or intended optimization
quality from TEPL's ideal-algebraic proof.

The audit is by one analyst and is not an independent second review. Publish the
span ledger and annotated source listing so the boundaries and categories can
be challenged and reproduced. One mapped helper, `ScalarMultiplyReduction`, is
called only from the convolution visitor in the pinned snapshot; its TEPL dot
identity is valid but excluded from the XLA comparison.

Store the two measured source texts, all proof/compile/validation evidence,
artifact hashes, per-variant and responsibility counts, compiler hash, baseline
measurement hash, span ledger, protocol, and source hashes. Check inputs again
after execution. `--check` recomputes compaction, source counts, cohort membership,
and source attribution without invoking the compiler. Preserve the older frozen
coverage and historical attribute-inference records.
