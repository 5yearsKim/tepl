# Three-representation source-section comparison

Select the 31 non-control `axis_inference` tasks with successful proof,
compilation, and JSON validation from
`../coverage/results/broadcast_update.json` (the 31/32 companion run). Require
`status=proved`, `additional-guard: true`, successful compile/validation return
codes, and an artifact hash. Both successful input source hashes and original
source hashes must match the recorded run before counting. The broader inventory
has six additional unresolved slice/convolution axis tasks; they contribute no
successful rules here. Build the explicit representation by restoring
handwritten axis expressions to the same normalized graph as the successful
axis-auto task, using `support.matched_pair`. Spell out the axis fields in
`derive` rather than retaining inline whole-descriptor copies. For the
broadcast–transpose companion, use its existing local `d` as the explicit
broadcast mapping; retain its supported supplied shape construction.

For the inferred representation, omit inferable axis fields using
`support.compact_source`: preserve fields whose omission would restore
a schema default, supplied non-axis fields, required helper computations, and
fixed adapter assignments. Remove unused node labels, empty assignments, and
empty derive blocks. Conditions and graph structure must be preserved; node
labels may disappear when unreferenced. Do not leave explicit `auto` markers.
Store full measured source snapshots.

The main graph/table uses 30 cases with corresponding XLA branch mappings. The
dot scalar multiplier case lacks a corresponding XLA dot implementation in the
pinned snapshot; do not substitute the convolution implementation or count it
as zero. Include it in a separate 31-case TEPL-only table. These are matched
case identities, not a requirement that the explicit representation verify or
that XLA and TEPL support identical ranks, operand orders, or numerical policy.

Partition each complete TEPL rule into exactly these five rows:

1. Matching (LHS): the entire LHS graph span.
2. Legality checks (where block): the entire block, including keyword/braces,
   predicates, and locals. Absent blocks count zero.
3. Attribute reasoning (derive block): the entire block, including
   keyword/braces, all supplied/inferred fields, and all helper computations.
4. Replacement construction (RHS): the entire RHS graph span.
5. Other syntax: every remaining rule token, including shape/type declarations,
   rule header, rewrite arrow, and outer braces.

Any inline descriptor remaining in a graph belongs to that graph's section.
The explicit generation puts RHS attribute transformations in derive blocks.
This is a literal source-section classification. Use the scanner in
`scripts/support.py`; comments and whitespace count zero. Imports/use and shared dialect/compiler definitions
remain outside whole-rule totals. Enforce exhaustive, nonoverlapping accounting.

For XLA use the single reviewed source ledger `data/xla_spans.json`. Map XLA
matching, legality, attributes, and construction to the corresponding rows, and management
to Other syntax. XLA has no literal where/derive sections; these are source-line
responsibility assignments based on dominant responsibility. Deduplicate all
selected file/line tokens globally; conflicting categories are errors. Include
transformation-specific local helpers once. General HLO/shape APIs remain library
dependencies. Preserve separate excluded policy/logging counts, as specified in
the ledger. Annotations include original-value-recovery graph
bookkeeping in the unit-insertion broadcast branch. Annotations are from one
analyst's source inspection and have not been independently reviewed.

Generate the 30-case graph in PNG/SVG, Markdown and LaTeX tables for both cohorts,
an annotated XLA source listing, and JSON with per-case TEPL sections, source
snapshots/hashes, and tool, protocol, source-snapshot, manifest, and evidence
hashes. Use `scripts/section_comparison.py --run` to check both measured
representations with one compiler: proof, compilation, and JSON validation.
Record diagnostics for unresolved explicit rules and retain them for source
counting; require all 31 compact inferred variants to pass. Save verification
in `results/section_verification.json` with a digest of the exact source corpus,
compiler hash, dialect hashes, and timestamps. Frozen evaluation results remain
unchanged. Without `--run`, reuse only verification with the same source digest.
`scripts/section_comparison.py --check` verifies all text outputs and accounting;
graph generation additionally requires matplotlib. Inspect the rendered graph.

Both measured TEPL representations have the same normalized graph and conditions;
the reduction includes expression removal, omitted-field syntax, unused-label
removal, and empty-block removal. The expanded explicit baseline is not the
original compact handwritten baseline, and its reduction must not be presented
as savings against those original files. Do not infer runtime speed or
authoring effort from these lexical counts.
