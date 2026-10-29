# Specification-size protocol

This experiment measures how much source text axis inference removes from a
fixed set of XLA-derived TEPL rules. Tokens are the primary size metric. The
experiment does not measure authoring time, readability, optimizer speed, or
full XLA implementation equivalence.

## Questions and cohort

1. How much smaller are complete rules when RHS axis expressions become `auto`?
2. How much of that change comes from field expressions or helper computations?
3. How do savings vary by operation family, expression form, forced choices, and
   unit-view normalization?
4. Can reviewed XLA implementation excerpts support a separate size comparison?

Select all 32 non-control `axis_inference` cases in the frozen
`coverage/data/synthesis_classification.json` before running the new compiler.
Selection uses task type, never the old success result. Check the frozen source
and task hashes and require the current migration to produce the same tokens
and normalization records. A changed cohort requires a new protocol version.
The 80 verification-only cases and 97 excluded cases do not enter this size
experiment. The negative control must fail semantic verification.

Keep every attempted case and its diagnostics. The primary size cohort contains
only pairs for which both matched variants prove, compile, and validate.
Report all attempted pairs separately, including failures. Also recheck the
original handwritten rule for each case; its outcome does not silently replace
the paired acceptance criteria.

## Matched representations

The inferred variant is the current axis-only migration: every eligible
original RHS axis field is `auto`; all other fields remain supplied. Expand
descriptor copies into named fields and apply the existing checked singleton
view normalization.

Construct the explicit counterpart from that same migrated source. Restore each
target's original handwritten field expression, and restore only the transitive
closure of original local helpers those expressions require. Preserve helper
order and expressions. Do not substitute guards, fold expressions, choose a
different axis assignment, add assumptions, or remove legality conditions.

Require identical token sequences for tensor constraints, LHS and RHS patterns,
guards, supplied non-axis expressions, descriptor syntax, fixed adapter
assignments, and outer rule syntax. Only axis expressions and needed helper
statements may differ. This comparison isolates inference from migration,
descriptor expansion, formatting, and normalization.

Record the original handwritten file as a third TEPL representation. Its size
difference from the matched explicit version is a migration effect, not an
inference saving. It can be smaller because descriptor copies are compact.
Report a separate original-source comparison on the subset where all three
variants pass verification, compilation, and validation. A reduction against
the expanded matched baseline must not be described as a reduction against the
original compact files.

Each variant must independently pass:

- `tepl check --semantic-policy=algebraic --rule NAME`, with one proved rule and
  `additional-guard: true`;
- `tepl compile --semantic-policy=algebraic`;
- `tepl validate`, with no unresolved executable holes.

For inferred rules, the number of emitted automatic constructors must match
the target descriptor inventory. Any valid proved assignment counts; equality
to the handwritten assignment or XLA's profitability choice is not required.
This acceptance criterion is about semantic witnesses, not optimization quality.

## Counting policy

Use the experiment's versioned, maximal-munch lexical scanner. An identifier,
numeric literal, quoted string/character literal, or listed multi-character
operator counts as one token; remaining punctuation counts individually.
Comments and whitespace count zero. C++ raw strings are single tokens. No
preprocessing, macro expansion, subword tokenizer, or compiler IR is involved.
The operator inventory is shared across languages, so these are explicitly
defined lexical size units, not a claim about compiler-internal token counts.

Primary TEPL totals cover the complete `rule` declaration, including tensor
constraints, both patterns, guards, RHS attributes, and derivation helpers.
Imports are counted separately. Shared dialect files are counted once as whole
files and identified as setup context; they are not claimed to be a minimal
operation dependency slice. Never multiply shared definitions by rule count.

Record two line measures:

- **Code lines:** physical lines containing a non-comment lexical token. A
  multiline literal contributes all its occupied lines. Blank/comment-only
  lines cost zero.
- **Layout lines:** use a fixed layout for both TEPL variants: flush a line
  after `{`, `}`, `;`, and `=>`, and before `}` if text is pending. This avoids
  crediting a rule written on one long line. The raw source is retained so that
  another formatting convention can be evaluated later.

Layout lines are secondary and remain sensitive to the stated layout policy.
Do not compare TEPL layout lines directly to raw C++ physical lines.

## Attribution and summaries

Assign every TEPL rule token to exactly one component:

| Component | Includes |
|---|---|
| Axis values | Value expressions of original target fields; each `auto` costs one token |
| Local helpers | Complete `let` statements, including names and punctuation |
| Supplied values | Non-target field values, excluding fixed adapter assignments |
| Descriptor syntax | Labels, `.attrs`, assignment, field names, and record punctuation |
| Normalization adapter | Complete fixed RHS adapter assignments |
| Tensor constraints | Shape/dtype declarations before the LHS |
| LHS / RHS patterns | Complete graph-pattern token spans |
| Guard | Whole `where` block, including its local bindings and syntax |
| Rule / derive syntax | Remaining declaration, block, and arrow tokens |

The component differences must sum exactly to the complete-rule difference.
Classify each handwritten axis expression as a field copy, empty list, literal,
helper reference, or computed expression. Helper statements are counted only
in their own component; do not count their expanded bodies again as field costs.

Report aggregate token reduction and median per-rule reduction. Show per-case
counts and family totals. Separate cases whose targets are all marked
scalar-forced from the remaining cases, and normalized from unnormalized cases.
The frozen forced marker recognizes only rank-zero broadcast's empty mapping;
it does not prove that other tasks admit multiple assignments. A general
uniqueness audit would need its own procedure. Retain zero or negative savings.
Show all attempted pairs as a sensitivity table, clearly including unproved
specifications. This fixed corpus is not a random sample; do not attach a
population confidence interval or a significance claim to these totals.

## XLA excerpt audit

The catalog's source references often cover whole handlers and helper methods.
They include unrelated rewrites and therefore provide context, not automatically
valid excerpts for a compression ratio.

`data/xla_excerpts.json` initially records those contexts for every selected case.
To mark a case `reviewed`, choose exact inclusive line ranges in the pinned XLA
snapshot and write a case-specific review note. Include matching, legality, RHS
attribute calculations, and construction needed by the chosen rewrite. Identify
shared helpers, alternate branches, source policy, and any omitted behavior.
Preserve the catalog's numerical and specialization caveats. A second reviewer
should check these decisions before publication.

Separate the algebraic transformation from backend profitability/layout policy
where the source permits it; report both boundaries if both claims are useful.
An excerpt must be understandable with its declared shared dependencies. Do not
select only an assignment while omitting required calculations, or count an
entire handler as if it implemented only one rule.

Count overlapping and adjacent ranges once per source file for aggregate sizes.
Do not sum per-case C++ totals when they share code. Measure selected tokens in
the full source lexical stream so a comment crossing a boundary is handled
correctly. Keep excerpt counts pending in the main table until every member of
the paired cohort has explicit ranges and a review note. Broad context totals
are a diagnostic appendix and must not be presented as matched XLA costs.

## Provenance and limitations

Freeze the compiler hash, source and fixture hashes, original selection and
result files, experiment scripts and protocol, XLA snapshot, excerpt manifest,
and current proof document. Check those hashes again after execution. Store
commands, proof output, compile/validation results, artifact hashes, source
text, and per-field attribution in JSON. `--check` must detect changed inputs or
stale reports without invoking the compiler. Do not overwrite the historical
attribute-inference measurements.

The arithmetic model is ideal `algebraic`, not unrestricted IEEE arithmetic or
all XLA backend policies. Some cases specialize ranks or operations. Completion
may select an equivalent witness different from XLA's intended optimization.
Any paper should report these facts next to the size and inference results.
