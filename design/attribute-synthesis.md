# Attribute synthesis

Production synthesis fills field-level `axis` holes in a fixed replacement tree.
Extent, coordinate, and general fields remain supplied, defaulted, or optional.
Acceptance requires independent semantic verification. For operation contracts
and proof obligations, see [semantic verification](semantic-verification.md).

## Descriptor lifecycle

Checking represents requests with `AutoField` and expands defaults through
`src/core/checking/attribute_fields.h`. Omitted attributes and explicit
constructors enter semantic planning through checked derivations.
`Plan::descriptors` is the descriptor-state table: captured descriptors read
source metadata; constructed descriptors contain supplied values and individual
holes in schema order. Nodes carry no separate automatic mask or hole manifest.

Materialization replaces only automatic fields and preserves supplied expressions
and defaults. Unresolved holes cannot enter executable IR. Optional operation
labels remain occurrence metadata, separate from tensor captures and descriptors.
`node.attrs` resolves to a descriptor reference; source spelling is retained for
compiler diagnostics and suggestions.

## Construction and verification

`attr_synth::synthesize(plan, strategy)` returns an unverified candidate or a
construction failure. `semantics::proveRule` tries identity, ordered, then general
construction. Every candidate is checked in a fresh verification context before
acceptance. Construction or verification failure permits fallback; it does not
prove that every attribute assignment is impossible.

| Owner | Responsibility |
| --- | --- |
| `src/core/attr_synth/` | Candidate strategies, axis selectors, descriptor construction |
| `src/core/semantics/symbolic/` | Typed expressions, coordinate correspondence, normalization |
| `src/core/semantics/index_maps/` | Axis maps, transport, and set reasoning |
| `src/core/semantics/proof/` | Independent legality/equivalence verification |
| `src/core/semantics/prove_rule.cc` | Strategy orchestration and checked acceptance |

Shared lowering/reasoning lives in `//src/core/semantics:reasoning`. Synthesis and
`//src/core/semantics:proof` depend on it without depending on each other.
`//src/core/semantics:rule_proof` coordinates them. The ordered strategies share `//src/inference` component decomposition and
validation. The `//src/core/attr_synth:ordered` target exposes compilation to
standalone concrete callers.

The identity route copies descriptors for matching operation trees. Verification
checks operation/operand/attribute identity, including repeated captures, before
general access correspondence. Calculated-read axis selection is handled by
`axis_selectors.cc`; former coordinate/extent/scalar construction remains in
[excluded legacy code](../src/core/attr_synth/legacy/README.md).

## Ordered specialization

The [ordered implementation](../src/core/attr_synth/ordered/) separates axis
membership, ordering requirements, and positional descriptor construction.

| Entry point | Purpose |
| --- | --- |
| `recognize(Definition)` | Recognize a conservative axis/value template independently of operation, group, and field names |
| `admit(Plan, Mode)` | Check semantic/tree restrictions and the RHS attribute policy |
| `compile(Plan, Mode)` | Produce completed postorder instructions and field adapters |
| `inference::emit(...)` | Construct the shared symbolic metadata IR using the proved three passes |
| `construct(Context&, EmissionBlock, ...)` | Specialize the shared IR into independently checked positional expressions |

`contract.*` recognizes generalized contractions with batch/left-free/right-free
output order, unrestricted full permutations, whole-axis unary sums preserving
survivor order, and attribute-free addition of equally shaped operands. Extra
semantic restrictions are rejected. This is a conservative recognizer, not a
decision procedure for equivalent spellings.

Addition is processed around the single-component constructor: distribute the
source, require an explicit separated RHS sum, and match components by distinct
input sets. Every component is read-once, although input sets may overlap.
Source descriptor copies keep their captured identities. RHS unknowns remain
independent across the original replacement tree; distribution never copies them.
Component axis partitions remain separate, and actual additions must have equal
ordered operand shapes.

After identity fails, an admitted ordered attempt can record
`law: ordered-requirements-construction`. Generation returns source-only
feasibility obligations and attribute expressions defined when those obligations
hold. Generation does not specialize its predicate using the written guard
or call the guard prover as a construction precondition. A fresh
verifier must prove feasibility from source legality and the written conditions,
then check builder definedness, legality, and equivalence. Generation may succeed
while guard verification remains Unknown. Feasibility is never assumed or inserted
as an automatic runtime guard. Admission or bounded verification failure leaves
general synthesis available.

The concrete solver covers contraction, permutation, summation, and separated
addition in its extent-at-least-two domain. `Compilation::domain_requirements`
exposes that domain separately from admission. Rank-zero tensors and empty axis
lists are allowed. Singleton/zero extents, mixed supplied/automatic descriptors,
and repeated captures within a component remain outside its completeness claim;
other verified compiler routes may support them.

## Shared constructor and residual program

Both ordered symbolic synthesis and native execution now start with the same
`inference::emit` metadata IR. Emission precedes proof-context creation and
normalization. `attr_synth::Result::metadata_program` remains available even if
bounded expression specialization fails. The duplicate symbolic membership,
requirement and layout recurrences are removed; the symbolic adapter interprets
IR nodes through established axis mappings/partitions. A paired contraction
permutation is an optional finite-sum reindexing checked by the independent
verifier. Membership-set enumeration is unobservable; ordered axes remain
sequence values.

The completeness guarantee concerns construction and interpretation of this
residual program on admitted plans. Independent guard certification and optional
Rust/C++ expression lowering may still be unresolved. Existing CLI policy modes,
portable schemas, broader construction, and sound extensions retain their
boundaries; no complete guard prover or broader completeness is claimed.

## Native metadata evaluation

[`src/inference/evaluate.h`](../src/inference/evaluate.h) interprets completed
instructions from [`inference.h`](../src/shared/metadata/inference.h), without
compiler plans or symbolic search. `ordered::compile` adapts semantic plans for
compiler-side callers and solver tests.

1. Extract named physical input ports and captured positional attributes. Retain
   the complete port partition, including reduced classes, and output order.
2. Derive mandatory classes M from capture support and the output interface,
   then canonical retained classes P bottom-up. Inputs keep all axes; permutations
   preserve them; sums retain mandatory child axes; contractions retain mandatory
   shared axes and all unpaired axes.
3. Propagate required subsequences from the root. Contractions require
   batch/left-free/right-free block order; a transpose removes its child's order
   requirement; a sum passes the surviving-axis requirement unchanged.
4. Check distinctness, membership in P, input subsequences, contraction block
   order, and exact equality of the root retained set with the target output set.
5. Choose complete layouts and translate logical selections into positional
   fields. Sums remove selected child axes and preserve survivor order.

For additive programs these passes run independently per matched component.
Their predicates are conjoined and independent assignments combined. Results
expose component networks rather than a merged global partition. Serialized
component schedules are validated against the original trees.

Logical identity is a physical-port equivalence class, never an extent or a
capture-support set alone. `analyze` establishes requirements without deciding
applicability; `construct` checks feasibility before using analyzed metadata;
`replayReplacement` checks actual replacement descriptors. The standalone
`feasibility` and `evaluate` APIs support certificates and mathematical tests.
`inference::check()` independently rebuilds source requirements and checks actual
RHS layouts, partitions, and output order without trusting constructor choices.

A checked order inversion or retained-set obstruction yields an impossibility
certificate in the admitted concrete domain. Missing metadata, unsupported operations, invalid
source descriptors, out-of-domain extents, and exhausted work have separate
outcomes. Concrete impossibility is not currently used to reject an entire
rank-polymorphic rule schema.

## Conditions and suggestions

Source logical-axis references remain typed until both semantic trees exist,
then lower to pure metadata expressions. Membership uses `LogicalAxisSet`,
ordered axes use `LogicalAxis[]`, and positions use `u64[]`.

`check --suggest-where` performs bounded search over source-group equalities and
identity permutations. Each proposal needs fresh synthesis, independent
verification, and a concrete legal metadata witness under the proposed
restriction. That witness establishes non-vacuity, not universal equivalence.
Suggestions are reviewable source; they do not change the original rule's
acceptance or compiled applicability. Search failure means no suggestion found.

## Research and validation

The [research index](../research/attribute-synthesis/README.md) separates the
paper, completeness/soundness arguments, independent drafts, and finite checks.
The broader signed/weighted state-table mathematical constructions do not imply
that production search implements their full completeness guarantees.

```sh
bazel test //tests:ordered_synthesis_test //tests:attribute_reasoning_test \
  //tests:where_suggestions_cli_test //tests:named_nodes_test
```

Tests exercise positive/negative certificates, independent attribute oracles,
metadata domains, repeated inputs, defaults, suggestions, and corrupted
constructors. They provide implementation evidence, not formal refinement proof.
