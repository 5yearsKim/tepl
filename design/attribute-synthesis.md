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
`//src/core/semantics:rule_proof` coordinates them. The concrete ordered adapters
have a separate `//src/core/attr_synth:ordered` target using `//src/inference`.

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
| `deriveRequirements(...)` | Propagate required subsequences before choosing attributes |
| `construct(...)` | Construct symbolic positional fields bottom-up |

`contract.*` recognizes generalized contractions with batch/left-free/right-free
output order and unrestricted full permutations. It rejects extra contraction
assertions and recognizes ordinary permutation legality for transposes. This is
a conservative recognizer, not a decision procedure for equivalent spellings.

After identity fails, an admitted ordered attempt can record
`law: ordered-requirements-construction`. Ordering constraints must follow from
source legality and written conditions. They are never inserted as runtime
applicability guards. Failure of admission or bounded symbolic reasoning leaves
general synthesis available.

The complete concrete solver retains its original contraction/permutation,
read-once, extent-at-least-two domain; rank-zero tensors and empty attribute
lists are allowed. Singleton/zero extents, mixed supplied/automatic descriptors,
and repeated captures can be supported by other verified compiler routes without
expanding that solver's completeness claim.

## Native metadata evaluation

[`src/inference/evaluate.h`](../src/inference/evaluate.h) interprets completed
instructions from [`inference.h`](../src/shared/metadata/inference.h), without
compiler plans or symbolic search. The ordered `concrete.*` convenience APIs
adapt plans for compiler-side callers and solver tests.

1. Extract named physical input ports and captured positional attributes. Retain
   the complete port partition, including reduced classes, and output order.
2. Derive required live classes at each RHS subtree from capture supports and
   the final output interface.
3. Propagate ordering requirements from the root. Contractions require
   nondecreasing batch/left-free/right-free roles; a transpose removes its
   child's ordering requirement.
4. Check subsequences and block order with ordinary metadata expressions.
5. Choose complete layouts and translate common logical pairing sequences into
   positions on both operands.

Logical identity is a physical-port equivalence class, never an extent or a
capture-support set alone. `analyze` establishes requirements without deciding
applicability; `construct` requires established preconditions;
`replayReplacement` checks actual replacement descriptors. The standalone
`feasibility` and `evaluate` APIs support certificates and mathematical tests.
`inference::check()` independently rebuilds source requirements and checks actual
RHS layouts, partitions, and output order without trusting constructor choices.

Only a checked order inversion yields an impossibility certificate in the
admitted concrete domain. Missing metadata, unsupported operations, invalid
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
