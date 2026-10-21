# Ordered contraction/permutation synthesis

This directory separates axis membership, ordering requirements, and descriptor
construction. It implements the compilation side of
[the completeness plan](../../../../../docs/complete_attribute_synthesis_plan.md).
The [completeness proof draft](../../../../../docs/complete_attribute_synthesis_proof.md)
proves the concrete algorithm's necessity, constructive sufficiency,
termination, and certificate properties in its stated domain. Section 12 also
proves correctness of a specified abstract generator for feasibility and
attribute programs. The C++ implementation now provides the corresponding
compiled-plan path; formal implementation refinement remains separate work.

This implementation stage covers the C++ compiler and structural solver.
Rust code-generation and runtime changes are deferred.

## Entry points

| API | Purpose |
| --- | --- |
| `recognize(Definition)` | Recognize a conservative checked axis/value template, independently of operation, group, or field names. |
| `admit(Plan, Mode)` | Check the shared semantic/tree fragment and the mode-specific RHS attribute policy. |
| `compile(Plan, Mode)` | Generate completed postorder instructions, canonical field adapters, and static subtree capture sets. |
| `deriveRequirements(...)` | Compute symbolic roles and propagate required subsequences before assigning attributes. |
| `construct(...)` | Construct symbolic positional fields bottom-up using those requirements. |
| `solve(Plan, Metadata)` | Convenience wrapper around compilation and native program evaluation. |
| `check(Plan, Metadata, ConcreteResult)` | Independently replay completed positional attributes and compare networks, or check the claimed order inversion. |

`contract.*` recognizes generalized contractions with the declared batch,
left-free, right-free output convention and unrestricted full permutations.
It rejects extra contraction assertions and recognizes only the ordinary
permutation-validity assertions for transposes. Field and group indices are
recovered structurally. This is not a recognizer for every semantically
equivalent definition spelling.

## Compiler integration

The default synthesizer first tries requirements-driven construction for an
admitted plan. All order constraints remain proof obligations under the
original declared conditions. They are never assumed or silently added as
runtime guards. A completed candidate is checked in the existing fresh-context
verifier before acceptance. Successful proofs record
`law: ordered-requirements-construction`.

If admission or bounded symbolic reasoning cannot support this route, the
general synthesizer remains available. Therefore this integration does not
claim completeness of symbolic source correspondence, the implication prover,
or the final general verifier. Both paths still preserve the selected dtype
and arithmetic checks. Concrete impossibility certificates are not currently
used to reject an entire rank-polymorphic rule schema.

The explicit `SemanticOptions::inferred_applicability` option selects the new
path instead. The CLI exposes it as `--infer-applicability` on `check`,
`compile`, and source `rewrite`; it requires `--semantic-policy=algebraic`.
An admitted rule receives a completed `inference::Program`. No general
symbolic correspondence or condition implication search is needed to generate
this program. Rule conditions are still evaluated before attribute derivations
and inference. Mixed explicit/automatic or pinned RHS fields are rejected.

At a match, the native engine runs source analysis, boundaries, order
propagation, feasibility, and construction once, sharing the analysis. A false
feasibility result skips the match before positional construction. Backend
shape/dtype checks still validate the replacement before graph mutation.
`VerificationKind::InferredApplicability` and inspection output identify the
conditional guarantee; the mode does not promise feasibility for every match
satisfying the user's conditions.

Fully explicit RHS rules use the same compiled schedules in `Mode::Validate`.
After ordinary descriptor derivations, the evaluator replays the supplied RHS
attributes and compares the complete source/RHS network and output order. It
does not substitute another feasible assignment. Such rules need no constructor.

Compiled JSON stores the completed schedules and declares the
`tepl.inferred-applicability.v1` capability. Loading checks tree structure,
capture sets, field bijections, references to the actual rule, and field-role
adapters against operation axis contracts. Deferred code-generation backends
reject these programs explicitly; native C++ execution is currently supported.

For example, from the repository root:

```sh
bazel-bin/tepl compile examples/semantic/inferred_applicability.tepl \
  --semantic-policy=algebraic --infer-applicability -o /tmp/inferred.json
bazel-bin/tepl rewrite /tmp/inferred.json --graph feasible --trace
bazel-bin/tepl rewrite /tmp/inferred.json --graph infeasible --trace
```

The first graph rewrites; the second retains its required output permutation,
even though both inputs and outputs have equal extents.

## Native metadata-program evaluator

[`src/inference`](../../../../../src/inference/evaluate.h) interprets the
completed instructions in [`inference.h`](../../../../shared/metadata/inference.h),
independently of compiler plans or symbolic proof search. The former duplicate
concrete execution engine was removed; `concrete.*` retains the convenience
Plan APIs used by compiler-side callers.

1. Source extraction records each named physical input port and interprets
   captured permutations and contraction lists. A union-find retains the full
   port partition, including reduced classes, and the ordered final interface.
2. Boundary analysis derives the required live classes at each RHS subtree
   from their capture supports and final-output membership.
3. Ordering analysis propagates required subsequences from the root. A
   contraction requires nondecreasing batch/left-free/right-free roles;
   a transpose removes the child's ordering requirement.
4. Feasibility checks exact local role/physical-order inversions.
5. Construction chooses complete layouts only after those checks succeed and
   converts the same logical pairing sequences into positions on both operands.

Logical identity is a physical-port equivalence class, never an extent or a
capture-support set alone. Equal-sized axes remain distinct. The concrete
metadata domain requires every extent to be at least two; rank-zero tensors
and empty attribute lists are allowed.

The native API offers `feasibility` (no construction) and `evaluate` (analysis
plus guarded construction/explicit validation). Its result reports `Feasible`,
`Solved`, `Impossible`, `Unsupported`,
`OutsideDomain`, `InvalidSource`, `MissingMetadata`, or `ResourceLimit`.
Explicit validation additionally reports `InvalidReplacement` or `NotEquivalent`.
Only a checked order inversion means `Impossible`. Missing data, singleton or
empty dimensions, invalid source descriptors, and exhausted work do not.

`check()` rebuilds the source, recomputes the requirements, and interprets the
actual RHS positional descriptors without using the constructor's choices.
It checks all recorded layouts and compares the complete port partition and
ordered output. It also verifies negative witnesses against rederived roles.
It certifies structural ideal-arithmetic semantics in the admitted domain;
backend dtype restrictions and coverage of user conditions are separate.

## Validation

```sh
bazel test //tests:ordered_synthesis_test
bazel test //tests:inferred_applicability_test //tests:inferred_applicability_cli_test
```

The test enumerates positional attributes using a separate equivalence-matrix
oracle. It checks 1,584 decisions across scalar/vector inputs, tree shapes,
capture orders, and optional transposes, including 240 impossible layouts.
Additional regressions cover multiple unsorted contracting axes, equal extents,
rank-128 permutations, corrupted certificates, and an internal transpose
required by a parent's output order. The compiler regression evaluates the
generated metadata constructors under the original declared conditions.

Native integration tests cover actual source and serialized-JSON rewriting,
one program across scalar through rank-128 inputs, impossible equal-shaped
layouts, internal transpose construction, explicit assignment validation,
source descriptor sharing, unsupported schemas, and corrupted program rejection.
These finite tests corroborate the implementation; they do not prove
arbitrary-rank completeness or formal compiler refinement. The general verifier
remains the acceptance boundary for the default declared-condition mode.
