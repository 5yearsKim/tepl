# Ordered contraction/permutation synthesis

This directory separates axis membership, ordering requirements, and descriptor
construction. The [unified completeness proof draft](../../../../examples/research/complete_attribute_synthesis_proof.md)
proves exact feasibility and constructive synthesis for a mathematical
contraction/permutation/unary-summation algorithm and separated additive
components. Section 6 gives the metadata-program generation argument.
This implementation covers its original contraction/permutation specialization;
unary summation and additive component support on the exact path remain
extension work. Formal implementation refinement remains separate work.

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

After descriptor-copy identity fails, `semantics::proveRule` tries the ordered
strategy in `attr_synth::synthesize` for an admitted plan. All order constraints
remain proof obligations under the original declared conditions. They are never
assumed or silently added as runtime guards. The strategy returns an unverified
candidate; semantic orchestration checks it in the fresh-context verifier before
acceptance. Successful proofs record
`law: ordered-requirements-construction`.

If admission or bounded symbolic reasoning cannot support this route, the
general synthesizer remains available. Therefore this integration does not
claim completeness of symbolic source correspondence, the implication prover,
or the final general verifier. Both paths still preserve the selected dtype
and arithmetic checks. Concrete impossibility certificates are not currently
used to reject an entire rank-polymorphic rule schema.

TEPL accepts semantic rules only after independent verification under source
legality and the written conditions. It never materializes additional runtime
feasibility guards. `check --suggest-where` performs bounded search for a
verified sufficient restriction; review and add a suggestion before normal
compilation. `restrictions.cc` checks proposals with fresh synthesis and
verification, and rejects suggestions without a concrete legal witness.
Runtime inference programs and the old `--infer-applicability` mode are retired.

The complete concrete solver below remains available for mathematical
experiments and structural certificates in its admitted extent >= 2 domain.
Those restrictions do not constrain the general symbolic verifier. Identical
operation trees, including repeated captures, use verified attribute copies;
other practical cases use general construction and independent verification.
Mixed explicit/automatic RHS descriptors, singleton extents, and zero extents
may therefore be accepted without extending the completeness claim.

## Native metadata-program evaluator

[`src/inference`](../../../../src/inference/evaluate.h) interprets the
completed instructions in [`inference.h`](../../../shared/metadata/inference.h),
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
4. Ordinary `where` expressions check subsequences and batch/free block order.
5. Construction chooses complete layouts only after those checks succeed and
   converts the same logical pairing sequences into positions on both operands.

Logical identity is a physical-port equivalence class, never an extent or a
capture-support set alone. Equal-sized axes remain distinct. The concrete
metadata domain requires every extent to be at least two; rank-zero tensors
and empty attribute lists are allowed.

The native API separates `analyze`, `construct`, and `replayReplacement`.
Analysis makes no applicability decision; construction requires established
preconditions. The standalone `feasibility` (no construction) and `evaluate`
(analysis plus guarded construction/explicit validation) APIs remain available
for mathematical solver tests and certificates. Results report `Analyzed`, `Feasible`,
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
bazel test //tests:ordered_synthesis_test //tests:attribute_reasoning_test \
  //tests:where_suggestions_cli_test //tests:named_nodes_test
```

The ordered-solver tests exercise its original completeness domain. Attribute
reasoning tests cover normal rule acceptance, descriptor-copy identity,
singleton/zero extents, mixed attributes, repeated inputs, independent
verification, and sufficient restriction suggestions. CLI tests ensure that
suggestions never silently accept or strengthen a rule. Tests provide evidence
for implementation correspondence, not a formal compiler refinement proof.
