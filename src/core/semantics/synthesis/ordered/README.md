# Ordered contraction/permutation synthesis

This directory separates axis membership, ordering requirements, and descriptor
construction. It implements the compilation side of
[the completeness plan](../../../../../examples/research/complete_attribute_synthesis_plan.md).
The [completeness proof draft](../../../../../examples/research/complete_attribute_synthesis_proof.md)
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
An admitted rule receives a completed `inference::Program` containing the
source/replacement schedules. Its feasibility formula is emitted as ordinary
typed `where` conditions by `shared/catalog/axis_conditions.*`; there is no
separate runtime checker call. `check --infer-applicability` prints a suggested
complete named rule. Copying it into source permits normal compilation without the
flag. The flag explicitly requests insertion of the conditions into the
completed program; ordinary compilation never silently adds them.

Named node properties expose required subsequences, batch/free sets, and
physical input order. Sequences have type `LogicalAxis[]`; membership has the
distinct type `LogicalAxisSet`. `filter`, `concat`, equality, and `subsequence`
express the checks. Public syntax uses `result.axes.B.members`,
`result.required_order`, and `A.axes.order`. The compiler lowers these to closed
queries with internal node IDs; numeric query calls are not source builtins.
Identities are canonical physical-port classes determined by captured source
attributes at each match. Labels do not add tensor captures or alter synthesis.

The compiler and executable loader regenerate the required formula and check
its conjuncts after expanding local bindings and conjunctions. Removing a
condition or hiding it under a disjunction fails verification. This route does
not claim complete implication reasoning for differently spelled predicates;
the existing symbolic prover remains available for ordinary rules. Mixed
explicit/automatic RHS attributes remain outside admission.

At a match, `rewrite::AxisAnalysis` caches source replay, boundaries, and order
propagation. The ordinary condition evaluator decides feasibility. Only after
conditions pass does the constructor use the same analysis to assign positions;
it does not repeat a feasibility decision. Backend shape/dtype checks validate
the replacement before graph mutation.

Fully explicit RHS rules use the same schedules in `Mode::Validate`.
`result.legal` exposes descriptor legality; equality of source/RHS
`axes.partition` and `axes.order` properties establishes equivalence. RHS queries and dependent
local bindings execute after ordinary descriptor derivations. Independent
source conditions execute first, and `where`/`derive` local scopes remain
separate. These rules need no constructor and never substitute a different
assignment.

Compiled JSON stores the completed schedules and declares the
`tepl.inferred-applicability.v1` capability. Loading checks tree structure,
capture sets, field bijections, references to the actual rule, and field-role
adapters against operation axis contracts, and the mandatory explicit conditions. Deferred code-generation backends
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

## Explicit condition workflow

```sh
bazel-bin/tepl check examples/semantic/inferred_applicability.tepl \
  --semantic-policy=algebraic --infer-applicability --rule remove_output_permutation
bazel-bin/tepl compile examples/semantic/explicit_axis_conditions.tepl \
  --semantic-policy=algebraic -o /tmp/explicit.json
bazel-bin/tepl rewrite /tmp/explicit.json --graph feasible --trace
```

`axis_domain()` checks source legality and the theorem's metadata domain.
Outside-domain data is not mathematical infeasibility. Missing data skips a
match; work exhaustion remains a distinct resource limit. Extra user conditions
may intentionally exclude feasible matches; exactness concerns the generated
conjunction itself within its domain.
