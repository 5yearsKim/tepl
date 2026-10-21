# Plan: complete attribute synthesis for TEPL

**Status:** proposed refactor and proof development, October 21, 2026. This document specifies work to do; it does not claim that the current implementation is complete or that the proposed proofs have been independently checked.

**Refactor progress:** the first structural implementation now lives in
[`synthesis/ordered`](../src/core/semantics/synthesis/ordered/README.md).
It includes conservative operation/schema admission, a concrete source-port
network and structural solver, separately represented symbolic axis/order
requirements, bottom-up construction, and concrete witness checking. The main
synthesizer tries the new requirements-driven construction for admitted schemas
and independently verifies candidates before acceptance, retaining the generic
fallback. An explicit inferred-applicability mode now compiles supported rules
into validated metadata schedules and executes a generated feasibility check
and guarded constructor in the native C++ engine. Fully explicit attributes
use a separate replay-validation mode. The default declared-condition semantics
is preserved; `--infer-applicability` opts into match-time applicability.

**Proof progress:** [the concrete completeness proof](complete_attribute_synthesis_proof.md)
now supplies hand proofs of exact network semantics using one-hot inputs,
forced boundaries and pairings, necessary ordering constraints, legal and
semantics-preserving construction, termination, and certificate soundness and
acceptance. Section 12 additionally specifies an abstract metadata-program
generator and proves its exact feasibility checks and correct construction for
arbitrary finite ranks in F0. It is not a mechanized proof or a formal
verification of the C++ implementation. Section 13 specifies the supported TEPL
templates and a translation preserving meanings and attribute assignments.
The implementation now follows this architecture; formally establishing its
refinement of the abstract generator remains open. The checklist below
remains a review/acceptance checklist; universal condition coverage is separate
future work.

**Current execution scope:** continue the C++ compiler/semantic refactor and its
tests. Rust code-generation and runtime implementation work are deferred at the
user's request; the backend milestones below remain future work.

## 1. Objective and recommended scope

Refactor TEPL so that, for a precisely admitted contraction/permutation fragment, attribute inference is a terminating constructive decision procedure:

\[
\operatorname{Infer}(S,p)=\operatorname{Solved}(\beta)
\quad\Longleftrightarrow\quad
\exists\beta'\;\operatorname{Valid}(S,p,\beta').
\]

Every returned `beta` must itself be valid. Here `S` is the source/replacement schema and `p` is concrete source metadata. The replacement expression structure is fixed; its attributes are unknown.

The central architectural separation is:

```text
Certified operation semantics + legal LHS metadata + RHS structure
    -> exact source logical-axis representation
    -> forced intermediate axis sets and pairings
    -> necessary ordering requirements
    -> compatible complete layouts
    -> concrete positional attributes
    -> independently checked certificate
```

Do not equate an axis set with a layout. The set records which logical indices survive; the layout records their order. Attribute construction must satisfy both.

Develop three guarantees separately:

| Guarantee | Meaning | Required milestone |
| --- | --- | --- |
| Concrete synthesis completeness | At every admitted concrete metadata configuration, find attributes exactly when they exist. | Structural solver and its proof |
| Uniform generated-program completeness | One finite generated metadata program implements that solver for all admitted finite ranks. | Symbolic compilation and evaluation correspondence |
| Coverage under declared conditions | Every legal source match satisfying the user's condition is realizable. | A separate implication proof over metadata |

The first two do not automatically establish the third. A complete procedure can correctly report that some source matches have no replacement.

## 2. Mathematical contract and admission boundary

### 2.1 Objects and quantifiers

Let:

- `S = (L, R)` be a source expression and a replacement skeleton.
- `p` contain input ranks/shapes, source attributes, and relevant dtype metadata.
- `X` range over independently variable input tensor values with those shapes.
- `beta` contain every automatic replacement attribute, including intermediate operations.
- `D(S,p)` state the theorem's fragment and metadata-domain assumptions.
- `Legal_L(p)` state source legality.
- `C(p)` be the user's successfully evaluated metadata-only condition, when present.

For legal source metadata, define:

\[
\begin{aligned}
\operatorname{Valid}(S,p,\beta) \iff {}&
\operatorname{Legal}_R(S,p,\beta)\\
&\land\operatorname{Shape}(L,p)=\operatorname{Shape}(R,p,\beta)\\
&\land\forall X:\ \llbracket L\rrbracket_{p,X}
=\llbracket R\rrbracket_{p,\beta,X}.
\end{aligned}
\]

`Legal_R` covers all intermediate operations. Shape is an ordered vector. Element equality is considered only for legal, equally shaped expressions. Required dtype preservation must also be established at the TEPL integration boundary.

The synthesis quantifiers are `exists beta, forall X`. Attributes may depend on metadata but not on tensor values. Do not replace this with `forall X, exists beta`.

### 2.2 Initial admitted fragment F0

The initial complete algorithm should admit exactly an independently specified class:

1. Both expressions are finite trees over the same named independent captures, each occurring exactly once on each side.
2. Binary nodes have the certified generalized-contraction contract: select disjoint paired batching and contracting axes, retain batches, sum contractions, and preserve unpaired axes.
3. Their output order is batch-list order, then left-free axes in child order, then right-free axes in child order.
4. Unary nodes are full axis permutations with no change to tensor values other than coordinate reordering.
5. Every RHS operation has independent, unrestricted automatic descriptors. No pinned fields, shared descriptors, hidden coupling, or fixed RHS permutations in F0.
6. Source attributes are known at the concrete instance and legal. They can select arbitrary legal axis arrangements.
7. Every physical dimension has finite extent at least two. Empty axis sequences and rank-zero tensors are allowed; zero-extent and singleton dimensions are outside F0.
8. Scalar semantics are exact real arithmetic. There are no tensor-value assumptions or side effects.
9. Legality is exactly the certified contract, or additional obligations are proved redundant in the admitted domain. Backend-specific restrictions cannot be ignored.

Initially exclude addition, repeated input occurrences, broadcasting, reshape, diagonal extraction, separate unary reduction, coordinate arithmetic, quantization, and arbitrary scalar calls. These are extension projects with their own proofs.

The restriction on dimensions is needed for semantic distinguishability, not for rank bounding. The theorem covers every finite rank. Equal dimension sizes never identify different logical axes.

The mathematical real-arithmetic theorem is not a strict IEEE floating-point equivalence theorem. Use the existing algebraic policy only with a documented correspondence to these ideal semantics; do not label strict execution complete on the strength of this proof.

### 2.3 Admission must not depend on synthesis success

Define a syntactic semantic-contract grammar and prove its interpretation. An admission checker recognizes this grammar using operation definitions, not operation names.

A renamed user-defined contraction can qualify. A definition with an extra attribute restriction cannot qualify merely because its arithmetic looks like a dot product. Initially reject unfamiliar assertions unless the admission checker has a certified redundancy rule for them.

A conservative recognizer may miss equivalent alternative spellings. Claim completeness for the explicitly admitted class, not for every operation semantically equivalent to a contraction.

Separate schema admission from metadata-domain checking. A rank-polymorphic schema can be structurally admitted while a particular match has a singleton dimension and is outside F0. Such a match is `OutsideDomain`, not `Impossible`.

## 3. Current implementation: reusable parts and gaps

These observations are based on the checked-in source inspected for this plan. Proposed filenames below are design suggestions, not existing modules.

| Existing component | Current behavior | Planned treatment |
| --- | --- | --- |
| [definition.h](../src/core/semantics/definition.h), [axes.h](../src/shared/metadata/axes.h) | Represent element semantics, bindings, outputs, attributes, and assertions. | Extract certified operation contracts from these definitions. |
| [plan.cc](../src/core/semantics/plan.cc) | Builds a semantic `Plan`, calls `synthesize`, and rejects unproved rules outside inspection mode. | Add explicit dispatch and separate coverage handling. |
| [normalize.h](../src/core/semantics/symbolic/normalize.h), [normalize.cc](../src/core/semantics/symbolic/normalize.cc) | `Correspondence` stores anchors, boundaries, and interfaces; `correspond()` derives boundaries and preferred internal interfaces together. | Retain general path; add independent requirements/layout structures for F0. |
| [synthesize.cc](../src/core/semantics/synthesis/synthesize.cc) | Recursively constructs children, selects preferred group orders, assigns holes, and verifies the candidate. | Preserve general path; add requirements-first construction for F0. |
| [index_inference.h](../src/core/semantics/synthesis/index_inference.h) | Explicitly excludes coupled/inverse equations and ambiguous coordinate bases. | Outside the first complete fragment; no need to solve these to establish F0. |
| [context.cc](../src/core/semantics/context.cc) | `Context::require()` records an unestablished obligation and throws. `entails(false)` means unestablished, not refuted. | Preserve this proof discipline. Add a distinct executable applicability representation. |
| [proof/verify.cc](../src/core/semantics/proof/verify.cc) | Checks a candidate in a fresh context, but reuses general correspondence and bounded reasoning. | Keep independent verification; add a structural checker complete for F0 certificates. |
| [ir.h](../src/core/semantics/ir.h) | `ProofStatus` has `Proved` and `Unknown`; `RuleProof` holds constructors and diagnostic guards. | Add separate synthesis/applicability/coverage information. |
| [materialize.cc](../src/core/semantics/materialize.cc), [completion.cc](../src/core/semantics/completion.cc) | Materialize verified constructors into derived descriptors and reject unresolved holes. | Integrate only checked structural results and explicitly authorized guarded semantics. |
| [proof_policy.h](../src/core/semantics/proof/proof_policy.h) | Uses fixed depth/work limits for incomplete proof attempts. | Keep resource failures distinct from unsatisfiability; do not use fixed caps as the theoretical decision procedure. |
| [rule_emitter.cc](../src/codegen/rust/rule_emitter.cc) | Generated Rust checks conditions, derives attributes, then prepares the RHS. | Preserve this order and test applicability before partial constructor operations. |

Important current behavior: `materialize()` can append `proof.guard` to conditions, but `Context::require()` throws when a condition is unproved. This is not an existing exact-guard synthesis feature. Do not turn failed obligations into assumptions or silently append them to obtain a completeness claim.

The existing research note mentions an older `ordered.*` implementation. Such files were not present in the inspected `src/core/semantics` tree. Treat the research arguments and standalone experiments as starting material, not evidence that the proposed pipeline is integrated.

## 4. Proposed internal structures and module boundaries

Add an isolated subdirectory, for example `src/core/semantics/synthesis/ordered/`. Keep the existing generic synthesizer available for unsupported schemas.

| Proposed module | Responsibility | Main proof obligation |
| --- | --- | --- |
| `contract.h/.cc` | Recognize and check operation contracts and schema membership. | Admission soundness |
| `network.h/.cc` | Build the source partition, extents, capture-port maps, and output sequence. | Exact source semantics |
| `requirements.h/.cc` | Compute subtree supports, live sets, and operation roles. | Forced-boundary lemma |
| `ordering.h/.cc` | Propagate required subsequences and produce checks/obstructions. | Necessary and sufficient layout constraints |
| `construct.h/.cc` | Choose compatible layouts and emit positional descriptors. | Constructive sufficiency and legality |
| `certificate.h/.cc` | Independently check source extraction, success witnesses, and failure obstructions. | Checker soundness and acceptance of generated witnesses |
| `compile.h/.cc` | Generate metadata programs implementing the concrete algorithm. | Evaluation correspondence across ranks |

Start with explicit concrete types, separate from symbolic `S` expressions:

```cpp
// Sketch only; choose repository-consistent ID and container types.
struct InputPort { CaptureId capture; std::size_t position; };
struct LogicalAxis {
  AxisId id;
  Extent extent;
  std::vector<InputPort> ports;
  CaptureSet support;
};
struct SourceNetwork {
  std::vector<LogicalAxis> axes;
  std::map<CaptureId, AxisSequence> input_layouts;
  AxisSequence output;
};
struct NodeRequirements {
  AxisSet live;
  AxisSet batch, contract, left_free, right_free;
  AxisSequence required_order;  // A subsequence, not a full layout.
};
struct NodeSolution {
  AxisSequence layout;
  ConcreteDescriptor attributes;
};
```

Do not represent logical identity only by support sets: two different axes can occur in exactly the same inputs. Preserve the full partition of named physical ports. Extents are annotations, not identity keys.

Use deterministic logical labels, such as the least input port in each partition class under a fixed capture/position order. This is a selection convention; semantic proofs should be invariant under consistent renaming of internal labels.

Define separate result categories:

```text
Concrete instance:
  Solved(attributes, certificate)
  Impossible(order_obstruction)
  Unsupported(schema_reason)
  OutsideDomain(metadata_reason)
  InvalidSource(reason)
  ResourceLimit(reason)

Schema compilation:
  Compiled(domain_predicate, applicability_program, attribute_program, proof)
  Unsupported(reason)
  ResourceLimit(reason)

Declared-condition coverage:
  Established
  Refuted(concrete_legal_counterexample)
  Unknown(reason)
```

Malformed internal certificates are implementation errors, not evidence of mathematical impossibility. Missing runtime metadata is also distinct from an impossible replacement.

## 5. Refactor stages and algorithms

### 5.1 Admission and dispatch

1. Preserve the current synthesizer as the generic implementation.
2. Add operation-contract recognition and schema checks before selecting the complete path.
3. Verify capture occurrence counts, tree structure, attribute independence, field coverage, and arithmetic policy.
4. Record domain obligations such as all extents being at least two.
5. Route unsupported cases to the existing general path without advertising completeness.
6. Route admitted cases through the structural pipeline. A proven negative instance should not be relabeled as a generic proof-search failure.

For concrete source metadata, pins or sharing on the source side do not by themselves break the algorithm: the actual legal source attributes are already known. Restrictions on universal schema coverage require separate analysis. F0's unrestricted-attribute requirement applies to the unknown RHS descriptors.

Do not change the public meaning of `auto` during this initial refactor. The first integration should preserve declared-condition coverage requirements. Exact generated applicability can initially be exposed through inspection and internal testing.

### 5.2 Exact source network extraction

Traverse the source tree in producer order:

1. Allocate one port for each physical input axis, including its input identity and position.
2. Maintain each intermediate's ordered live-port references.
3. At a legal source contraction, identify paired ports by logical equivalence; retain batch classes and remove contracted classes from the live output.
4. Preserve left-free and right-free physical relative order; apply the declared batch-list order.
5. At a transpose, reorder the live sequence using the source permutation convention.
6. Canonicalize equivalence classes and update all recorded references. Retain contracted classes in the global network even though they no longer appear in live outputs.
7. Record each class's physical ports, common extent, input support, and final output membership/position.

An explicit disjoint-set representation is suitable for the first concrete implementation. Do not use a shape comparison to merge classes.

Check invariants: each class has at most one port per input; every intermediate contains each live class at most once; each summed class has support in at least two inputs; every class occurs in an input read. These facts will be used in the proofs.

### 5.3 Forced axis requirements

For every RHS subtree `v`, compute its capture set `U_v`. Let `F` be the source's ordered final output, and `set(F)` its set.

\[
P_v=\partial U_v=
\{a\mid S(a)\cap U_v\ne\varnothing
\land(a\in\operatorname{set}(F)\lor S(a)\nsubseteq U_v)\}.
\]

At a contraction with left/right live sets `L`, `R` and parent live set `P`:

\[
B=L\cap R\cap P,\qquad K=(L\cap R)\setminus P,
\]
\[
L_f=L\setminus R,\qquad R_f=R\setminus L.
\]

Check or derive the structural identities `P = B union L_f union R_f` and pairwise disjointness of these output roles. Input axes partition into batch, contract, and free roles. These are exact logical-axis statements, not choices of list order.

No attributes are assigned in this pass. The generic `correspond()` may remain unchanged initially; the complete path should not depend on its preferred `interfaces` order or ambiguity heuristics.

### 5.4 Top-down ordering constraints

Start with `Q_root = F`. Maintain that `Q_v` is a distinct subsequence requirement over `P_v`.

```text
constrain(v, Q):
    record Q as required_order[v]
    assert distinct(Q) and set(Q) subset live[v]

    if v is an input:
        add Subsequence(Q, physical_logical_layout[v])

    if v is an unrestricted automatic transpose:
        constrain(child(v), [])

    if v is a contraction:
        QB = filter(Q, batch[v])
        QL = filter(Q, left_free[v])
        QR = filter(Q, right_free[v])
        add EqualSequence(Q, concat(QB, QL, QR))
        constrain(left(v), QL)
        constrain(right(v), QR)
```

The accumulated conjunction is `Phi_S(p)`, defined on legal metadata in F0. The leaf check can be implemented as `Q == filter(input_layout, set(Q))`, using distinctness and subset invariants.

Batch requirements do not propagate into the children: unrestricted paired position lists can select batch axes in any desired common order. Transposes stop order propagation because their unknown permutation can impose the parent's required order.

At concrete metadata, evaluate checks exactly. A dot failure supplies two axes in descending block order; a leaf failure supplies an inverted pair. Include the node, axes, required order, and derived roles in a negative certificate. The checker must reconstruct why the requirement reaches that node.

For symbolic metadata, build executable sequence predicates. Do not call the incomplete `entails()` routine and interpret an unproved result as `false`.

### 5.5 Bottom-up construction

Define an order-extension helper:

\[
\operatorname{extend}(Q,U,T)=Q\mathbin{+\!\!+}
\operatorname{filter}(T,U\setminus\operatorname{set}(Q)),
\]

where `Q` is distinct and contained in `U`, and `T` enumerates `U` exactly once. This returns a permutation of `U` containing `Q` as a subsequence. A deterministic order of logical labels supplies `T` when needed.

```text
construct(v):
    if input:
        return its physical logical-axis layout

    if transpose:
        child_layout = construct(child(v))
        output = extend(required_order[v], live[v], child_layout)
        emit permutation positions(child_layout, output)
        return output

    if contraction:
        left_layout  = construct(left(v))
        right_layout = construct(right(v))
        QB = filter(required_order[v], batch[v])
        batch_order = extend(QB, batch[v], stable(batch[v]))
        contract_order = stable(contract[v])

        emit left_batch  = positions(left_layout,  batch_order)
        emit right_batch = positions(right_layout, batch_order)
        emit left_contract  = positions(left_layout,  contract_order)
        emit right_contract = positions(right_layout, contract_order)

        return concat(batch_order,
                      filter(left_layout, left_free[v]),
                      filter(right_layout, right_free[v]))
```

The transpose line assumes output axes equal `gather(child_layout, permutation)`. Admission must record the actual descriptor convention and provide a proved adapter if it is inverse-oriented.

Reuse TEPL's `positions`, filtering, and descriptor materialization only where their preconditions have been established. The two contracting lists use the same logical-axis sequence; independently sorting physical positions on the two sides can change pairings and is incorrect.

Replace the complete path's use of local preferred orders in `Constructor::build()` with this construction. It is not necessary to delete that behavior from the general path.

### 5.6 Independent certificates

Success certificates should expose the source partition, subtree boundaries, local roles, order requirements, chosen layouts, and concrete descriptor mappings. A checker should recompute inexpensive structural facts rather than accept the producer's assertions as premises.

For concrete witnesses, independently interpret the completed RHS descriptors, rebuild its network and ordered output, validate all operation contracts, and compare with the source. This verifies success without rerunning synthesis choices.

For negative certificates, check the source representation, boundary derivation, propagation of the required subsequence, and the local inversion. Merely failing to reconstruct the producer's candidate is not an impossibility proof.

For symbolic programs, use a schema-level derivation of the primitive rules and their preconditions. Testing one instantiated certificate does not certify every metadata instance.

During development, compare the new checker with the general verifier on overlapping supported cases. Eventually an F0 result must not require successful bounded general correspondence/proof search: that would reintroduce incompleteness after construction.

### 5.7 Uniform symbolic compilation

Prove the concrete algorithm first, then implement a compiler that residualizes it with respect to the fixed rewrite schema. Ranks, dimensions, and source descriptors remain runtime metadata inputs.

Preferred implementation route:

1. Build an explicit metadata procedure for source-axis extraction, boundaries, order checks, and construction.
2. Represent arbitrary-length groups with executable list/set operations; reuse existing internal intrinsics where their semantics suffice.
3. Generate `Phi_S` and `f_S` from the same requirements representation so they cannot drift apart.
4. Make primitive preconditions explicit: distinctness, membership, valid permutations, bounds, and extent consistency.
5. Prove evaluation correspondence for every emitted operation and compose it over the generated procedure.

**Critical feasibility checkpoint:** existing symbolic group mappings must be shown to represent every source port equivalence and required refinement in F0. Finite ranks are unbounded, and arbitrary source permutations cannot simply be replaced by a finite set of axis-role labels. Do not assume current bounded mapping search supplies a complete representation.

If existing completed metadata expressions cannot express the whole admitted algorithm, choose one explicit resolution before claiming schema-level completeness:

- Add typed internal sequence/provenance operations with complete concrete implementations in both native and Rust runtimes; or
- Introduce a checked structural metadata-program representation, interpreted/generated at runtime, with exact source partition and layout operations.

The latter remains metadata computation, not tensor-value computation. Keep it separate from the compiler's mutable proof `Context`. This is a genuine additional engineering task, not functionality already provided by an intrinsic named in this plan.

Existing [AxisContext](../runtime/rust/src/axes.rs) creates operation-local axis identities. Cross-operation source provenance requires a proved representation/transport scheme; local identity creation alone is not the source-network algorithm.

Completed IR must contain no inference holes. Adding an executable operation may require updates to intrinsic catalogs, lowering, validation/serialization, the native evaluator, Rust runtime, and code generation. Prove source-input and JSON-input paths have the same semantics.

### 5.8 Guards, coverage, and materialization

Keep three objects separate:

- `D`: the declared domain of the theorem.
- `Phi`: exact replacement applicability within that domain.
- `C`: the user's existing rule condition.

Preserve current full-coverage behavior by requiring `Legal_L and C => D and Phi` before accepting an F0-only implementation of an ordinary rule. If `D` does not hold for all matches, route other matches to a sound fallback or require an explicit domain restriction; do not classify them as unrealizable.

A future explicit guarded-inference mode may apply under `C and D and Phi`, with its semantics reported to the user. It must be a deliberate language/API feature. Do not silently strengthen an existing rule's condition to make a completeness claim true.

The checker may verify a conditional theorem under `Legal_L and D and C and Phi`, but it must expose that theorem accurately. A separate coverage proof discharges `C => D and Phi` when unconditional coverage under `C` is promised.

Evaluate source-domain checks and applicability before partial descriptor calculations. Generated Rust already calls `check_where()` before `derive()`; verify the native evaluator's order too. Preserve short-circuit behavior and user local-binding dependencies.

An applicability predicate must be total on its promised legal domain. Refuted applicability means a skipped match in an explicitly guarded mode, not an evaluator exception. Resource exhaustion and missing metadata retain separate statuses.

### 5.9 File integration checklist

| File or area | Planned change |
| --- | --- |
| `src/core/semantics/synthesis/ordered/` (new) | Add the structural contracts, solver, certificates, and compiler in stages. |
| `synthesis/synthesize.cc` | Introduce dispatch; preserve generic fallback and provenance of the selected guarantee. |
| `symbolic/normalize.h/.cc` | Initially leave generic correspondence intact; share helpers only after proving their needed contracts. |
| `ir.h`, `options.h`, `print.cc`, `report.cc` | Represent applicability/coverage separately; report unsupported, outside-domain, resource, and impossibility outcomes accurately. |
| `plan.cc` | Integrate admission, synthesis-program verification, and declared-condition coverage. |
| `proof/verify.cc` and new structural checker | Retain fresh-context checking; add a separately justified route for F0 certificates. |
| `materialize.cc`, `completion.cc` | Materialize verified fields/programs; never treat diagnostic obligations as authorized guards. |
| `src/shared/metadata/`, `src/shared/builtins/intrinsics/` | Extend executable representation only if required by the uniform compiler. |
| `src/core/lowering/`, `src/executable/`, `schema/` | Update lowering, validation, and serialized form if executable constructs change. |
| `src/rewrite/`, `runtime/rust/`, `src/codegen/rust/` | Implement/test matching execution semantics, partial-operation preconditions, and guard-before-construction behavior. |
| `src/core/semantics/BUILD.bazel`, `tests/BUILD.bazel` | Register modules/tests; audit broad `proof` globs before adding a dedicated library to avoid compiling sources twice. |

The supported generated backend is Rust according to the current codegen documentation. Old C++ generation files are not evidence of a supported backend requiring this feature.

## 6. Mathematical proof development

### 6.1 Definitions to fix before proving the algorithm

Use named physical ports `(capture, physical axis position)` as the common universe. A network consists of a partition of those ports, extents attached to its classes, and an ordered list `F` of distinct output classes. All classes not in `F` are summed. Include ordered output-position attachments when comparing networks.

Fix the input shapes when comparing expressions. Network equality means equality of the port partition and ordered output attachments, up to internal class-label renaming; it is stronger than equal support sets or equal extent vectors.

For each input `t`, let `A_t` be its ordered sequence of logical classes. The network denotes:

\[
\llbracket N\rrbracket(X)[o]
=\sum_{\rho\in\prod_{a\notin\operatorname{set}(F)}[d_a]}
\prod_t X_t\bigl((\eta_{o,\rho}(a))_{a\in A_t}\bigr),
\]

where `[d] = {0,...,d-1}`, `o` fixes the coordinates of `F` in order, and `rho` assigns summed coordinates. Empty products and sums over an empty axis sequence have their usual meanings; an empty axis sequence has one coordinate assignment.

The admission assumptions imply that every logical class occurs in an input and each input contains any class at most once. Prove those invariants rather than assuming them for arbitrary programs.

### 6.2 Lemma A: admission and source lowering preserve semantics

**Statement.** A checked admitted definition has exactly its certified contraction/permutation semantics and legality. Lowering a legal admitted source tree produces a network with the same ordered shape and value at every output coordinate.

**Proof plan.** First prove each supported definition-template rule, including its attribute-field adapter and assertions. Then induct over the source tree:

- A leaf creates its physical ports and reads the same ordered coordinates.
- A permutation changes only output-coordinate order.
- A contraction identifies the selected paired coordinates, retains batches, and sums contractions. Legal paired extents agree. Independent child reductions can be combined by finite-sum reindexing and distributivity under exact arithmetic.
- The output construction follows the certified batch/left-free/right-free order.

Record the intermediate coordinate correspondence as an invariant. This proof is relative to declared primitive semantics; the backend implementation must satisfy those declarations separately.

**Code connection:** `contract`, `network`, source metadata evaluation, and source adapters.

### 6.3 Lemma B: exactness of the network representation

**Statement.** For two legal F0 expressions over the same independent captures and fixed input shapes,

\[
\forall X:\llbracket E_1\rrbracket(X)=\llbracket E_2\rrbracket(X)
\quad\Longleftrightarrow\quad N(E_1)=N(E_2).
\]

Equality includes ordered output shape. Different output rank or shape already refutes equivalence.

**Sufficiency.** Apply Lemma A and the network denotation: identical partitions and output attachments generate the same finite sum for each output coordinate.

**Necessity.** Fix an output coordinate and expand into polynomials over independent entry variables `z_(t,i)`. Every monomial chooses one entry of every named input. Its coefficient is one exactly when those selected coordinates obey the network's class equalities and output attachments, and zero otherwise. The selected entries determine every summed coordinate because no class is unobserved.

If partitions or output attachments differ, include output positions as named ports. Find two ports joined in one representation and separated in the other. Assign the separated classes coordinates zero and one and consistently extend the remaining classes. Those coordinates exist because all extents are at least two. This gives an entry tuple and output coordinate accepted by one representation and excluded by the other.

Equivalently, set each input to the one-hot tensor at its selected entry. The two expressions then differ at that output coordinate. Independent captures and the absence of unused summed classes prevent accidental cancellation or multiplicity. Therefore universally equal expressions cannot have different networks.

**Review obligations:** explicitly cover output-port attachment differences, scalar inputs/outputs, equal-sized distinct axes, and the impossibility of a class containing two ports of one input in F0. Do not reuse this necessity proof for repeated inputs, singleton extents, or additive expressions.

**Code connection:** this lemma justifies treating logical-axis relationships as necessary semantic facts rather than a sufficient matching heuristic.

### 6.4 Lemma C: every valid RHS has the forced boundaries

**Statement.** In any valid RHS realization, the live logical classes at a subtree with input set `U` are exactly `partial U` from Section 5.3.

**Required inclusion.** If a class is needed by an outside input or by the final output, it cannot be eliminated inside the subtree. A reduction would destroy the required coordinate connection. Transposes cannot recreate it, and identifying a different class later changes the network, which Lemma B rules out.

Also justify that a subtree cannot keep two disconnected local pieces of the same final class: after their least common ancestor, a binary contraction cannot identify two axes already inside its single child. Thus all in-subtree occurrences of a live class must already have been joined into one boundary axis.

**Reverse inclusion.** Let a class be summed and have all its support inside `U`. Its support has at least two leaves. At the lowest common ancestor of those leaves, both children carry the class. Above that node no sibling carries it, so no later binary contraction could eliminate it by a correct pairing. It must be reduced there, and cannot survive to the subtree boundary. A permutation does not eliminate axes.

Thus the boundary is exactly forced, not just a lower bound. The local role formulas for batch, contraction, and free axes follow immediately.

**Code connection:** `requirements`; this is the key justification for avoiding a search over pairings and reduction placement.

### 6.5 Lemma D: necessity of ordering constraints

**Statement.** Every valid RHS assignment satisfies all constraints generated by the top-down pass.

**Proof plan.** At the root, the completed RHS must have order `F`, so it satisfies `Q_root = F`. Propagate the following implication down the tree: if the realized layout contains `Q_v`, then the generated local check holds and the relevant child layouts contain the propagated requirements.

- At a contraction, every output layout has a batch block, followed by left-free axes, followed by right-free axes. Any required subsequence must respect that block order. This proves `Q = QB ++ QL ++ QR`.
- Free-axis order is inherited from each child. Therefore `QL` and `QR` must be subsequences of their child layouts.
- Batch-list order is selectable independently of the physical locations in either child, so no additional child-order requirement is necessary for batches.
- At an automatic transpose, no restriction on child order follows from the required parent order.
- At a leaf, the layout is fixed, so the required subsequence must occur in that physical order.

Together with Lemmas B and C:

\[
\forall\beta:\operatorname{Valid}(S,p,\beta)\Longrightarrow\Phi_S(p).
\]

**Code connection:** `ordering`; no constraint may encode an arbitrary preferred layout as if it were forced.

### 6.6 Lemma E: order extension and positional realization

Prove the following finite-sequence facts independently:

1. `extend(Q,U,T)` lists every axis of `U` exactly once and contains `Q` as a subsequence under its stated preconditions.
2. If `T` is distinct and contains every member of `Q`, then `positions(T,Q)` is defined and `gather(T, positions(T,Q)) = Q`.
3. Applying `positions` to disjoint logical-axis sets gives disjoint positional selections in the same distinct layout.
4. For two layouts containing the same ordered logical list `K`, selecting their respective `positions(...,K)` pairs precisely the corresponding logical indices.
5. Such paired indices have equal extents because they are occurrences of the same certified source class.
6. A full reordering yields a valid permutation, including the empty/rank-zero case.

Specify whether each primitive is total or partial, and prove its preconditions at every emitted call. Distinguish arbitrary mathematical integers from checked machine representations in implementation refinement.

**Code connection:** reuse or prove contracts for filtering, `positions`, inverse/permutation adapters, set differences, and descriptor construction. Tests of these functions complement but do not replace these lemmas.

### 6.7 Lemma F: constructive sufficiency

**Statement.** If `Phi_S(p)` holds, bottom-up construction yields valid replacement attributes.

Use a stronger induction invariant than final equality alone. For each RHS subtree `v`, the constructed operation tree:

1. is legal;
2. realizes the source logical connections among inputs in `U_v`;
3. sums exactly those source classes internal to `U_v` that are not final outputs;
4. has exactly `P_v` as its live axis set;
5. has a distinct output layout containing `Q_v` as a subsequence, with the correct extents.

**Input case.** A leaf has its prescribed physical input layout. Its boundary contains all its axes: an internal summed class cannot have support in only one input. The leaf ordering check establishes the subsequence requirement.

**Permutation case.** By induction the child realizes the correct partial network. Lemma E supplies a full permutation extending `Q_v`; reordering preserves the partial network and yields the required output layout.

**Contraction case.** By induction both children satisfy their requirements. Pair common logical classes. Retain `B` and reduce `K`. Lemmas C and E give exactly the required parent boundary, legal disjoint positional lists, and matching paired extents. Choose a batch order extending `QB`. The child free subsequences extend `QL` and `QR`; the successful block check ensures their concatenation contains `Q_v`.

Each internal summed class is eliminated when its full support first joins; no class required outside or at the output is eliminated. Hence the parent realizes the required partial network. This proves the invariant.

At the root, `P_root = set(F)` and `Q_root = F`. A distinct layout over that set containing the full sequence `F` must equal `F`. The complete network equals the source's, so Lemma B yields equality for all input tensors.

Therefore:

\[
\Phi_S(p)\Longrightarrow
f_S(p)\downarrow\ \land\operatorname{Valid}(S,p,f_S(p)).
\]

**Code connection:** `construct`; the proof also establishes that deterministic local choices do not remove solutions after the constraints have passed.

### 6.8 Lemma G: termination and exact decisions

For a finite source/RHS tree and finite ranks:

- Source extraction processes finitely many nodes, ports, and legal pairings.
- Boundary computation performs finite support/set operations.
- Requirement propagation visits the finite RHS tree and filters finite lists.
- Every order check is a finite exact sequence comparison/subsequence check.
- Construction visits the finite tree and performs finite position lookups and descriptor assembly.

Use induction on tree size for recursive passes and list length for primitive loops. Implement iterative traversals if necessary to avoid treating a fixed recursion-depth cap as a semantic limitation.

The procedure does not use an incomplete SMT query as its final existence decision. In F0, global layout feasibility is exactly the recursively propagated sequence checks; Lemma F is the proof of that fact.

A straightforward concrete implementation is polynomial in expression size and total input-axis count, with extent-bit costs included for metadata checks. Derive and document a bound for the actual data structures before stating a tight complexity theorem. This is metadata complexity, not tensor-element evaluation cost.

Fixed resource caps can interrupt a terminating algorithm. State the theorem for the mathematical algorithm or an implementation with sufficient resources; report interruptions separately. A larger cap alone is not a completeness argument.

### 6.9 Main theorem: complete concrete attribute synthesis

**Theorem.** For every admitted F0 schema `S` and legal source configuration `p` satisfying `D(S,p)`, the structural algorithm terminates. It returns `Solved(beta)` exactly when a valid replacement attribute assignment exists, and every returned assignment is valid. Otherwise it returns a checked impossibility obstruction.

**Proof.** If some valid assignment exists, Lemmas B–D imply every generated constraint holds. Lemmas E–F construct a valid assignment, and Lemma G guarantees termination. Conversely, a returned assignment is valid by Lemma F and the positional legality lemmas. If a local order check fails, Lemma D excludes every valid assignment. The check's finite inversion witnesses that failure and can be independently verified. Exhaustiveness of these branches follows from exact constraint evaluation. End of proof.

Equivalently, within the theorem's domain:

\[
\boxed{
\exists\beta\,\operatorname{Valid}(S,p,\beta)
\iff\Phi_S(p)
\iff\operatorname{Infer}(S,p)\text{ succeeds}.
}
\]

This is a constructive theorem: it gives both an existence characterization and an actual witness function. It does not say every admitted RHS skeleton is realizable for every source configuration.

### 6.10 Corollary: certificates and the implementation boundary

Prove two checker properties:

\[
\operatorname{CheckSuccess}(S,p,\beta,w)=\mathrm{true}
\Longrightarrow\operatorname{Valid}(S,p,\beta),
\]

\[
\operatorname{Infer}(S,p)=\operatorname{Solved}(\beta,w)
\Longrightarrow\operatorname{CheckSuccess}(S,p,\beta,w)=\mathrm{true}.
\]

Give analogous properties for generated negative witnesses. Checker soundness alone is insufficient for end-to-end completeness: a checker could reject all witnesses.

Separate mathematical acceptance from resource-limited execution. Audit that schema admission, source extraction, construction, and verification all use compatible attribute/permutation conventions.

### 6.11 Theorem: generated metadata programs implement the solver

Let `Compile(S)` produce a finite program `P_S`. On every admitted legal `p`, require:

\[
\operatorname{Eval}(P_S,p)
\simeq\operatorname{Infer}(S,p),
\]

where equivalence means the same applicability result and corresponding valid attributes/certificates. If both use the same stable-order convention, strengthen this to identical attribute tuples. Internal logical-label renaming is not an observable difference.

Prove:

1. Source metadata and physical ports are represented faithfully, including arbitrary source permutations and group lengths.
2. Every primitive implements the mathematical set/sequence operation and checks its domains.
3. Symbolic boundary and ordering expressions evaluate to the concrete requirements.
4. Generated control flow evaluates applicability before partial construction.
5. Attribute emission uses the certified field adapters and matches the concrete constructor.
6. Materialization, executable lowering, native evaluation, and Rust generation preserve the metadata program's meaning.

The resulting rank-polymorphic contract is:

\[
\forall p:\operatorname{Legal}_L(p)\land D(S,p)
\Longrightarrow
\left(G_S(p)\iff\exists\beta\,\operatorname{Valid}(S,p,\beta)\right),
\]

\[
\operatorname{Legal}_L(p)\land D(S,p)\land G_S(p)
\Longrightarrow f_S(p)\downarrow\land\operatorname{Valid}(S,p,f_S(p)).
\]

The compiled program is finite even though its list inputs have unbounded finite lengths. Prove this through the metadata instruction semantics, not by enumerating ranks. If the compiler emits only a restricted subset of constructors, characterize that subset and prove it can express this particular algorithm.

### 6.12 Separate theorem: coverage under a supplied condition

To accept an ordinary rule with condition `C` through this complete path, establish:

\[
\forall p:\operatorname{Legal}_L(p)\land C(p)
\Longrightarrow D(S,p)\land G_S(p).
\]

Combine this with the generated-program theorem to obtain the existing desired legality/equivalence guarantee under `C`.

For arbitrary metadata predicates, do not assume the current implication prover is complete. Either define an additional decidable guard fragment and prove its solver, or report coverage as unknown when it cannot be established. A complete match-time attribute procedure and an incomplete schema-coverage prover are distinct claims and can coexist.

The rank-two obstruction/cutoff in the research notes is optional later work. It is not needed for concrete synthesis completeness, and arbitrary user guards need not be preserved by deleting axes.

## 7. Proof-to-code traceability

Maintain this table as code and formal definitions evolve:

| Property | Producer | Independent check or evidence | Mathematical result |
| --- | --- | --- | --- |
| Operation belongs to F0 | Contract extractor | Contract-template derivation and assertion audit | Lemma A |
| Source network denotes source | Network extractor | Source replay/provenance certificate | Lemmas A–B |
| Axis roles are forced | Boundary pass | Support and least-common-ancestor argument | Lemma C |
| No valid ordering is excluded | Top-down order pass | Role-word/subsequence derivation | Lemma D |
| Chosen orders are realizable | Constructor | Extension and position contracts | Lemma E |
| Completed RHS is legal/equivalent | Constructor | RHS replay and network comparison | Lemma F |
| No admitted instance is left undecided | Exact sequence checks | Termination measures and exhaustive case split | Lemma G |
| Checked solver is complete | Constructor/checker pair | Checker accepts generated witnesses | Section 6.10 |
| Generated programs preserve decisions | Compiler and backend | Evaluation correspondence | Section 6.11 |
| Original user condition is covered | Coverage prover | Checked implication or explicit counterexample | Section 6.12 |

For a paper proof, state primitive contracts and prove the structural algorithm rigorously. For a stronger mechanized result, formalize finite ports, partitions, lists, network denotation, and the constructor in a proof assistant, then address implementation refinement separately. Mechanizing an abstract algorithm does not automatically verify the C++ or Rust implementations.

## 8. Validation plan

### 8.1 Structural regression cases

Add dedicated C++ tests for the new solver and certificate checker. Cover at least:

| Case | Expected result |
| --- | --- |
| Ordinary matrix reassociation | Construct a valid assignment. |
| Arbitrary-rank contractions with several batch and reduction axes | Construct correct paired lists and ordered outputs. |
| Equal-sized distinct logical axes | Preserve identities; do not infer equivalence from shape alone. |
| Scalar inputs and outputs; empty batch/contract/free groups | Legal construction with correct empty sequences. |
| Required output blocks in descending role order | Checked impossibility. |
| Required leaf order reversed without a transpose | Checked impossibility. |
| Automatic internal transpose needed to satisfy a parent | Construct the needed internal permutation. |
| Two automatic transposes around an input | Choose a valid assignment despite nonuniqueness. |
| Multiple reduction axes with simultaneous list reorderings | Recognize equivalent exact-arithmetic choices. |
| Shared/pinned RHS attributes, repeated captures, extra assertions | Outside F0 admission; no false impossibility claim. |
| Singleton or zero-sized dimensions | Outside metadata domain; retain generic behavior separately. |
| Strict arithmetic policy | Do not apply the ideal-arithmetic completeness theorem. |
| Exhausted budget or missing metadata | Distinct nonlogical failure status. |

One internal-transpose case is:

\[
L[j,i,k]=A[i,j]C[k],\qquad
R=\operatorname{dot}[\mathrm{auto}](\operatorname{transpose}[\mathrm{auto}](A),C).
\]

The valid realization reverses the two axes of `A`, then forms an outer product. Use this to ensure partial intermediate order requirements are propagated. It is a proposed regression case, not a confirmed current-code failure.

A negative case is:

\[
L[j,i]=A[i]B[j]C[j],\qquad
R=\operatorname{dot}[\mathrm{auto}](A,\operatorname{dot}[\mathrm{auto}](B,C)).
\]

The root must place left-free `i` before right-free `j`, contradicting required `[j,i]`. An unrestricted root transpose repairs this instance. Verify both outcomes.

### 8.2 Exhaustive differential oracle

Use [check_ordered_contractions.py](../examples/research/check_ordered_contractions.py) as existing research evidence, not as the implementation to copy into the checker. It enumerates legal positional pairings and batch orders and compares networks. It quotients simultaneous contraction-list reorderings, which is justified only under the exact arithmetic policy.

Compare the new C++ solver against independent exhaustive attribute enumeration on small legal sources and RHS trees. For every case:

1. The solver's existence decision agrees with the oracle.
2. Returned descriptors are independently legal and reconstruct the exact source network/order.
3. Failure certificates independently check and the oracle finds no solution.
4. Deterministic selection is stable across repeated executions.

The network oracle shares the mathematical abstraction in Lemma B. Add exact small-tensor evaluation, including one-hot witnesses, to test semantic lowering separately. Use exact integers/rationals with no overflow, not tolerance-based floating-point equality as the correctness oracle.

### 8.3 Compiler and runtime validation

For each generated metadata program, compare concrete solver output with native evaluation and generated Rust at varying ranks, source permutations, batch/contract counts, and extents. Test source and serialized-JSON compilation paths.

Include deliberately infeasible metadata that would make a positional lookup invalid if construction ran too early. The applicability check must reject it before that lookup. Verify source conditions and local bindings retain their evaluation order.

Mutation-test certificates: alter one pairing, reduction role, output position, extent, or claimed failure role and require rejection. A certificate checker that merely repeats a shared unverified assumption does not supply independent assurance.

### 8.4 Commands and interpretation

Existing regression targets relevant after code changes:

```sh
bazel test //tests:semantic_test //tests:reasoning_test //tests:axis_interface_test
bazel test //tests:codegen_test //tests:rust_codegen_test
cargo test --manifest-path runtime/rust/Cargo.toml --locked
```

Register new dedicated tests in `tests/BUILD.bazel` as the modules are added. Run focused tests first; run backend checks when materialization or execution changes.

Existing standalone research experiments, from the repository root:

```sh
python3 examples/research/check_ordered_contractions.py --transpose-cases
python3 examples/research/check_ordered_contractions.py --inputs 4 --max-rank 1 --max-total-axes 4 --transpose-cases
```

These commands are planned validation, not results of this document-writing task. Small-model agreement cannot establish arbitrary-rank completeness; the lemmas do that.

## 9. Implementation milestones and exit criteria

### M0 — Freeze F0 and the mathematical interfaces

- Write the exact operation-contract grammar and metadata domain.
- Define source networks, validity, boundary requirements, and attribute conventions.
- Identify backend legality/dtype assertions that require admission evidence.
- Separate concrete inference, symbolic compilation, and coverage statements in documentation.

**Exit:** every unsupported feature has a clear admission reason; no definition of F0 depends on inference success. Review Lemma B's distinguishability assumptions before broad implementation work.

### M1 — Implement admission and source networks

- Add `contract` and `network` modules with concrete data structures.
- Independently replay source operations and validate physical-port provenance.
- Add renamed-operation tests and rejection tests for extra restrictions.
- Develop Lemmas A and B alongside the implementation.

**Exit:** exact source networks agree with independent concrete semantics on small cases; the hand proofs state all premises and have been reviewed.

### M2 — Implement exact requirements, ordering, and construction

- Add boundary/role computation, top-down subsequence propagation, and bottom-up construction.
- Add explicit negative outcomes and local ordering witnesses.
- Compare with exhaustive positional attribute enumeration.
- Complete proofs of Lemmas C–G and the concrete synthesis theorem.

**Exit:** every in-domain concrete case terminates with a valid assignment or checked obstruction when resources suffice. General bounded proof search is not required for this decision.

### M3 — Integrate certificates and inspection

- Add success/failure checking and synthesis-status reporting.
- Integrate the complete path without changing the general fallback's claims.
- Expose exact applicability separately from declared-condition coverage.
- Prove the checker accepts every witness produced by the constructor.

**Exit:** users can distinguish impossible matches, unsupported schemas, outside-domain metadata, and resource limits. No unchecked constructors enter `VerifiedProgram`.

### M4 — Generate uniform metadata programs

- Prototype compilation of source equivalences and arbitrary-length ordering operations.
- Resolve the representability checkpoint in Section 5.7; extend metadata IR/runtime if needed.
- Implement domain/applicability checks before attribute evaluation.
- Prove primitive and whole-program evaluation correspondence.

**Exit:** the same compiled schema agrees with the concrete solver across arbitrary finite ranks by proof, with differential runtime evidence. If only a smaller compilable subclass is established, document that narrower theorem instead of claiming all F0 schemas.

### M5 — Preserve coverage semantics and backend behavior

- Integrate `C => D and Phi` obligations into ordinary rule acceptance.
- Keep unknown coverage explicit; report a refutation only with a checked legal counterexample.
- Add any explicitly designed guarded-inference mode separately.
- Validate native execution, Rust generation, and serialization.

**Exit:** accepted rules retain the promised applicability semantics; generated execution preserves legality and equivalence. The scope of coverage completeness is stated independently.

### M6 — Extend one dimension at a time

Recommended first extension: sums of components distinguishable by capture multidegree, where each component is read-once and unknown attributes are private to its branch. Prove component decomposition and common output legality before applying the F0 solver branchwise.

Later consider shared descriptors, pinned attributes, singleton/empty extents, or unary reductions. Each changes a specific lemma and may require a new solver. Shared unknowns must remain joint variables; normalization must not duplicate one unknown into independent choices.

Do not delay the core theorem for the rank-cutoff, solution-counting, or optimal-transpose-repair projects. They are follow-on results.

## 10. Risks, decisions, and completion checklist

| Risk | Required response |
| --- | --- |
| Axis abstraction is only sound, not exact | Strengthen Lemma B or narrow F0; otherwise negative answers are unjustified. |
| Extra operation legality excludes a constructed descriptor | Include it in the certified contract or reject admission. |
| Preferred intermediate layouts overconstrain solutions | Store requirements as subsequences; choose full layouts only during construction. |
| Symbolic source groups cannot represent arbitrary pairings | Add a proved representation or state a narrower generated-program fragment. |
| General verifier rejects valid structural witnesses | Use a proved structural checker; do not make incomplete proof search mandatory. |
| Guard synthesis changes the original rule's meaning | Preserve coverage checking; introduce guarded semantics explicitly if desired. |
| Fixed bounds are mistaken for completeness | Distinguish resource-limited execution from the terminating mathematical algorithm. |
| Runtime axis identity is confused with equal extents | Preserve and verify cross-operation physical-port provenance. |
| Tests share the same abstraction bug | Combine structural oracle checks, exact tensor evaluation, and independent proof review. |

Before claiming the concrete theorem:

- [ ] F0 and the metadata domain are independently defined.
- [ ] Admission validates both semantics and legality.
- [ ] Source representation exactness is proved in both directions.
- [ ] Forced boundaries and pairings are proved necessary.
- [ ] Ordering checks are proved necessary for every realization.
- [ ] Construction is proved legal and sufficient by structural induction.
- [ ] Concrete checking and construction terminate without fixed-rank assumptions.
- [ ] Generated success and failure certificates are accepted by a sound checker.
- [ ] Resource failures cannot be reported as impossibility.

Before claiming completeness of generated TEPL inference:

- [ ] The metadata language expresses the algorithm for the claimed schema class.
- [ ] Source-axis representation and emitted primitive evaluation correspond to the concrete model.
- [ ] Materialization, native execution, and Rust code generation preserve that meaning.
- [ ] Conditional applicability is distinguished from universal coverage under user conditions.
- [ ] Arithmetic, dtype, dimension, and resource assumptions are visible in the claim.

## 11. Suggested paper statement and supporting references

Proposed theorem statement after the corresponding work is complete:

> For every admitted contraction/permutation rewrite schema and every legal source configuration in the stated semantic domain, TEPL's structural attribute procedure terminates and constructs a legal equivalent replacement if and only if a valid replacement attribute assignment exists. Its generated metadata program implements this procedure uniformly over arbitrary finite ranks.

The second sentence requires the compiler correspondence theorem; omit it until that result is established. Add coverage under declared conditions only with the separate implication obligation.

Repository foundations:

- [Existing soundness proof](soundness_proof.md): semantic lowering, metadata definedness, and proof-premise discipline.
- [Ordered-contraction research draft](../examples/research/ordered_contractions.md): exact networks, forced boundaries, layout constraints, and obstruction arguments.
- [Attribute realizability and ambiguity](../examples/research/auto_realizability.md): solution sets and possible extensions.
- [Semantic decomposition proofs](../examples/research/semantic_decomposition_proofs.md): additive-component decomposition and shared-attribute qualifications.
- [Rust code generation](../src/codegen/README.md): current generated-program and runtime boundaries.

The generalized-contraction output convention is documented in the [StableHLO specification](https://openxla.org/stablehlo/spec#dot_general). The completeness proof here concerns the stated ideal semantics, not all StableHLO numerical execution policies.

[TensorRight](https://lsrcz.github.io/files/POPL25.pdf) is relevant to arbitrary-rank verification and rank bounds. The proposed TEPL emphasis is exact realizability of a fixed RHS skeleton with unknown attributes and constructive inference. Establishing the novelty of that combination requires a focused literature review; this implementation plan does not make a priority claim.
