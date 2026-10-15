# TEPL: Synthesizing Attribute Programs from Tensor Rewrite Semantics

> Paper outline for a prospective PLDI submission. Implemented mechanisms, proposed theorems, and planned evaluation are distinguished below. No claim of general completeness or established novelty is assumed.

## Abstract

- **Problem:** Tensor rewrite rules are concise in algebraic form, but their implementations require substantial attribute-transformation logic.
- **Gap:** Symbolic patterns express the computation structure; matching alone does not derive the permutations, contraction dimensions, and other attributes needed by the replacement.
- **Approach:** TEPL retains algebraic rewrite skeletons and delegates supported attribute transformations to a compiler using operation semantics.
- **Technical insight:** Coordinate correspondences used to establish semantic equivalence also provide constructive witnesses for attribute synthesis.
- **Mechanism:** Arbitrary-length axis sequences, compositional ordered mappings, lazy partition refinement, and algebraic normalization.
- **Guarantee sought:** Every accepted constructor preserves legality and tensor semantics under explicit preconditions and the selected arithmetic policy.
- **Evaluation:** Handwritten attribute logic eliminated, remaining callbacks and guards, semantic-definition cost, production-derived coverage, and synthesis performance; results to be measured.

## 1. Introduction

### 1.1 Motivation

- Algebraic tensor rewriting is used for tensor optimization.
- A rule is simple in algebraic form, but its implementation can be complex. XLA's dot reassociation is an example: a short equation requires substantial dimension-mapping logic. [XLA implementation](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc)
- Frameworks such as egg support concise symbolic operation rewriting, but tensor operations have rich attributes that also need to be transformed.
- egg can represent attributes; its symbolic patterns do not automatically derive their transformations from tensor semantics. Users must provide additional expressions or custom host-language logic. [egg's rewrite application interface](https://docs.rs/egg/latest/egg/trait.Applier.html)
- Simply matching LHS operations and attributes does not determine the attributes needed by RHS operations.
- Users therefore need additional expressions or complex rewrite logic in Rust to support these tensor transformations.
- A major source of this complexity is manual attribute transformation; legality checks, numerical policy, and profitability also contribute.
- Cascaded operations change axis positions, pairings, and output order. Equal dimension sizes do not establish equal coordinate identities.
- These transformations must work for arbitrary-rank tensors; concrete-rank examples cannot establish general correctness.
- **Research question:** What if we could design a language that makes tensor rewriting as easy and intuitive as writing its algebraic form, while leaving complex attribute transformation to the compiler?
- TEPL pursues this through operation semantics, symbolic axis mapping, and automatic RHS attribute inference under explicit conditions.
- **Evidence to prepare:** Pin the XLA example to a specific revision and identify the attribute-transformation code separately from other checks before reporting a line count.

### 1.2 Central Thesis

- **Language objective:** Design a domain-specific language dedicated to practical tensor rewriting, enabling users to express algebraic transformations concisely while supporting rich attributes, explicit conditions, and integration with host compilers.
- Support multiple dialects, each defining its own operations, attributes, and semantics.
- Define operations once within a dialect, then import and reuse them across rules.
- Represent tensor operations, graph patterns, and attribute bindings directly in the language.
- Support host-function delegation for application-specific logic; opaque host functions require specifications before their behavior can be verified.
- **Central technical advance:** Introduce `auto` to infer RHS attributes from operation semantics, leaving users to specify the intended transformation and its conditions.
- Use symbolic axis mapping to construct and verify reusable attribute programs for arbitrary-rank tensors within the supported semantics.
- Verify rules using operation legality and user-written conditions; rejct rules when correctness cannot be proved.

### 1.3 Proposed Contributions

1. **A practical DSL for tensor rewriting**
   - Support user-defined dialects and operations.
   - Express tensor rewrites as intuitive algebraic equations, with explicit conditions when needed.
   - Provide attribute binding and host-function delegation for application-specific logic.

2. **Semantics-driven synthesis of attribute programs**
   - Introduce `auto` to synthesize RHS attribute programs from operation semantics, removing the need for manual attribute transformations in supported rewrites.
   - Produce reusable functions of LHS metadata for arbitrary-rank tensors.
   - Introduce **Axes IR**, building on prior aggregated-axis representations, with explicit ordered interfaces, operand mappings, and value semantics.

3. **Symbolic Axis Mapping for composed operations**
   - Track how logical axes correspond across composed tensor operations, enabling the compiler to match their semantics and infer the required RHS attributes.
   - Split and compose arbitrary-length axis groups while preserving coordinate order and pairings.
   - Reuse established mappings across producer–consumer operations to support algebraic matching and attribute construction.

## 2. High-Level Language Design

### 2.1 Dialects, Operations, and Attributes

- A dialect groups user-defined operations with their operand signatures, attribute schemas, and semantics.
- Define operations once, then import and reuse them across rules; different dialects may define different operation vocabularies.
- Attributes configure an operation independently of its tensor operands. For example, transpose carries a permutation list.
- Running example: define transpose in a dialect file, `tensor.tepl`.

```tepl
dialect TensorLang {
  op transpose(input: tensor) -> tensor {
    attrs {
      permutation: index[];
    }
  }
}
```

- `dtype` preserves the element type; `axes` declares coordinate bindings, legality, and output order; `semantics` gives the value at those coordinates.
- Transpose changes the output coordinate order while preserving the coordinate-indexed input value. `yield input` therefore does not imply an unchanged tensor interface.
- Introduce these blocks here; defer their formal representation and composition to Axes IR.

### 2.2 Imports and Rewrite Rules

- Import a dialect and select operations used by the rules.
- Express rewrites as algebraic equations: an LHS pattern followed by an RHS replacement.
- Nested operations express producer–consumer edges; tensor captures identify matched values or subcomputations.
- Repeated captures refer to the same matched value. Distinguish this application graph from the semantic graph used for reasoning.

```tepl
from "tensor.tepl" import TensorLang as t;
use t::{transpose};
```

- Running pattern: `(transpose[@outer] (transpose[@inner] X))`.
- Its intended replacement is one transpose of `X`; the remaining task is to construct its permutation.
- Tensor declarations may additionally constrain shape and dtype; equal dimension sizes alone do not establish coordinate identity.

### 2.3 Attribute Binding and Reuse

- `op[@name]` binds the operation's attribute descriptor on the LHS.
- Captured attributes are available in `where` conditions, `derive` expressions, and RHS operations.
- Field access, such as `@outer.permutation`, refers to an individual attribute where supported by the expression context.
- Reusing `@name` on a compatible RHS operation reuses its descriptor; derived descriptors can also configure RHS operations.
- Transpose composition binds `@inner` and `@outer`; copying either alone generally cannot construct the replacement permutation.
- Descriptor compatibility checks attribute structure, not mathematical correctness.

### 2.4 Host-Function Delegation

- `$function(...)` invokes application-supplied logic using tensor captures, descriptors, or supported metadata expressions.
- Host functions can return decisions for conditions or descriptors for replacement construction.
- Example interfaces: `$supports_composition(@outer, @inner)` and `$compose_transposes(@outer, @inner)`.
- This supports target policies, opaque metadata, and transformations outside the symbolic synthesis fragment.
- A failed callback prevents rule application; the host implements the function's behavior.
- Opaque host behavior is not automatically verified. The current semantic checker rejects unsupported host-dependent proofs; semantic contracts are a future extension.

### 2.5 Conditions with `where`

- `where` states when a matched rewrite is applicable; all conditions must hold.
- Conditions can use captured metadata, pure predicates, and host-function calls.
- Example clause for the running transpose pattern:

```tepl
where {
  len(@outer.permutation) == len(@inner.permutation);
  $supports_composition(@outer, @inner);
}
```

- The length condition illustrates attribute access; for a legal transpose composition it already follows from operation legality.
- The host predicate can express application policy; it does not independently establish semantic equivalence.
- Semantic rules can state axis-routing conditions such as `@d.lhs_batching_dimensions == @inner->B`; introduce their meaning with the mapping algorithm.
- For verified rules, every correctness obligation must follow from operation legality and supported user-written conditions.

### 2.6 Attribute Construction with `derive`

- `derive` names replacement descriptors computed from the captured information.
- Combine binding, a host condition, and host-based construction in one rule:

```tepl
rule compose_transposes_host {
  (transpose[@outer] (transpose[@inner] X))
  => (transpose[@combined] X)
  where {
    $supports_composition(@outer, @inner);
  }
  derive {
    @combined = $compose_transposes(@outer, @inner);
  }
}
```

- If the inner permutation is `p` and the outer permutation is `q`, the callback constructs a descriptor with permutation `r[i] = p[q[i]]`.
- `where` decides applicability; `derive` constructs attributes; the RHS specifies the replacement computation.
- This example illustrates the host interface, not an automatically verified semantic rule.

### 2.7 Automatic Attributes with `auto`

- Replace manual descriptor construction with `auto` when operation semantics support synthesis.

```tepl
rule compose_transposes_auto {
  (transpose[@outer] (transpose[@inner] X))
  => (transpose[auto] X)
}
```

- Matching supplies tensor captures and LHS descriptors; semantic compilation derives a reusable program for the RHS permutation.
- The resulting constructor computes `r[i] = p[q[i]]` without a handwritten composition callback.
- The host policy from the previous example is omitted to isolate semantic synthesis; `auto` does not infer or replace application policy.
- Users still define operation semantics and any required correctness conditions. Unsupported inference is rejected rather than silently delegated to a solver.
- Constructor synthesis occurs at compile time; intended application evaluates the constructor on matched metadata.
- Modern constructor emission into generated Rust/C++ remains future work; existing host export and semantic synthesis have distinct implementation boundaries.

### 2.8 From `auto` to Attribute Synthesis

- `auto` lets users leave RHS attributes to the compiler. **But how can the compiler find them?**
- Matching the LHS gives known attributes and tensor metadata.
- The compiler must turn that information into a program that computes the missing RHS attributes.
- That program must work across tensor ranks and shapes, and preserve the computation under the user's conditions.
- The next section defines this problem precisely.

## 3. Problem Definition

### 3.1 What Must the Compiler Find?

- Some attributes are left as `auto` in the RHS. **How can the compiler infer them?**
- Finding attribute values at matching time is easier when ranks, shapes, and LHS attributes are known. However, deriving a formula before the matched tensors are known is not a trivial task.
- The compiler takes the following inputs:
  - User-defined operation semantics and rewrite rules.
  - Any conditions required for the rewrite.
- The output should be:
  - A reusable formula, not an attribute value for one concrete tensor.
  - For example, composing transposes with permutations `inner` and `outer` gives `result[i] = inner[outer[i]]`.
- Rule compilation derives this formula without concrete permutations; matching later supplies their values.

### 3.2 What Makes the Result Correct?

- The inferred formula must be evaluable and produce valid attributes for the RHS operations.
- With those attributes, the RHS must produce the same tensor as the LHS, under the selected arithmetic policy.
- The compiler must prove this for every legal input satisfying the user's conditions; otherwise, it rejects the rule as unproved.

### 3.3 Mathematical Definition

- Write the LHS as `L(p, X)` and the RHS as `R(p, beta, X)`.
- `p` is all known source metadata: shapes, and captured attributes.
- `X` denotes input tensor values; `beta` denotes the missing RHS attributes.
- `G(p)` is the user-written condition; `f(p)` is the attribute program to synthesize.
- The intended correctness statement is:

\[
\forall p.\quad \operatorname{LegalLHS}(p)\land G(p)
\Rightarrow
p\in\operatorname{dom}(f)\land
\operatorname{LegalRHS}(p,f(p))\land
\forall X.\ L(p,X)=R(p,f(p),X).
\]

- Read this as: **for every legal LHS satisfying the conditions, the computed RHS is legal and gives the same result for every input value.**
- \(p \in \operatorname{dom}(f)\) means the source metadata is valid input to the inferred attribute program; operations such as inverse permutation and position lookup can be evaluated.
- Equality includes ordered output shape, dtype, and element values; exact algebra does not establish floating-point reassociation.

### 3.4 When the Compiler Cannot Prove the Rule

- Return a checked attribute program when construction and verification succeed.
- Otherwise reject the rule and report the reason: malformed input, unsupported reasoning, unresolved matching, or a proof-work limit.
- Failure to prove a rule does not necessarily mean the equation is false or that no valid attributes exist.
- If several attribute choices are possible, accept any choice that can be verified; otherwise report the unresolved case.

## 4. Axes IR

### 4.1 Tensor Representation

- Tensor: value expression, ordered output axes, extents, dtype, and reusable axis mappings.
- Scalar layer: `Read`, `Const`, `Add`, `Product`, and `Reduce`.
- Index layer: coordinates, concatenation, selection, scale, and offset.
- Axis layer: scoped groups and symbolic sequence operations.
- Groups have arbitrary finite length, including zero where operation legality permits.
- A group is an ordered sequence of coordinates, not a flattened tensor dimension.
- Output axes may remain even when the value is constant or independent of them.

### 4.2 Operation Semantics

- Addition: shared input/output coordinates; pointwise `Add`.
- Transpose: permuted ordered interface; preserved coordinate-indexed values.
- Reduction: selected bound coordinates; surviving output interface.
- Dot: paired batch group `B`, paired contraction group `K`, free groups `I` and `J`.
- Dot value: `Reduce(K, Product(lhs, rhs))`; output: `B ++ I ++ J`.
- Dot output has three groups; its semantic relation also contains the bound group `K`.
- Broadcast: inserted output coordinates absent from the input access.
- Convolution and slice: calculated accesses; algebraic index normalization supplements axis correspondence.

## 5. Symbolic Axis Mapping

### 5.1 Ordered Mapping Witnesses

- Define `gather(P, q)` as selection from sequence `P` in the order given by `q`.
- Store `AxisMap(source, destination, positions)` with invariant:
  `gather(destination, positions) = source`.
- Example: a dot operand associates `B`, `K`, and its free group with physical axis positions.
- Paired operand bindings share logical coordinates only after pairing lengths and extents are validated.
- Use scoped identities to distinguish unrelated coordinates with equal extents.

### 5.2 Producer–Consumer Composition

- Each constructed tensor exposes `TensorAxisInterface(axes, mappings)`.
- Resolve consumer bindings against the producer's ordered interface.
- If `gather(P, p) = G` and `gather(G, q) = H`, derive:
  `gather(P, gather(p, q)) = H`.
- Publish the composed witness immediately for subsequent consumers.
- Restrict mappings through selections only when the relevant coordinates survive.
- Invert permutations only with a bijectivity witness; partial selections are not invertible permutations.
- Reuse mappings across arbitrarily many graph edges, subject to implementation resource limits.

### 5.3 Lazy Ordered Partitions

- For valid distinct selector `q`, construct `AxisPartition(P, q)`:
  - Selected axes: `gather(P, q)`.
  - Remaining positions: `ordered_complement(range(len(P)), q)`.
  - Remaining axes: selection using those remaining positions.
  - Retain both position lists as witnesses into `P`.
- Selected axes preserve selector order; remaining axes preserve parent order.
- Do not assume `selected ++ remaining = P`; the witnesses recover original placement.
- Keep expressions symbolic; no enumeration of unknown axis counts is required.
- Memoize splits by parent and selector; refine only relationships required by composition or matching.

### 5.4 Refining Two Partitions of One Interface

- Producer and consumer may partition the same interface differently.
- Compare their selectors, rather than group names or dimension sizes.
- `matchPositions(p, q)` returns paired lists `(u, v)` satisfying:
  `gather(p, u) = gather(q, v)`.
- Preserve the left selector's traversal order and the corresponding right-side positions.
- Record an overlap view with witnesses into both groups.
- Propagate the same selected slots to a paired operand, preserving contraction correspondence.

### 5.5 Reassociation Example with Interleaved Selection

```tepl
rule associate {
  (dot[@d] (dot[@inner] A B) C)
  => (dot[auto] A (dot[auto] B C))
  where {
    @d.lhs_batching_dimensions == @inner->B;
  }
}
```

- `@inner->B` denotes the producer batch group's output positions; the condition preserves that batch routing.
- The rule illustrates guarded generalized-dot reassociation, not unconditional associativity.
- Producer output: `P = B ++ I ++ J`; outer contraction selector: `q`.
- Match `q` separately against positions of `I` and `J` in `P`.
- Derive selected/remaining views of both groups and their original positions.
- Transfer selected slots through the outer dot's paired selector into `C`.
- Construct the intermediate RHS tensor; record all surviving coordinate mappings.
- Construct the final RHS operation by querying this intermediate interface.
- Verify algebraic equivalence and output ordering after attribute construction.
- Concrete illustration only: `P = [i0, i1, j0, j1]`, `q = [3, 0]` selects `[j1, i0]`.
  - Selected `I` view: `[i0]`; slot `[1]` in the contraction sequence.
  - Selected `J` view: `[j1]`; slot `[0]` in the contraction sequence.
  - Concatenating these views in `I, J` order would change contraction pairing; retain the slot witnesses.
- General implementation uses symbolic selectors for the same construction.

### 5.6 Representation and Lookup

- `wiring_`: established interface/group/position witnesses.
- `splits_`: selected/remaining views for individual selection requests.
- `partitions_`: complete disjoint decompositions of interfaces; support coverage and disjointness reasoning.
- `overlaps_`: intersections with paired position witnesses between established groups.
- Store symbolic expressions by shared references; records use vectors with associative lookup indexes.
- Mapping lookup follows registered origins with visited sets and composes witnesses.
- Pure mapping operations avoid invoking the full prover; contextual fallbacks may discharge unresolved obligations.
- Invalidate cached results when established relationships change; keep unresolved relationships explicit.
- Measure record and expression growth; do not claim polynomial complexity for the entire synthesis algorithm.

## 6. Attribute Program Synthesis

### 6.1 Algorithm

1. Elaborate operation definitions, source metadata, holes, and user conditions.
2. Lower the LHS; establish legality and compose producer–consumer coordinates.
3. Normalize scalar expressions using policy-permitted algebraic laws.
4. Normalize the RHS arithmetic skeleton and identify term/access correspondences.
5. Build RHS operations using matched logical groups and reusable interfaces.
6. Construct position attributes through mapping lookup; construct supported numeric attributes through index-expression matching.
7. Resolve output extents and prove attribute/domain obligations.
8. Extract source-only metadata expressions and independently verify the completed rewrite.

### 6.2 Algebraic and Coordinate Reasoning

- Example: `(dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B C))`; normalize by distributivity and reduction linearity, then recover both descriptors from coordinate correspondence.
- Scalar laws: associativity, commutativity, distributivity, reduction linearity, and coefficient collection, with explicit side conditions.
- Coordinate laws: selection composition, permutation identities, alpha-renaming, and ordered partition relations.
- Calculated reads: normalize supported affine accesses and compare scoped coordinate bases.
- Attribute matching: solve whole coefficient/offset holes; reject unsupported coupled or inverse equations.
- Geometry alone does not establish value equivalence or reduction-domain equality.
- No requirement for exhaustive equality saturation or enumeration of concrete attribute assignments.

### 6.3 Residual Constructor Programs

- Constructors contain source attributes, sequence operations, extent arithmetic, and shared subexpressions.
- Examples: inverse permutation, gathered positions, ordered complements, and composed slice offsets.
- Slice composition constructs `start = inner.start + inner.stride * outer.start`, `stride = inner.stride * outer.stride`, and `size = outer.size`; source legality must imply the resulting access bounds.
- Reject residual RHS holes or dependence on tensor element values.
- Compile-time synthesis produces the program; intended matching-time execution supplies concrete metadata.
- Call this staged evaluation; laziness applies to partition/refinement work, not necessarily constructor execution.
- Current implementation stores/checks constructors; modern Rust/C++ constructor emission remains future work.

## 7. Correctness and Scope

### 7.1 Proposed Proof Structure

- **Lowering lemma:** Operation lowering preserves the declared denotation.
- **Mapping lemma:** Composition, selection, and refinement preserve ordered coordinate witnesses.
- **Normalization lemma:** Each enabled rewrite preserves values and domains under its side conditions.
- **Constructor lemma:** Emitted expressions are source-dependent, well-defined, and realize the required coordinate relations.
- **Acceptance theorem:** Successful verification implies the contract in Section 3.3.
- These are proof obligations for the paper; executable checks alone do not constitute formal proofs.

### 7.2 Verification Boundary

- Re-lower the completed RHS in a fresh context.
- Recheck accesses, reduction domains, output order, extents, dtype, and legality.
- Treat synthesis obligations as goals, never additional assumptions.
- Constructor provenance and mapping records do not substitute for semantic verification.
- The verifier shares semantic infrastructure with synthesis; it is not a separately mechanized proof kernel.

### 7.3 Completeness Boundary

- General completeness and weakest-precondition inference are not claimed.
- Investigate restricted completeness for a precisely defined fragment with fixed term correspondence, bijective axis routing, and directly solvable attribute holes.
- Establish such a theorem before presenting the fragment as complete.
- Unsupported cases include unresolved correspondence, general inverse equations, and exhausted proof budgets.

## 8. Implementation

- Frontend and operation-defined semantics; no per-operation inference routine required for the supported fragment.
- `src/core/index_semantics/`: lowering, mapping, normalization, synthesis, and verification.
- `axis_bindings.*`, `axis_maps.cc`, `axis_transport.cc`: mapping storage, refinement, and transport.
- `index_expr.*`, `index_inference.*`: calculated access normalization and attribute matching.
- `synthesize.cc`, `verify.cc`: constructor production and rechecking.
- Modern semantic inference uses no Z3 search; legacy code-generation components remain separate.
- Document numerical policy, bounded reasoning, implementation trust assumptions, and unsupported features.

## 9. Evaluation Plan

### 9.1 Research Questions

- **RQ1 — Authoring effort:** How much handwritten attribute logic is eliminated, and what callbacks, guards, and semantic definitions remain necessary?
- **RQ2 — Coverage:** Which practical attribute transformations can be synthesized, including those that require newly constructed rather than copied descriptors?
- **RQ3 — Scalability:** How do composition depth, partition overlap, and expression size affect synthesis?
- **RQ4 — Generality:** Can new operations be supported through semantic definitions alone?
- **RQ5 — Reliability:** Does verification reject corrupted constructors and invalid rewrites?

### 9.2 Workloads and Comparisons

- Select and document a production-derived corpus with explicit inclusion/exclusion criteria.
- Include distribution, transpose/dot, guarded reassociation, reductions, broadcasts, slices, and supported convolution rewrites.
- Compare equivalent rewrites implemented with handwritten constructors, TEPL host delegation, and TEPL `auto`; keep applicability conditions and numerical policies aligned.
- Include simple descriptor reuse and complex attribute reconstruction; report them separately.
- Use a feasible bounded-search baseline to assess synthesis cost, with its rank and attribute-domain limits stated.
- Compare related verifiers on shared expressible tasks; distinguish verification coverage from synthesis coverage.
- Report required user guards and semantic-definition effort, not only rule length.
- Pin production examples to revisions; separate attribute transformation from profitability, graph construction, and unrelated infrastructure.

### 9.3 Measurements and Ablations

- Per-rule attribute-code size and remaining host callbacks; distinguish eliminated logic from logic moved into shared helpers.
- User-supplied guards and operation-definition size; report both initial semantic-specification cost and reuse across multiple rules.
- Compare total authored code for a rule corpus as well as per-rule savings; treat code size as an effort proxy, not evidence of usability by itself.
- Coverage denominator; accepted, unsupported, invalid, and budget-limited cases.
- Synthesis/verification time, peak memory, constructor size, and eventual matching-time cost.
- Scale source composition depth, overlap patterns, repeated factors, and concrete constructor inputs.
- Disable reusable interfaces, lazy refinement, or mapping caches to isolate their effects.
- Test interleaved selections, empty groups, multiple contractions, rank-zero cases, and supported zero extents.
- Use independent numerical references and constructor mutation tests as implementation evidence.
- Numerical testing supplements universal reasoning; it does not replace it.
- Leave all performance and coverage numbers unfilled until measured.

## 10. Related Work and Novelty Positioning

- **TensorRight:** Aggregated axes and arbitrary-rank rewrite verification are prior work; axis grouping alone is not a novelty claim. [POPL 2025](https://doi.org/10.1145/3704865)
- **Tensor rewrite synthesis and equality saturation:** Compare TASO/TENSAT on generated artifacts, semantic assumptions, and treatment of attributes; examine primary papers before asserting distinctions.
- **Semantics-driven synthesis:** Compare synthesis targets, parameterization, and correctness contracts with superoptimization and rewrite-synthesis systems.
- **Index-notation and affine compiler IRs:** Explain how TEPL reconstructs operation attributes from symbolic coordinate correspondence.
- **Rewrite DSLs and host callbacks:** Compare declarative authoring with manually implemented attribute transformations.
- Candidate distinction to substantiate: compositional synthesis of reusable, rank-polymorphic attribute programs from user-defined tensor semantics.
- Avoid “first” claims until a systematic literature review supports them.

## 11. Limitations and Future Work

- Incomplete inference; ambiguity and finite proof budgets can reject valid rules.
- Partition and normalized-expression growth across long compositions.
- Restricted convolution, broadcasting, and positive-size/positive-stride slice semantics.
- General padding, data-dependent indexing, coupled attribute equations, and unsupported numerical policies.
- Semantic contracts for host functions and a smaller trusted verification kernel.
- Integration of modern constructor programs into generated rewrite implementations.

## 12. Conclusion

- Reiterate the central result to establish: semantic coordinate correspondence can generate executable attribute transformations.
- Summarize only guarantees and practical benefits supported by proofs and measurements.

## Appendix / Artifact

- Formal IR grammar, denotational semantics, mapping invariants, and algebraic side conditions.
- Full synthesis pseudocode and proof details.
- Benchmark definitions, guards, synthesized constructors, and failure classifications.
- Reproduction commands, versioned environments, and independent reference tests.
