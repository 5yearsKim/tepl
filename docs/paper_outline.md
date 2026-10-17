# TEPL: Synthesizing Attribute Programs from Tensor Rewrite Semantics

> Paper outline for a prospective PLDI submission. Implemented mechanisms, proposed theorems, and planned evaluation are distinguished below. No claim of general completeness or established novelty is assumed.

## Abstract

- **Problem:** Tensor rewrite rules are concise in algebraic form, but their implementations require substantial attribute-transformation logic.
- **Gap:** Symbolic patterns express the computation structure; matching alone does not derive the permutations, contraction dimensions, and other attributes needed by the replacement.
- **Approach:** TEPL (Tensor Equality Pattern Language) lets users write tensor rewrite rules with explicit conditions and attribute changes. When replacement attributes are omitted, the compiler infers them from operation semantics and checks supported rewrites for validity and equivalent results.
- **Technical insight:** Coordinate correspondences used to establish semantic equivalence also provide constructive witnesses for attribute synthesis.
- **Mechanism:** Arbitrary-length axis sequences, compositional ordered mappings, lazy partition refinement, and algebraic normalization.
- **Guarantee sought:** Every semantically verified rewrite has computable replacement attributes and preserves legality and tensor semantics under explicit preconditions and the selected arithmetic policy.
- **Evaluation:** Authoring effort with explicit and inferred attributes, remaining attribute expressions and conditions, semantic-definition cost, production-derived coverage, and synthesis performance; results to be measured.

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

TEPL describes tensor rewrites, their conditions, and their attribute changes together. Attribute changes can be written by users or inferred from operation semantics. The compiler checks supported rewrites for validity and equivalent results.

- **Language objective:** Let users express algebraic rewrite patterns, applicability conditions, and attribute changes in one language.
- Support multiple dialects, each defining its own operations, attributes, and semantics.
- Define operations once within a dialect, then import and reuse them across rules.
- Represent tensor operations, graph patterns, and attribute bindings directly in the language.
- Let users write conditions and compute replacement attributes directly in the language.
- **Central technical advance:** Infer omitted RHS attributes from operation semantics; users can also request inference explicitly with `auto`.
- Use symbolic axis mapping to construct and verify reusable attribute programs for arbitrary-rank tensors within the supported semantics.
- Verify semantic rules using operation legality and user-written conditions; reject them as unproved when correctness cannot be established.

### 1.3 Proposed Contributions

1. **A language for tensor rewrites**
   - Support user-defined dialects and operations.
   - Express rewrite patterns, applicability conditions, and attribute changes together.
   - Let users capture and reuse attributes, or compute replacement attributes directly in the language.

2. **Automatic attribute inference**
   - Synthesize omitted RHS attributes from operation semantics, removing the need to write attribute transformations for supported rewrites.
   - Produce reusable functions of LHS metadata for arbitrary-rank tensors.
   - Introduce **Axes IR**, building on prior aggregated-axis representations, with explicit ordered interfaces, operand mappings, and value semantics.

3. **Symbolic axis mapping for composed operations**
   - Track how logical axes correspond across composed tensor operations, enabling the compiler to match their semantics and infer the required RHS attributes.
   - Split and compose arbitrary-length axis groups while preserving coordinate order and pairings.
   - Reuse established mappings across producer–consumer operations to support algebraic matching and attribute construction.

## 2. Background

- TensorRight verifies tensor graph rewrites for tensors of arbitrary rank and size. [POPL 2025](https://doi.org/10.1145/3704865)
- It uses aggregated axes to represent groups with an arbitrary number of axes, allowing one rule to describe tensors of different ranks.
- This provides background for TEPL's use of symbolic axis groups to reason about rewrites and infer replacement attributes.

## 3. High-Level Language Design

### 3.1 Dialects, Operations, and Attributes

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

### 3.2 Imports and Rewrite Rules

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

### 3.3 Attribute Binding and Reuse

- `op[@name]` binds the operation's attribute descriptor on the LHS.
- Omitting LHS attributes captures each operation's descriptor independently and anonymously; name captures when expressions or RHS operations need to refer to them.
- Captured attributes are available in `where` conditions, `derive` expressions, and RHS operations.
- Field access, such as `@outer.permutation`, refers to an individual attribute where supported by the expression context.
- Reusing `@name` on a compatible RHS operation reuses its descriptor; derived descriptors can also configure RHS operations.
- Transpose composition binds `@inner` and `@outer`; copying either alone generally cannot construct the replacement permutation.
- Descriptor compatibility checks attribute structure, not mathematical correctness.

### 3.4 Writing Attribute Changes

- Write conditions and compute replacement attributes directly in the language.
- Expressions can read captured attributes and available shape and dtype bindings; they do not execute tensor computations.
- Introduce local bindings, builtins, lists, indexing, conditionals, and comprehensions as examples need them.
- For transpose composition, `gather(@inner.permutation, @outer.permutation)` computes the replacement permutation.
- Descriptor constructors collect computed fields into immutable attributes for an RHS operation.
- Some expressions can execute but cannot yet be checked for semantic correctness by the compiler.

### 3.5 Conditions with `where`

- A `where` block ends with a condition that decides whether the rule applies. Write any local calculations before it. Only `let` bindings end with `;`.
- Conditions can use captured metadata, builtins, and local expressions. Combine predicates with `&&` and `||`.
- Example clause for the running transpose pattern:

```tepl
where {
  let rank = len(@inner.permutation);
  len(@outer.permutation) == rank
}
```

- The length condition illustrates attribute access; for a legal transpose composition it already follows from operation legality.
- Omit `where` when no additional condition is needed.
- A condition may express applicability or application policy; the compiler does not infer profitability decisions.
- Semantic rules can state axis-routing conditions such as `@d.lhs_batching_dimensions == @inner->B`; introduce their meaning with the mapping algorithm.
- For verified rules, every correctness obligation must follow from operation legality and supported user-written conditions.

### 3.6 Attribute Construction with `derive`

- `derive` names replacement descriptors computed from the captured information.
- Combine captured attributes and an explicit descriptor constructor in one rule:

```tepl
rule compose_transposes_explicit {
  (transpose[@outer] (transpose[@inner] X))
  => (transpose[@combined] X)
  derive {
    @combined = {
      permutation: gather(@inner.permutation, @outer.permutation)
    };
  }
}
```

- If the inner permutation is `p` and the outer permutation is `q`, the expression constructs a descriptor with permutation `r[i] = p[q[i]]`.
- `where` decides applicability; `derive` constructs attributes; the RHS specifies the replacement computation.
- Local bindings can name intermediate computations. A spread such as `{ ..@outer, permutation: permutation }` copies fields before applying explicit overrides.
- `where` and `derive` have separate local scopes; derivations may read captured and earlier derived descriptors.
- With the transpose semantics, this explicit constructor is supported by semantic verification. Checking attribute types alone does not prove that the rewrite preserves the result.

### 3.7 Automatic Attributes with `auto`

- Omit replacement attributes when operation semantics support synthesis. An explicit `[auto]` annotation requests the same inference.

```tepl
rule compose_transposes_auto {
  (transpose (transpose X)) => (transpose X)
}
```

- Matching supplies tensor captures and LHS descriptors; semantic compilation derives a reusable program for the RHS permutation.
- In the explicit rule, the user writes how to combine the permutations. In this rule, the compiler derives that computation from transpose semantics: `r[i] = p[q[i]]`.
- Each omitted RHS descriptor is inferred independently. Use `[@name]` to reuse an existing descriptor or `[auto, field = value]` to pin a field during inference.
- Users still define operation semantics and any required correctness conditions. Unsupported inference is rejected with a diagnostic.
- Constructor synthesis occurs at compile time; application evaluates the constructor on matched metadata.
- Completed constructors execute in the greedy rewrite engine or generated Rust with the shared runtime. C++ generation remains unavailable.

### 3.8 From `auto` to Attribute Synthesis

- `auto` lets users leave RHS attributes to the compiler. **But how can the compiler find them?**
- Matching the LHS gives known attributes and tensor metadata.
- The compiler must turn that information into a program that computes the missing RHS attributes.
- That program must work across tensor ranks and shapes, and preserve the computation under the user's conditions.
- The next section defines this problem precisely.

## 4. Problem Definition

### 4.1 What Must the Compiler Find?

- Some RHS attributes are omitted or marked `auto`. **How can the compiler infer them?**
- Finding attribute values at matching time is easier when ranks, shapes, and LHS attributes are known. However, deriving a formula before the matched tensors are known is not a trivial task.
- The compiler takes the following inputs:
  - User-defined operation semantics and rewrite rules.
  - Any conditions required for the rewrite.
- The output should be:
  - A reusable formula, not an attribute value for one concrete tensor.
  - For example, composing transposes with permutations `inner` and `outer` gives `result[i] = inner[outer[i]]`.
- Rule compilation derives this formula without concrete permutations; matching later supplies their values.

### 4.2 What Makes the Result Correct?

- The compiler must establish that replacement attributes can be computed, satisfy the replacement operations' requirements, and preserve the result whenever the rule's conditions hold.
- An explicit constructor supplies the attribute program; automatic inference asks the compiler to discover it. Both require checking for semantic rules.
- The attribute program must be evaluable and produce valid attributes for the RHS operations.
- With those attributes, the RHS must produce the same tensor as the LHS, under the selected arithmetic policy.
- The compiler must prove this for every legal input satisfying the user's conditions; otherwise, it rejects the rule as unproved.

### 4.3 Mathematical Definition

- Write the LHS as `L(p, X)` and the RHS as `R(p, beta, X)`.
- `p` is all known source metadata, including shapes, dtypes, and captured attributes.
- `X` denotes input tensor values; `beta` denotes the RHS attributes to construct.
- `G(p)` is the user-written condition; `f(p)` is the attribute program, supplied explicitly or synthesized for missing attributes.
- The intended correctness statement is:

\[
\forall p.\quad \operatorname{LegalLHS}(p)\land G(p)
\Rightarrow
p\in\operatorname{dom}(f)\land
\operatorname{LegalRHS}(p,f(p))\land
\forall X.\ L(p,X)=R(p,f(p),X).
\]

- Read this as: **for every legal LHS satisfying the conditions, the computed RHS is legal and gives the same result for every input value.**
- \(p \in \operatorname{dom}(f)\) means the source metadata is valid input to the attribute program; operations such as inverse permutation and position lookup can be evaluated.
- Define `G(p)` through successful guard evaluation to `true`. The formal account must cover ordered local evaluation, short-circuiting, indexing, and checked arithmetic, including evaluation failures.
- Equality includes ordered output shape, dtype, and element values; exact algebra does not establish floating-point reassociation.

### 4.4 When Can the Compiler Accept a Rewrite?

- The compiler accepts a semantic rewrite when it can show that the replacement attributes are computable, the replacement operations are valid, and the result is preserved under the stated conditions.
- The compiler checks correctness both for attributes computed explicitly in `derive` and for attributes inferred automatically.
- If the compiler cannot prove correctness, it rejects the rule as unproved. The rule may still be correct and can be implemented manually, but its correctness must be established separately.

## 5. Proposed Method

### 5.1 Axes IR

Axes IR is TEPL's intermediate representation of tensor computations. It describes how operations use tensor axes and compute output values. By translating operation definitions into this shared representation, TEPL can track axis relationships across operations, infer replacement attributes, and check that rewrites preserve the result.

#### 5.1.1 Symbolic Axis Groups

- A symbolic axis group borrows the idea of grouping an arbitrary number of axes from [TensorRight's aggregated axes](https://doi.org/10.1145/3704865).
- Both support reasoning about tensors of arbitrary rank.
- While TensorRight represents an aggregated axis as a set of named axes, TEPL represents a symbolic axis group as an ordered sequence.
- TEPL tracks position mappings between these groups and tensor axes to construct replacement attributes.

A symbolic axis group represents an ordered sequence of axes without fixing how many axes it contains. For example, one group can represent all contraction axes of a tensor operation.

TEPL declares a group using `bind`, which associates the group with axis positions in one or more operands. When a binding includes two operands, their selected axes are paired in list order. The operation's attributes determine the positions, so the same definition can describe tensors of different ranks and axis arrangements.

Operands bound to the same group must select the same number of axes, with matching sizes at corresponding positions. Equal sizes alone do not establish a pairing; the binding specifies it.

The following `dot_general` definition illustrates these bindings. It takes two tensor operands and four attributes specifying their batch and contraction axes. Each attribute is a list of axis positions. The excerpt shows the attribute declarations and axis bindings; dtype and value semantics are omitted here.

```tepl
op dot_general(lhs: tensor, rhs: tensor) -> tensor {
    attrs {
        lhs_batching_dimensions: index[];
        rhs_batching_dimensions: index[];
        lhs_contracting_dimensions: index[];
        rhs_contracting_dimensions: index[];
    }

    axes {
        let lb = attrs.lhs_batching_dimensions;
        let rb = attrs.rhs_batching_dimensions;
        let lc = attrs.lhs_contracting_dimensions;
        let rc = attrs.rhs_contracting_dimensions;

        let lf = ordered_complement(all_axes(lhs), lb ++ lc);
        let rf = ordered_complement(all_axes(rhs), rb ++ rc);

        bind B { lhs: lb, rhs: rb };
        bind K { lhs: lc, rhs: rc };
        bind I { lhs: lf };
        bind J { rhs: rf };

        yield B ++ I ++ J;
    }
}
```

Axis positions start at zero. `all_axes(t)` lists all axis positions of `t`, and `++` joins sequences in order. `ordered_complement` selects the remaining axes in operand order, and `yield` specifies the output axis order.

This definition supports different numbers and arrangements of axes. As one concrete example, consider the axis orders below. The labels identify logical axes:

```text
lhs:    [i, k0, b, k1]
rhs:    [k1, j, k0, b]
output: [b, i, j]
```

For this instance, the attributes are:

```text
lhs_batching_dimensions = [2]
rhs_batching_dimensions = [3]
lhs_contracting_dimensions = [1, 3]
rhs_contracting_dimensions = [2, 0]
```

The batch binding pairs axis `2` of `lhs` with axis `3` of `rhs`. The contraction binding pairs positions in list order: `(1, 2)` for `k0` and `(3, 0)` for `k1`. The remaining axes form `I = [i]` and `J = [j]`, giving the output order `B ++ I ++ J = [b, i, j]`. The contraction group `K` is omitted from the output because those axes are reduced.

- Here, `K` contains two axes. The same definition supports other group lengths without enumerating individual axes.
- Grouping preserves each axis separately and retains its order; it does not flatten the axes into one dimension.
- A group may be empty when the operation permits it, such as the batch group of a dot operation without batch axes.

These mappings let the compiler recover where logical axes appear in each operand and use those positions to construct replacement attributes.

#### 5.1.2 Expressing Tensor Computations

- The `axes` block defines axis relationships and output order. The `semantics` block defines how each output element is computed.
- TEPL uses **value expressions** to compute tensor element values and **index expressions** to calculate the indices used to read them.
- Operand references use the coordinates established by axis bindings. Explicit index expressions handle accesses that require additional calculations, such as slices and convolutions.
- `Read` connects the two: it takes an operand and an index expression, then returns the value at those indices.
- A `semantics` block contains a value expression returned by `yield`.

**Value Expressions**

- For example, elementwise addition pairs every axis of `lhs` with the corresponding axis of `rhs`. Both operands bind to the same symbolic axis group `S`, and the output preserves that group.

```tepl
axes {
    bind S {
        lhs: all_axes(lhs),
        rhs: all_axes(rhs)
    };
    yield S;
}

semantics {
    yield Add(lhs, rhs);
}
```

- The shared binding requires the operands to have the same rank and matching axis sizes.
- At each coordinate in `S`, `lhs` and `rhs` refer to the corresponding operand elements. `Add(lhs, rhs)` adds those values to produce the output element.
- This describes elementwise addition without broadcasting.
- Value expressions describe how tensor element values are computed:

| Expression | Meaning |
| --- | --- |
| `input` | Read an operand element using its axis bindings. |
| `Const(c)` | Produce a signed integer constant. |
| `Add(a, b)` | Add two scalar values. |
| `Product(a, b)` | Multiply two scalar values. |
| `Reduce(G, e)` | Sum `e` over all coordinate combinations of the axes in group `G`. |
| `Read(input, indices)` | Read an operand element at explicitly calculated indices. |

- `Reduce` denotes summation here; other reductions, such as maximum, are not implied by this primitive.

**Index Expressions**

- Some operations read input elements at shifted or scaled indices. Index expressions describe these calculations and produce a sequence with one index for each accessed axis.
- The index expressions are:

| Expression | Meaning |
| --- | --- |
| `Coords(G)` | Get the current coordinates of axis group `G` in group order. |
| `IndexAdd(a, b)` | Add index sequences element by element. |
| `Concat(a, b)` | Join index sequences in order. |
| `IndexScale(attrs.strides, indices)` | Multiply each index by its corresponding scale. |
| `IndexOffset(attrs.starts, indices)` | Add an offset to each index. |

- For example, a slice uses the following value expression to read at calculated indices:

```tepl
Read(input,
    IndexOffset(attrs.starts,
        IndexScale(attrs.strides, Coords(O))))
```

- `O` is the output axis group. The index expression computes input indices as `starts + strides * output_indices`, separately for each axis.
- `Read` retrieves the input element at those indices. The index calculation selects the element; the enclosing value expression produces its value.

**Combining Expressions with Axis Groups**

- Value and index expressions use the symbolic axis groups declared in the `axes` block.
- Axis bindings determine which operand elements participate in the expression. Operands bound to the same group use corresponding coordinates.
- A reduction sums over the coordinates of its group while the remaining output coordinates stay fixed. Because a group can contain a variable number of axes, one reduction expression can describe summation over different numbers of dimensions.
- To complete the value definition of `dot_general` from Section 5.1.1, place the following block immediately after its `axes` block:

```tepl
semantics {
    yield Reduce(K, Product(lhs, rhs));
}
```

- `Product(lhs, rhs)` multiplies elements whose coordinates are determined by the bindings for `B`, `I`, `J`, and `K`. `Reduce(K, ...)` sums these products over the contraction coordinates.
- The output coordinates belong to `B`, `I`, and `J`, as specified by `yield B ++ I ++ J`. The contraction group `K` participates in the computation but does not appear in the output.
- For the preceding concrete example, this computes:

\[
\operatorname{out}[b,i,j]
=
\sum_{k_0,k_1}
\operatorname{lhs}[i,k_0,b,k_1]\,
\operatorname{rhs}[k_1,j,k_0,b].
\]

- Value and index expressions, together with axis bindings, describe a range of tensor operations. Elementwise operations combine values at shared coordinates; transpose changes the output axis order; summation and dot products use reductions.
- Slices and convolutions calculate input indices to select the elements they need.
- Broadcasting adds output axes while reusing input values along those axes.
- Output shape is described separately from element computation. For example, a tensor filled with zeros still has a defined shape.

<br>
- These expressions give the compiler a common way to describe different operations. The next section explains how axis mappings connect these descriptions across composed operations.

### 5.2 Symbolic Axis Mapping

#### 5.2.1 Ordered Mapping Witnesses

- Define `gather(P, q)` as selection from sequence `P` in the order given by `q`.
- Store `Mapping(interface, group, positions)` with invariant:
  `gather(interface, positions) = group`.
- Example: a dot operand associates `B`, `K`, and its free group with physical axis positions.
- Paired operand bindings share logical coordinates only after pairing lengths and extents are validated.
- Use scoped identities to distinguish unrelated coordinates with equal extents.

#### 5.2.2 Producer–Consumer Composition

- Each constructed tensor exposes `TensorAxisInterface(axes, mappings)`.
- Resolve consumer bindings against the producer's ordered interface.
- If `gather(P, p) = G` and `gather(G, q) = H`, derive:
  `gather(P, gather(p, q)) = H`.
- Publish the composed witness immediately for subsequent consumers.
- Restrict mappings through selections only when the relevant coordinates survive.
- Invert permutations only with a bijectivity witness; partial selections are not invertible permutations.
- Reuse mappings across arbitrarily many graph edges, subject to implementation resource limits.

#### 5.2.3 Lazy Ordered Partitions

- For valid distinct selector `q`, construct `AxisPartition(P, q)`:
  - Selected axes: `gather(P, q)`.
  - Remaining positions: `ordered_complement(range(len(P)), q)`.
  - Remaining axes: selection using those remaining positions.
  - Retain both position lists as witnesses into `P`.
- Selected axes preserve selector order; remaining axes preserve parent order.
- Do not assume `selected ++ remaining = P`; the witnesses recover original placement.
- Keep expressions symbolic; no enumeration of unknown axis counts is required.
- Memoize splits by parent and selector; refine only relationships required by composition or matching.

#### 5.2.4 Refining Two Partitions of One Interface

- Producer and consumer may partition the same interface differently.
- Compare their selectors, rather than group names or dimension sizes.
- `matchPositions(p, q)` returns paired lists `(u, v)` satisfying:
  `gather(p, u) = gather(q, v)`.
- Preserve the left selector's traversal order and the corresponding right-side positions.
- Record an overlap view with witnesses into both groups.
- Propagate the same selected slots to a paired operand, preserving contraction correspondence.

#### 5.2.5 Reassociation Example with Interleaved Selection

```tepl
rule associate {
  (dot[@d] (dot[@inner] A B) C)
  => (dot[auto] A (dot[auto] B C))
  where {
    @d.lhs_batching_dimensions == @inner->B
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

#### 5.2.6 Representation and Lookup

- `mappings_`: one record per interface and group, stored in a vector with stable IDs and lookup indexes.
- `splits_`: selected/remaining views for individual selection requests.
- `partitions_`: complete disjoint decompositions of interfaces, referencing mapping IDs; support coverage and disjointness reasoning.
- Keep each mapping's first registered position expression. Equivalent selectors share its ID; partitions retain their original selector when it differs.
- `overlaps_`: intersections with paired position witnesses between established groups.
- Store symbolic expressions by shared references; records use vectors with associative lookup indexes.
- Mapping lookup follows registered origins with visited sets and composes witnesses.
- Pure mapping operations avoid invoking the full prover; contextual fallbacks may discharge unresolved obligations.
- Invalidate cached results when established relationships change; keep unresolved relationships explicit.
- Measure record and expression growth; do not claim polynomial complexity for the entire synthesis algorithm.

### 5.3 Attribute Program Synthesis

#### 5.3.1 Algorithm

1. Elaborate operation definitions, source metadata, holes, and user conditions.
2. Lower the LHS; establish legality and compose producer–consumer coordinates.
3. Normalize scalar expressions using policy-permitted algebraic laws.
4. Normalize the RHS arithmetic skeleton and identify term/access correspondences.
5. Build RHS operations using matched logical groups and reusable interfaces.
6. Construct position attributes through mapping lookup; construct supported numeric attributes through index-expression matching.
7. Resolve output extents and prove attribute/domain obligations.
8. Extract source-only metadata expressions and independently verify the completed rewrite.

- If several attribute choices are possible, accept any choice that can be verified; otherwise report the unresolved case.

#### 5.3.2 Algebraic and Coordinate Reasoning

- Example: `(dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B C))`; normalize by distributivity and reduction linearity, then recover both descriptors from coordinate correspondence.
- Scalar laws: associativity, commutativity, distributivity, reduction linearity, and coefficient collection, with explicit side conditions.
- Coordinate laws: selection composition, permutation identities, alpha-renaming, and ordered partition relations.
- Calculated reads: normalize supported affine accesses and compare scoped coordinate bases.
- Attribute matching: solve whole coefficient/offset holes; reject unsupported coupled or inverse equations.
- Geometry alone does not establish value equivalence or reduction-domain equality.
- No requirement for exhaustive equality saturation or enumeration of concrete attribute assignments.

#### 5.3.3 Residual Constructor Programs

- Constructors contain source attributes, sequence operations, extent arithmetic, and shared subexpressions.
- Examples: inverse permutation, gathered positions, ordered complements, and composed slice offsets.
- Slice composition constructs `start = inner.start + inner.stride * outer.start`, `stride = inner.stride * outer.stride`, and `size = outer.size`; source legality must imply the resulting access bounds.
- Reject residual RHS holes or dependence on tensor element values.
- Compile-time synthesis produces the program; matching-time execution supplies concrete metadata.
- Call this staged evaluation; laziness applies to partition/refinement work, not necessarily constructor execution.
- Completed attribute programs lower to executable IR for the greedy rewrite engine and Rust generation. Generated Rust evaluates synthesized attributes through functions returning concrete descriptor types.

### 5.4 Correctness and Scope

#### 5.4.1 Proposed Proof Structure

- **Lowering lemma:** Operation lowering preserves the declared denotation.
- **Mapping lemma:** Composition, selection, and refinement preserve ordered coordinate witnesses.
- **Normalization lemma:** Each enabled rewrite preserves values and domains under its side conditions.
- **Constructor lemma:** Emitted expressions are source-dependent, well-defined, and realize the required coordinate relations.
- **Expression evaluation lemma:** Lowering completed expressions preserves guard and attribute evaluation, including checked arithmetic, indexing, and short-circuit behavior.
- **Acceptance theorem:** Successful semantic verification implies the contract in Section 4.3 for explicit and synthesized attribute programs.
- These are proof obligations for the paper; executable checks alone do not constitute formal proofs.

#### 5.4.2 Verification Boundary

- Check explicit attribute programs and synthesized programs against operation legality and tensor semantics.
- Re-lower the completed RHS in a fresh context.
- Recheck accesses, reduction domains, output order, extents, dtype, and legality.
- Treat synthesis obligations as goals, never additional assumptions.
- Constructor provenance and mapping records do not substitute for semantic verification.
- The verifier shares semantic infrastructure with synthesis; it is not a separately mechanized proof kernel.
- The semantic checker supports a limited set of attribute computations and conditions. An expression may execute even when the checker cannot prove the rewrite correct.
- Report why compilation failed: malformed input, unsupported reasoning, unresolved matching, or a proof-work limit. Expressions that cannot be lowered or proved are unsupported for semantic verification.
- Structural rules receive structural checks without a tensor-equivalence proof; keep this distinction explicit in reported guarantees.

#### 5.4.3 Completeness Boundary

- General completeness and weakest-precondition inference are not claimed.
- Investigate restricted completeness for a precisely defined fragment with fixed term correspondence, bijective axis routing, and directly solvable attribute holes.
- Establish such a theorem before presenting the fragment as complete.
- Unsupported cases include unresolved correspondence, general inverse equations, and exhausted proof budgets.

## 6. Implementation

- TEPL checks the rule, infers missing attributes, verifies supported semantic rewrites, and lowers completed rules for execution.
- Frontend and operation-defined semantics; no per-operation inference routine required for the supported fragment.
- `src/core/semantics/`: semantic lowering, completion, and verification orchestration.
- `semantics/index_maps/`: mapping storage, refinement, and transport.
- `semantics/symbolic/`: expression construction, normalization, and calculated access reasoning.
- `semantics/synthesis/` and `semantics/proof/`: attribute construction and semantic verification.
- `src/executable/`: runtime representation, validation, and serialization; completed rules contain no attribute holes or proof plans.
- `src/rewrite/`: direct greedy graph rewriting from executable programs.
- `src/codegen/`: Rust generation from executable programs; `runtime/rust/` provides shared metadata evaluation and egg integration. C++ generation remains unavailable.
- Modern semantic inference uses no Z3 search.
- Document numerical policy, bounded reasoning, implementation trust assumptions, and unsupported features.

## 7. Evaluation Plan

### 7.1 Research Questions

- **RQ1 — Authoring effort:** How much does TEPL simplify writing rewrites, and how much additional attribute logic does inference remove? What explicit expressions, conditions, and semantic definitions remain necessary?
- **RQ2 — Coverage:** Which practical rewrites can be executed, verified, and synthesized, including those that require newly constructed rather than copied descriptors?
- **RQ3 — Scalability:** How do composition depth, partition overlap, and expression size affect synthesis?
- **RQ4 — Generality:** Can new operations be supported through semantic definitions alone?
- **RQ5 — Reliability:** Does verification reject incorrect user-written constructors, corrupted synthesized constructors, and invalid rewrites?

### 7.2 Workloads and Comparisons

- Select and document a production-derived corpus with explicit inclusion/exclusion criteria.
- Include distribution, transpose/dot, guarded reassociation, reductions, broadcasts, slices, and supported convolution rewrites.
- Compare equivalent rewrites in three forms: handwritten host-language implementations, TEPL with explicit `derive` expressions, and TEPL with inferred attributes. Keep applicability conditions and numerical policies aligned.
- Use the explicit TEPL baseline to measure the language's benefit, then compare it with inference to measure the additional work synthesis removes.
- Include simple descriptor reuse and complex attribute reconstruction; report them separately.
- Use a feasible bounded-search baseline to assess synthesis cost, with its rank and attribute-domain limits stated.
- Compare related verifiers on shared expressible tasks; distinguish verification coverage from synthesis coverage.
- Report required user guards and semantic-definition effort, not only rule length.
- Pin production examples to revisions; separate attribute transformation from profitability, graph construction, and unrelated infrastructure.

### 7.3 Measurements and Ablations

- Per-rule attribute-code size and remaining explicit attribute expressions; distinguish eliminated logic from logic moved into shared builtins or semantic definitions.
- User-supplied guards and operation-definition size; report both initial semantic-specification cost and reuse across multiple rules.
- Compare total authored code for a rule corpus as well as per-rule savings; treat code size as an effort proxy, not evidence of usability by itself.
- Separate execution, verification, and synthesis coverage with explicit denominators; classify accepted, unsupported, invalid, and budget-limited cases.
- Synthesis/verification time, peak memory, constructor size, and matching-time cost in the greedy engine and generated Rust.
- Scale source composition depth, overlap patterns, repeated factors, and concrete constructor inputs.
- Disable reusable interfaces, lazy refinement, or mapping caches to isolate their effects.
- Test interleaved selections, empty groups, multiple contractions, rank-zero cases, and supported zero extents.
- Use independent numerical references, incorrect explicit constructors, and synthesized-constructor mutation tests as implementation evidence.
- Numerical testing supplements universal reasoning; it does not replace it.
- Leave all performance and coverage numbers unfilled until measured.

## 8. Related Work and Novelty Positioning

- **TensorRight:** Aggregated axes and arbitrary-rank rewrite verification are prior work; axis grouping alone is not a novelty claim. [POPL 2025](https://doi.org/10.1145/3704865)
- **Tensor rewrite synthesis and equality saturation:** Compare TASO/TENSAT on generated artifacts, semantic assumptions, and treatment of attributes; examine primary papers before asserting distinctions.
- **Semantics-driven synthesis:** Compare synthesis targets, parameterization, and correctness contracts with superoptimization and rewrite-synthesis systems.
- **Index-notation and affine compiler IRs:** Explain how TEPL reconstructs operation attributes from symbolic coordinate correspondence.
- **Rewrite DSLs and host callbacks:** Compare declarative authoring with manually implemented attribute transformations.
- Candidate distinction to substantiate: compositional synthesis of reusable, rank-polymorphic attribute programs from user-defined tensor semantics.
- Avoid “first” claims until a systematic literature review supports them.

## 9. Limitations and Future Work

- Incomplete inference; ambiguity and finite proof budgets can reject valid rules.
- Partition and normalized-expression growth across long compositions.
- Restricted convolution, broadcasting, and positive-size/positive-stride slice semantics.
- General padding, data-dependent indexing, coupled attribute equations, and unsupported numerical policies.
- Attribute computations and application conditions that the metadata language cannot express.
- Executable expressions outside the semantic checker's supported fragment; extending runtime support alone does not extend verification coverage.
- A smaller trusted verification kernel and broader expression verification.
- Further backend support and validation that expression semantics are preserved across execution paths.

## 10. Conclusion

- Reiterate the central result to establish: semantic coordinate correspondence can generate executable attribute transformations.
- Connect this result to the language: users write rewrite patterns and conditions, then supply attribute changes explicitly or ask the compiler to infer them.
- Summarize only guarantees and practical benefits supported by proofs and measurements.

## Appendix / Artifact

- Formal IR grammar, denotational semantics, mapping invariants, and algebraic side conditions.
- Full synthesis pseudocode and proof details.
- Benchmark definitions, guards, explicit and synthesized constructors, and failure classifications.
- Reproduction commands, versioned environments, and independent reference tests.
