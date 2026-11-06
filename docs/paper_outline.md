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

Tensor compilers optimize computations by applying algebraic rewrite rules.
Although these rules are often concise as equations, implementing them requires
more than replacing one sequence of operations with another. Tensor operations
carry attributes such as permutations, contraction dimensions, and broadcast
dimensions, which must change with the computation. XLA's dot reassociation
illustrates this gap: a short algebraic identity requires substantial logic to
map dimensions and construct the replacement operations' attributes.
[XLA implementation](https://github.com/openxla/xla/blob/main/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc)

The difficulty comes from tracking logical axes through composed operations.
Each operation can change axis positions, pairings, and output order, so
matching the source pattern does not by itself determine the attributes of the
replacement. Equal dimension sizes also do not imply that two axes represent
the same coordinates. Users must therefore write attribute transformations
that preserve these relationships across arbitrary tensor ranks, alongside
the rule's legality conditions and numerical requirements. This work motivates
a language that expresses attribute computations directly and can derive them
from operation semantics when supported.

- **Research question:** Can a tensor rewrite language let users express an algebraic transformation and its applicability conditions while deriving the required replacement attributes from operation semantics?
- TEPL combines explicit attribute construction with symbolic axis mapping and automatic RHS attribute inference under user-written conditions.
- **Evidence to prepare:** Pin the XLA example to a specific revision and identify the attribute-transformation code separately from other checks before reporting a line count.

**Figure 1 — Why TEPL?**

- [Four-panel storyboard](figures/why_tepl.yaml): symbolic rewriting, handwritten compiler code, TEPL manual, and TEPL automatic.
- **Caption:** TEPL combines concise rewrite rules with explicit or inferred attribute handling. The distribution example reuses dot attributes; automatic mode infers them from operation semantics.

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
      permutation: axis[];
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

- Running pattern: `(outer: transpose (inner: transpose X))`.
- Its intended replacement is one transpose of `X`; the remaining task is to construct its permutation.
- Tensor declarations may additionally constrain shape and dtype; equal dimension sizes alone do not establish coordinate identity.

### 3.3 Named Nodes and Attribute Reuse

- `(name: operation operands...)` optionally names an operation occurrence on either side.
- Every LHS operation captures its descriptor implicitly; a label does not capture an extra tensor.
- `source.attrs` refers to a whole descriptor and `source.attrs.permutation` to one field.
- `[source.attrs]` reuses the descriptor on a schema-compatible operation.
- `node.axes.G.members`, `.order`, and `.positions` distinguish logical membership, declaration order, and output positions.
- Unknown RHS descriptors are independent holes when attributes are omitted. Explicit `derive` assignments supply whole descriptors instead.
- Descriptor compatibility checks attribute structure; semantic verification establishes correctness.

### 3.4 Writing Attribute Changes

- Write conditions and compute replacement attributes directly in the language.
- Expressions can read captured attributes and available shape and dtype bindings; they do not execute tensor computations.
- Introduce local bindings, builtins, lists, indexing, conditionals, and comprehensions as examples need them.
- For transpose composition, `gather(inner.attrs.permutation, outer.attrs.permutation)` computes the replacement permutation.
- Descriptor constructors collect computed fields into immutable attributes for an RHS operation.
- Some expressions can execute but cannot yet be checked for semantic correctness by the compiler.

### 3.5 Conditions with `where`

- A `where` block ends with a condition that decides whether the rule applies. Write any local calculations before it. Only `let` bindings end with `;`.
- Conditions can use captured metadata, builtins, and local expressions. Combine predicates with `&&` and `||`.
- Example clause for the running transpose pattern:

```tepl
where {
  let rank = len(inner.attrs.permutation);
  len(outer.attrs.permutation) == rank
}
```

- The length condition illustrates attribute access; for a legal transpose composition it already follows from operation legality.
- Omit `where` when no additional condition is needed.
- A condition may express applicability or application policy; the compiler does not infer profitability decisions.
- Semantic rules can state axis-routing conditions such as `outer.attrs.lhs_batching_dimensions == inner.axes.B.positions`; introduce their meaning with the mapping algorithm.
- For verified rules, every correctness obligation must follow from operation legality and supported user-written conditions.

### 3.6 Attribute Construction with `derive`

- `derive` names replacement descriptors computed from the captured information.
- Combine captured attributes and an explicit descriptor constructor in one rule:

```tepl
rule compose_transposes_explicit {
  (outer: transpose (inner: transpose X))
  => (combined: transpose X)
  derive {
    combined.attrs = {
      permutation: gather(inner.attrs.permutation, outer.attrs.permutation)
    };
  }
}
```

- If the inner permutation is `p` and the outer permutation is `q`, the expression constructs a descriptor with permutation `r[i] = p[q[i]]`.
- `where` decides applicability; `derive` constructs attributes; the RHS specifies the replacement computation.
- Local bindings can name intermediate computations. A spread such as `{ ..outer.attrs, permutation: permutation }` copies fields before applying explicit overrides.
- `where` and `derive` have separate local scopes; derivations may read captured and earlier derived descriptors.
- With the transpose semantics, this explicit constructor is supported by semantic verification. Checking attribute types alone does not prove that the rewrite preserves the result.

### 3.7 Default Attribute Inference

- Omit replacement attributes to request synthesis from operation semantics. Node labels remain optional.

```tepl
rule compose_transposes_auto {
  (transpose (transpose X)) => (transpose X)
}
```

- Matching supplies tensor captures and LHS descriptors; semantic compilation derives a reusable program for the RHS permutation.
- In the explicit rule, the user writes how to combine the permutations. In this rule, the compiler derives that computation from transpose semantics: `r[i] = p[q[i]]`.
- Each omitted RHS descriptor is inferred independently. Use `[source.attrs]` for reuse or assign a complete `result.attrs` in `derive`. Partial attribute pins are not part of the language.
- Users still define operation semantics and any required correctness conditions. Unsupported inference is rejected with a diagnostic.
- Constructor synthesis occurs at compile time; application evaluates the constructor on matched metadata.
- Completed constructors execute in the native C++ greedy rewrite engine and in generated Rust/egg and C++/egg-c programs.

**Introducing attribute inference figure**

- [Storyboard](figures/introducing_auto.yaml): half-width introduction using transpose absorption into matrix dot. The TEPL compiler encloses synthesis, verification, and the generated function; runtime explicitly applies captured metadata to `f(p, q)` and attaches the resulting descriptor to the replacement dot.
- **Caption:** Omitting RHS attributes requests a replacement-attribute function. TEPL synthesizes and verifies that function from the rule and operation semantics at compile time. At runtime, the captured transpose permutation p and dot descriptor q feed `f(p, q)`, which constructs the replacement dot attributes. In this matrix example, the left contraction axis changes from 0 to 1 as the transpose is absorbed into the dot; only the changing field is shown.

### 3.8 From `auto` to Attribute Synthesis

- `auto` lets users leave RHS attributes to the compiler. **But how can the compiler find them?**
- Matching the LHS gives known attributes and tensor metadata.
- The compiler must turn that information into a program that computes the missing RHS attributes.
- That program must work across tensor ranks and shapes, and preserve the computation under the user's conditions.
- The next section defines this problem precisely.

## 4. Problem Definition

- The compiler must construct reusable RHS attribute programs and establish that the completed rewrite is correct. The following notation is shared by both requirements.

**Common notation**

| Symbol | Meaning |
| --- | --- |
| \(s\) | Source information: shapes and operation attributes available to the attribute program. |
| \(X\) | Input tensor values compatible with the shapes in \(s\). |
| \(a\) | RHS attributes to construct. |
| \(\operatorname{LHS}(s,X)\) | The tensor computed by the LHS. |
| \(\operatorname{RHS}(s,a,X)\) | The tensor computed by the RHS with attributes \(a\). |
| \(\operatorname{Guard}(s)\) | The user-written condition, which must evaluate successfully to `true`. |

Element types are fixed in this formulation.

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

**Formal Statement**

- Given the operation semantics, rewrite, and condition \(\operatorname{Guard}\), find an attribute program \(f\) that computes the RHS attributes from source information:

  \[
  a=f(s).
  \]

- The program \(f\) must satisfy the correctness condition in Section 4.2 for every legal source instance satisfying \(\operatorname{Guard}(s)\). It depends on source information, not tensor element values. An explicit constructor supplies \(f\); automatic inference constructs it for the missing attributes.

### 4.2 What Makes the Result Correct?

- The compiler must establish that replacement attributes can be computed, satisfy the replacement operations' requirements, and preserve the result whenever the rule's conditions hold.
- An explicit constructor supplies the attribute program; automatic inference asks the compiler to discover it. Both require checking for semantic rules.
- The attribute program must be evaluable and produce valid attributes for the RHS operations.
- With those attributes, the RHS must produce the same tensor as the LHS, under the selected arithmetic policy.
- The compiler must prove this for every legal input satisfying the user's conditions; otherwise, it rejects the rule as unproved.

**Formal Statement**

- The attribute program \(f\), whether supplied explicitly or synthesized, must satisfy:

\[
\forall s.\quad \operatorname{LegalLHS}(s)\land \operatorname{Guard}(s)
\Rightarrow
s\in\operatorname{dom}(f)\land
\operatorname{LegalRHS}(s,f(s))\land
\forall X.\ \operatorname{LHS}(s,X)=\operatorname{RHS}(s,f(s),X).
\]

- Read this as: **for every legal LHS satisfying the conditions, the computed RHS is legal and gives the same result for every input value.**
- \(s \in \operatorname{dom}(f)\) means the source information is valid input to the attribute program; operations such as inverse permutation and position lookup can be evaluated.
- Define \(\operatorname{Guard}(s)\) through successful guard evaluation to `true`. The formal account must cover ordered local evaluation, short-circuiting, indexing, and checked arithmetic, including evaluation failures.
- Equality includes ordered output shape and element values; exact algebra does not establish floating-point reassociation.

### 4.3 When Can the Compiler Accept a Rewrite?

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

**Figure 4 — Symbolic axis groups**

- [Storyboard](figures/symbolic_axis_groups.yaml): half-width figure with three stacked sections for arbitrary-length ordered groups, one concrete dot binding with a full-width position table, and output construction. Contraction axes appear in reversed physical order in Y, so K binds X positions `[2, 3]` to Y positions `[5, 2]`.
- **Caption:** Symbolic groups represent ordered axis sequences of arbitrary length. In this concrete instance, batch axes `b_a` and `b_b` share group `B`, and the contraction lists `[2, 3]` in X and `[5, 2]` in Y pair axes in list order despite their different physical arrangements. Reducing over `K` produces the ordered output `B ++ I ++ J`.

We describe these bindings using the following notation:

- **Tensor interface:** Let \(P=\langle a_0,\ldots,a_{n-1}\rangle\) denote a tensor's axes in dimension order. Its interface consists of \(P\) and the established mappings from axis groups into \(P\). Axis labels identify logical coordinates, not dimension sizes.
- **Axis group:** Let \(G=\langle g_0,\ldots,g_{m-1}\rangle\) be an ordered sequence of logical axes. Its order can differ from their order in the tensor.
- **Position sequence:** Let \(p=\langle p_0,\ldots,p_{m-1}\rangle\), where \(0\le p_j<n\) for \(0\le j<m\), locate the group in the tensor: \(g_j=a_{p_j}\). Positions are zero-based.
- **Mapping:** Define \(\operatorname{gather}(P,p)\) to select axes from \(P\) in the order specified by \(p\). A binding establishes the relationship:

  \[
  \operatorname{gather}(P,p)=G.
  \]

The following `dot_general` definition illustrates these bindings. It takes two tensor operands and four attributes specifying their batch and contraction axes. Each attribute is a list of axis positions. The excerpt shows the attribute declarations and axis bindings; dtype and value semantics are omitted here.

```tepl
op dot_general(lhs: tensor, rhs: tensor) -> tensor {
    attrs {
        lhs_batching_dimensions: axis[];
        rhs_batching_dimensions: axis[];
        lhs_contracting_dimensions: axis[];
        rhs_contracting_dimensions: axis[];
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

The operand axis orders give \(P\), and the contraction attributes give \(p\). Selecting the axes at those positions gives the contraction group \(G=K\) on each side:

\[
\begin{aligned}
P_{\mathrm{lhs}}&=\langle i,k_0,b,k_1\rangle,
&G_{\mathrm{lhs}}&=\langle k_0,k_1\rangle,
&p_{\mathrm{lhs}}&=\langle1,3\rangle,\\
P_{\mathrm{rhs}}&=\langle k_1,j,k_0,b\rangle,
&G_{\mathrm{rhs}}&=\langle k_0,k_1\rangle,
&p_{\mathrm{rhs}}&=\langle2,0\rangle.
\end{aligned}
\]

Both mappings identify the same ordered contraction group at different positions:

\[
\operatorname{gather}(P_{\mathrm{lhs}},p_{\mathrm{lhs}})
=G_{\mathrm{lhs}}=G_{\mathrm{rhs}}=K
=\operatorname{gather}(P_{\mathrm{rhs}},p_{\mathrm{rhs}}).
\]

- Here, `K` contains two axes. The same definition supports other group lengths without enumerating individual axes.
- The lengths \(n\) and \(m\), and the positions in \(p\), need not be known during rewrite compilation.
- Grouping preserves each axis separately and retains its order; it does not flatten the axes into one dimension.
- A group may be empty when the operation permits it, such as the batch group of a dot operation without batch axes.
- Group identities are scoped to distinguish unrelated coordinates with equal extents.

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

#### 5.1.3 Reasoning About Composed Operations

- **Value semantics provide a common language for tensor rewrites.**
  - Different tensor operations are expressed using a small set of value operations, including scalar addition, multiplication, and summation.
  - Algebraic rules over these value operations can support rewrites involving different tensor operations.
  - TEPL normalizes the LHS and RHS value expressions to check their equivalence under the permitted arithmetic laws.

- **Example: distributing a generalized dot over addition.**
  - Consider elementwise addition without broadcasting and identical dot attributes on both sides:

    ```text
    dot_general(X + Y, Z)
        ↔
    dot_general(X, Z) + dot_general(Y, Z)
    ```

  - Addition's local value expression reads operands whose axes are bound to the group `S`:

    ```tepl
    Add(X, Y)
    ```

  - The dot's local value expression uses its own batch, contraction, and free groups, reducing over `K`:

    ```tepl
    Reduce(K, Product(lhs, Z))
    ```

- **Composing value expressions requires identifying the axes they share.**
  - Simply substituting one operation's value expression into another does not establish how their axis groups correspond.
  - Replacing `lhs` with `Add(X, Y)` gives the expected arithmetic structure, but addition's operands are indexed through `S`, while the dot reduces over `K`.
  - TEPL must identify which positions in `S` correspond to `K` and preserve their pairing with the contraction axes of `Z`.

- **Symbolic axis mapping establishes this correspondence.**
  - The dot's contraction attribute selects axis positions in addition's output. Addition preserves axis order, so those positions identify the corresponding axes in both `X` and `Y`.
  - Symbolic axis mapping records this correspondence, allowing the dot's reduction and operand pairings to apply consistently to the reads inside `Add(X, Y)`.
  - The same process connects the batch and free axes, including when each group contains an unknown number of axes.

- **Once the mappings are established, the composed expressions can be compared.**
  - With fixed output coordinates and operand indices omitted, the two sides are:

    \[
    \sum_K (X + Y)Z
    \qquad\text{and}\qquad
    \sum_K XZ + \sum_K YZ.
    \]

  - In TEPL's value-expression notation, the LHS is:

    ```tepl
    Reduce(K, Product(Add(X, Y), Z))
    ```

  - The RHS is:

    ```tepl
    Add(Reduce(K, Product(X, Z)),
        Reduce(K, Product(Y, Z)))
    ```

  - Here, `X`, `Y`, and `Z` denote operand element reads under the aligned axis bindings, and `K` denotes the corresponding contraction group on both sides.
  - Distributing multiplication over addition and summation over addition brings these expressions to a common form.

  - Section 5.2 explains how these mappings are represented, composed, and refined.

### 5.2 Symbolic Axis Mapping

Symbolic axis mapping connects the axis groups defined by different operations. It tracks where each group appears in a tensor, allowing TEPL to compose value expressions while preserving axis order and operand pairings.

#### 5.2.1 Building and Composing Mapping Tables

- TEPL constructs axis mappings while composing operation semantics. Each record \((P,G,p)\) satisfies \(\operatorname{gather}(P,p)=G\).
- We display these relationships as tables: each column represents a tensor, the first row gives its current axis interface, and each group row gives that group's positions. A dash indicates no mapping for that group in the example. The tables include both registered and derivable mappings.
- Consider `T = Add(X, Y)` followed by `O = dot_general(T, Z)`. Let \(\rho(G)=\operatorname{range}(|G|)\) denote all positions in \(G\). Ranks and position attributes remain symbolic throughout the example.

**1. Build addition's interface**

- Addition binds the axes of `X` and `Y` to one ordered group \(S\) and yields \(S\). Its operand reads and result therefore have the following mappings:

| Interface / group | `X` | `Y` | `T` |
| --- | --- | --- | --- |
| Axis interface | \(S\) | \(S\) | \(S\) |
| \(S\) | \(\rho(S)\) | \(\rho(S)\) | \(\rho(S)\) |

**2. Establish the dot's operand layouts**

- The dot's bindings introduce batch group \(B\), contraction group \(K\), and free groups \(I\) and \(J\). Let \(b_L,b_R\) be the batch attributes, \(c_L,c_R\) the contraction attributes, and \(R\) the initial axis interface of `Z`. The free-axis positions are:

  \[
  \begin{aligned}
  f_L&=\operatorname{ordered\_complement}(\rho(S),b_L\mathbin{+\!\!+}c_L),\\
  f_R&=\operatorname{ordered\_complement}(\rho(R),b_R\mathbin{+\!\!+}c_R).
  \end{aligned}
  \]

- Establishing these bindings requires valid position lists and matching extents for paired axes. The groups must cover each operand without overlap.
- How can we represent the operand's axis order using the dot's groups \(B\), \(K\), and \(I\)? Their axes may be interleaved, so simply concatenating the groups does not necessarily preserve that order. TEPL uses the binding positions to place each axis in its original position:

  \[
  \begin{aligned}
  \widehat S&=\operatorname{gather}\bigl(B\mathbin{+\!\!+}K\mathbin{+\!\!+}I,
      \operatorname{inverse}(b_L\mathbin{+\!\!+}c_L\mathbin{+\!\!+}f_L)\bigr),\\
  \widehat R&=\operatorname{gather}\bigl(B\mathbin{+\!\!+}K\mathbin{+\!\!+}J,
      \operatorname{inverse}(b_R\mathbin{+\!\!+}c_R\mathbin{+\!\!+}f_R)\bigr).
  \end{aligned}
  \]

- Each concatenated position list is a permutation because the bindings form a complete, disjoint partition of the operand. Thus, the constructed left interface satisfies:

  \[
  \operatorname{gather}(\widehat S,b_L)=B,\qquad
  \operatorname{gather}(\widehat S,c_L)=K,\qquad
  \operatorname{gather}(\widehat S,f_L)=I.
  \]

- The right interface satisfies the corresponding bindings. Register these mappings and the operand partitions:

| Interface / group | `T` | `Z` |
| --- | --- | --- |
| Axis interface | \(\widehat S\) | \(\widehat R\) |
| \(B\) | \(b_L\) | \(b_R\) |
| \(K\) | \(c_L\) | \(c_R\) |
| \(I\) | \(f_L\) | — |
| \(J\) | — | \(f_R\) |

- \(\widehat S\) and \(\widehat R\) describe the operands using the dot's coordinates. This changes the symbolic coordinate description, not the tensors' physical axis order. In general, TEPL reuses an existing producer interface when it already satisfies the bindings.

**3. Apply the dot's axis mappings to `X` and `Y`**

- Addition describes `T`'s axes using \(S\), while the dot describes those same axes using \(\widehat S\). To combine their value expressions, TEPL must connect these two descriptions so the dot's batch, contraction, and free groups refer to the correct axes in `X` and `Y`.
- Addition preserves axis order, so each axis of `T = Add(X, Y)` corresponds to the same-position axes in `X` and `Y`. When the dot describes `T` using \(\widehat S\), TEPL applies that same description to `X` and `Y`, replacing their original axis description \(S\).
- Since `X`, `Y`, and `T` have corresponding axes in the same order, the positions \(b_L\), \(c_L\), and \(f_L\) identify the batch, contraction, and free axes in all three tensors.
- As a result, `X` and `Y` use the dot's coordinate groups: \(B\), \(K\), and \(I\) appear at positions \(b_L\), \(c_L\), and \(f_L\), respectively. With these mappings established, TEPL can compose the value expression:

  ```tepl
  Reduce(K, Product(Add(X, Y), Z))
  ```

- The reduction over \(K\) now refers to corresponding coordinates in `X`, `Y`, and `Z`.

**4. Extend the mapping table to the output**

- We now know how the dot's groups map to its inputs, including `X` and `Y`. A later operation, however, consumes the dot's output, where the surviving axes are arranged in output order and \(K\) is absent. We therefore extend the mapping table with each surviving group's output positions so subsequent operations can locate them.
- The dot yields \(B\mathbin{+\!\!+}I\mathbin{+\!\!+}J\). Register the output positions of these groups; \(K\) is reduced and does not appear in the result.
- The following table shows the interfaces used in the composed expression. The row \(\widehat S\) is addition's whole-axis group after the coordinate update:

| Interface / group | `X` | `Y` | `T` | `Z` | `O` |
| --- | --- | --- | --- | --- | --- |
| Axis interface | \(\widehat S\) | \(\widehat S\) | \(\widehat S\) | \(\widehat R\) | \(B\mathbin{+\!\!+}I\mathbin{+\!\!+}J\) |
| \(\widehat S\) | \(\rho(\widehat S)\) | \(\rho(\widehat S)\) | \(\rho(\widehat S)\) | — | — |
| \(B\) | \(b_L\) | \(b_L\) | \(b_L\) | \(b_R\) | \(\rho(B)\) |
| \(K\) | \(c_L\) | \(c_L\) | \(c_L\) | \(c_R\) | — |
| \(I\) | \(f_L\) | \(f_L\) | \(f_L\) | — | \(\lvert B\rvert+\rho(I)\) |
| \(J\) | — | — | — | \(f_R\) | \(\lvert B\rvert+\lvert I\rvert+\rho(J)\) |

- The \(B,K,I\) entries in `X` and `Y` follow from transporting the dot's partition into addition's reads. The entries in `O` follow from the output order. Adding a length to a position sequence shifts every position by that amount.

**5. Find a group's positions through existing mappings**

- A group's positions may not be available directly. For example, suppose we know where \(\widehat S\) appears in `X`, and where \(K\) appears within \(\widehat S\). How can we find \(K\)'s positions in `X`?
- Let \(p_X\) locate \(\widehat S\) in `X`. The dot's contraction positions \(c_L\) locate \(K\) within \(\widehat S\):

  \[
  \operatorname{gather}(P_X,p_X)=\widehat S,
  \qquad
  \operatorname{gather}(\widehat S,c_L)=K.
  \]

- Select the entries of \(p_X\) at positions \(c_L\). These entries give \(K\)'s positions in `X`. The derivation follows by composing the two selections:

  \[
  \begin{aligned}
  K&=\operatorname{gather}(\widehat S,c_L)\\
   &=\operatorname{gather}(\operatorname{gather}(P_X,p_X),c_L)\\
   &=\operatorname{gather}(P_X,\operatorname{gather}(p_X,c_L)).
  \end{aligned}
  \]

- Therefore, row \(K\), column `X` contains \(\operatorname{gather}(p_X,c_L)\). For addition, \(p_X\) lists all positions in order, so this simplifies to \(c_L\).
- This composition also works across longer chains of mappings, allowing a group's positions to be recovered without storing every relationship directly.

Addition exposes one axis group, making the correspondence straightforward. When a consumer selects axes across several producer groups, those groups must be partitioned to identify the required correspondences.

#### 5.2.2 Symbolic Axis Partitioning

**Symbolic axis partitioning figure**

- [Storyboard](figures/symbolic_axis_partitioning.yaml): full-width 1000 × 600 px figure organized as selection, partition by producer origin, and preservation of contraction pairings, followed by a symbolic mapping table.
- **Caption:** Partitioning retains three distinct descriptions of each selected axis: its position within the producer group, its slot in the consumer's contraction list, and its partner's position in Z. In this unbatched instance, left positions `[3, 0]` pair with right positions `[2, 0]`, making contraction slots distinct from tensor-axis positions. The retained mappings identify the subgroups needed by the reassociated dots.

**1. Why reassociation requires splitting groups**

- Dot reassociation changes the grouping of a product:

  \[
  (XY)Z\;\longrightarrow\;X(YZ).
  \]

- For generalized dots, constructing this replacement also requires determining which axes each dot contracts. Consider the following rule:

  ```tepl
  rule associate {
    (outer: dot (inner: dot X Y) Z)
    => (dot X (dot Y Z))
    where {
      outer.attrs.lhs_batching_dimensions == inner.axes.B.positions
    }
  }
  ```

- Generalized-dot reassociation can change the result if the two dots use incompatible batch axes. Here, the `where` clause requires the outer dot to use the inner dot's batch axes as its left batch axes, in the same order. `inner.axes.B.positions` gives their positions in the inner result. The rule applies only when this condition holds.
- The batch condition restricts when the rule applies; the completed replacement is checked for correctness as described in Section 5.3.
- Let `T = dot_general(X, Y)`. The inner dot contracts group \(K\) and produces the interface

  \[
  P_T=B\mathbin{+\!\!+}I\mathbin{+\!\!+}J,
  \]

  where \(I\) contains the free axes from `X` and \(J\) those from `Y`.
- Let \(c_L^{\mathrm{outer}}\) and \(c_R^{\mathrm{outer}}\) denote the outer dot's contraction positions in `T` and `Z`, respectively. The outer contraction group is \(L=\operatorname{gather}(P_T,c_L^{\mathrm{outer}})\). Since \(B\) remains batch, \(c_L^{\mathrm{outer}}\) selects axes from \(I\) and \(J\).
- In the addition–dot example, addition preserves one axis group \(S\), so the dot's mappings can be applied directly to both addition operands. Here, the inner dot's output contains separate groups \(I\) and \(J\), originating from `X` and `Y`. The outer dot may contract only part of each group, combining those selected axes into one contraction group \(L\).
- The existing mappings tell us where \(I\) and \(J\) appear, but constructing `dot_general(Y, Z)` requires knowing which axes of \(L\) came from each group. Axes from \(J\) can be contracted with `Y` in this new dot. The partners of axes from \(I\) must remain available for the final dot with `X`.
- How can we split \(I\) and \(J\) into their selected and remaining axes while preserving each selected axis's pairing with `Z`? Symbolic axis partitioning establishes these finer relationships.

**2. Find the selected part of each producer group**

- We solve this by separating the outer dot's contraction axes according to where they came from. For each producer group \(I\) or \(J\), identify which of its axes the outer dot selects and which remain. For each selected axis, retain both its position within the producer group and its slot in the outer contraction sequence. The first locates the axis in `X` or `Y`; the second identifies its partner in `Z`.

- Start with \(I\), the group of axes originating in `X`. The outer dot selects axes by their positions in `T`, so we compare its contraction positions with the positions occupied by \(I\).
- Since `T` orders its axes as \(B\mathbin{+\!\!+}I\mathbin{+\!\!+}J\), \(I\) starts after \(B\). Its positions ari
  p_I=|B|+\rho(I).
  \]

- Use `matchPositions` to find which outer contraction positions belong to \(I\):

  \[
  (u_I,v_I)=\operatorname{matchPositions}(c_L^{\mathrm{outer}},p_I).
  \]

  > **How `matchPositions` works**
  >
  > - Define \(\operatorname{matchPositions}(a,b)=(u,v)\) for two position sequences with distinct entries. It finds all values shared by \(a\) and \(b\) and returns their locations in both sequences, following the order of \(a\):
  >
  >   - Scan \(a\) from left to right. If \(a_i\) also occurs at position \(j\) in \(b\), append \(i\) to \(u\) and \(j\) to \(v\); otherwise, skip it.
  >   - The outputs contain positions within the two input sequences, not the shared values themselves. Thus, \(\operatorname{gather}(a,u)=\operatorname{gather}(b,v)\). If there are no matches, both outputs are empty.
  >
  > - For example:
  >
  >   \[
  >   a=\langle3,0,4\rangle,\quad b=\langle0,2,3\rangle
  >   \quad\Longrightarrow\quad
  >   u=\langle0,1\rangle,\quad v=\langle2,0\rangle.
  >   \]
  >
  >   The value `3` occurs at positions `0` and `2`, and `0` at positions `1` and `0`. The value `4` has no match. Both gathers therefore produce \(\langle3,0\rangle\), preserving the order in \(a\).
  >
  > - This scan defines the operation's meaning for concrete sequences. For symbolic inputs, the two results remain symbolic position expressions until enough information is available to evaluate or simplify them.

- Here, \(u_I\) gives their slots in the outer contraction sequence, while \(v_I\) gives their positions within \(I\). Both identify the same positions in `T`:

  \[
  \operatorname{gather}(c_L^{\mathrm{outer}},u_I)
  =\operatorname{gather}(p_I,v_I).
  \]

- Selecting those axes from \(I\), or from the outer contraction group \(L\), therefore gives the same subgroup:

  \[
  I_{\mathrm{sel}}
  =\operatorname{gather}(I,v_I)
  =\operatorname{gather}(L,u_I).
  \]

- The remaining axes are selected by the ordered complement of \(v_I\):

  \[
  w_I=\operatorname{ordered\_complement}(\rho(I),v_I),\qquad
  I_{\mathrm{rem}}=\operatorname{gather}(I,w_I).
  \]

- Thus, \(I_{\mathrm{sel}}\) contains the axes originating in `X` that the outer dot contracts, and \(I_{\mathrm{rem}}\) contains those it preserves. Retaining \(u_I\) lets us locate the selected axes' partners in `Z`.
- The selected view follows the outer contraction order; the remaining view follows the original order within \(I\). Their concatenation need not reproduce \(I\), so retain their position sequences.
- Apply the same construction to \(J\), using its positions after \(B\) and \(I\):

  \[
  p_J=|B|+|I|+\rho(J).
  \]

- In the following discussion, use \(G\in\{I,J\}\) for either producer group. The corresponding \(u_G,v_G\) identify its selected axes, and \(G_{\mathrm{sel}}\) and \(G_{\mathrm{rem}}\) denote its selected and remaining views.
- These expressions remain symbolic. The construction does not require knowing the lengths of \(I\), \(J\), or their selected portions.

- Partitioning adds mappings for the selected and remaining portions of \(I\) and \(J\). Each subgroup retains its positions in the producer interface and its original group. Selected subgroups also retain their slots in the outer contraction group \(L\). For either group, the remaining positions are \(w_G=\operatorname{ordered\_complement}(\rho(G),v_G)\).

| Group | Positions in \(P_T\) | Positions in \(I\) | Positions in \(J\) | Positions in \(L\) |
| --- | --- | --- | --- | --- |
| \(I_{\mathrm{sel}}\) | \(\operatorname{gather}(p_I,v_I)\) | \(v_I\) | — | \(u_I\) |
| \(I_{\mathrm{rem}}\) | \(\operatorname{gather}(p_I,w_I)\) | \(w_I\) | — | — |
| \(J_{\mathrm{sel}}\) | \(\operatorname{gather}(p_J,v_J)\) | — | \(v_J\) | \(u_J\) |
| \(J_{\mathrm{rem}}\) | \(\operatorname{gather}(p_J,w_J)\) | — | \(w_J\) | — |

- The \(L\) column provides the connection to `Z`: its slot sequences \(u_I\) and \(u_J\) identify the corresponding right-hand contraction axes, as explained in subsection 3.

**3. Map the selected axes to `Z`**

- We now know which contracted axes come from \(I\) and which come from \(J\). But constructing the replacement dots also requires knowing which axes in `Z` are paired with each selected subgroup.
- The outer dot pairs its left and right contraction lists in order: the first left axis pairs with the first right axis, the second with the second, and so on. Therefore, an axis's slot in the left contraction list also identifies its partner's slot in the right list.
- Subsection 2 gives us those slots: \(u_I\) for \(I_{\mathrm{sel}}\) and \(u_J\) for \(J_{\mathrm{sel}}\). Selecting the same slots from the right contraction list gives their partners' positions in `Z`:

  \[
  \operatorname{gather}(c_R^{\mathrm{outer}},u_I),
  \qquad
  \operatorname{gather}(c_R^{\mathrm{outer}},u_J).
  \]

- In the figure's unbatched example, `Z` has axes `[z₀, f, z₁]` and right contraction positions `[2, 0]`. The left contraction list selects `[j₁, i₀]`: `j₁` occupies slot `0` and pairs with `z₁` at Z position `2`; `i₀` occupies slot `1` and pairs with `z₀` at Z position `0`.
- We can now extend the subgroup mapping table with positions in `Z`:

| Selected group | Positions within producer group | Slots in \(L\) | Positions in `Z` |
| --- | --- | --- | --- |
| \(I_{\mathrm{sel}}\) | \(v_I\) | \(u_I\) | \(\operatorname{gather}(c_R^{\mathrm{outer}},u_I)\) |
| \(J_{\mathrm{sel}}\) | \(v_J\) | \(u_J\) | \(\operatorname{gather}(c_R^{\mathrm{outer}},u_J)\) |

- **Result:** Each selected subgroup now has a mapping to its paired axes in `Z`. These mappings can be combined with the producer mappings to describe the original computation using the refined groups.

**4. Extend the original mapping table with the subgroups**

- Partitioning identifies finer relationships within the original computation:

  ```text
  Inputs: X, Y, Z
  T = dot_general(X, Y)
  O = dot_general(T, Z)
  ```

- We now know which parts of \(I\) and \(J\) the outer dot contracts and where their partners occur in `Z`. To complete the mapping table, locate these subgroups in `X` and `Y`, then add the output positions of the groups that survive.
- Write \(b_X,k_X,i_X\) for the existing positions of \(B,K,I\) in `X`, and \(b_Y,k_Y,j_Y\) for those of \(B,K,J\) in `Y`. Composing these mappings with the subgroup positions gives the new entries. For example, \(I_{\mathrm{sel}}\) appears at \(\operatorname{gather}(i_X,v_I)\) in `X`.
- In `Z`, let \(b_Z\) be the batch positions and \(f_Z\) the positions of its free group \(F\). Abbreviate the selected partners found in subsection 3 as:

  \[
  z_I=\operatorname{gather}(c_R^{\mathrm{outer}},u_I),\qquad
  z_J=\operatorname{gather}(c_R^{\mathrm{outer}},u_J).
  \]

- The inner dot contracts \(K\), so it is absent from `T`. The outer dot contracts \(I_{\mathrm{sel}}\) and \(J_{\mathrm{sel}}\), so they are absent from `O`. Its surviving axes appear in the order:

  \[
  P_O=B\mathbin{+\!\!+}I_{\mathrm{rem}}
      \mathbin{+\!\!+}J_{\mathrm{rem}}\mathbin{+\!\!+}F.
  \]

- **Result:** The table below combines the mappings for the refined groups across the original computation. \(P_A\) denotes the ordered interface of tensor `A`. Each cell gives positions in that interface; a dash indicates no mapping for that group in the example.

| Group / interface | `X` | `Y` | `T` | `Z` | `O` |
| --- | --- | --- | --- | --- | --- |
| Axis interface | \(P_X\) | \(P_Y\) | \(P_T\) | \(P_Z\) | \(P_O\) |
| \(B\) | \(b_X\) | \(b_Y\) | \(\rho(B)\) | \(b_Z\) | \(\rho(B)\) |
| \(K\) | \(k_X\) | \(k_Y\) | — | — | — |
| \(I_{\mathrm{sel}}\) | \(\operatorname{gather}(i_X,v_I)\) | — | \(\operatorname{gather}(p_I,v_I)\) | \(z_I\) | — |
| \(I_{\mathrm{rem}}\) | \(\operatorname{gather}(i_X,w_I)\) | — | \(\operatorname{gather}(p_I,w_I)\) | — | \(\lvert B\rvert+\rho(I_{\mathrm{rem}})\) |
| \(J_{\mathrm{sel}}\) | — | \(\operatorname{gather}(j_Y,v_J)\) | \(\operatorname{gather}(p_J,v_J)\) | \(z_J\) | — |
| \(J_{\mathrm{rem}}\) | — | \(\operatorname{gather}(j_Y,w_J)\) | \(\operatorname{gather}(p_J,w_J)\) | — | \(\lvert B\rvert+\lvert I_{\mathrm{rem}}\rvert+\rho(J_{\mathrm{rem}})\) |
| \(F\) | — | — | — | \(f_Z\) | \(\lvert B\rvert+\lvert I_{\mathrm{rem}}\rvert+\lvert J_{\mathrm{rem}}\rvert+\rho(F)\) |

- These mappings identify the contracted and preserved portions of each producer group while retaining their operand pairings. Section 5.3 uses these relationships to construct the reassociated expression and its attributes.

The same procedure applies to other producer–consumer pairs. Compare their groups' position sequences on the shared interface, identify overlapping portions, and add mappings from each overlap to both groups. Each partition remains internally disjoint, while groups from different partitions may overlap.

#### 5.2.3 General Construction Algorithm

- The algorithm combines the two preceding examples. It processes producers before consumers, connects each operation's axis bindings to its inputs, and extends the mappings to its output. When groups overlap across operations, it adds the subgroup relationships from Section 5.2.2.
- **Input:** A source computation graph and its operation definitions.
- **Output:** Tensor interfaces, composed value expressions, and their established mappings and partitions.

**State: what the algorithm keeps**

- Write the current state as \(\Sigma=(V,M,\Pi)\):

| Component | Meaning |
| --- | --- |
| \(V\) | For each tensor `t`, \(V[t]=(P_t,e_t)\) stores its ordered axis interface and value expression. |
| \(M\) | The mapping table, represented here as a set of entries \((P,G,p)\). |
| \(\Pi\) | Which groups together cover an interface without overlap; called `partitions` in the pseudocode. Unlike \(M\), it describes a complete grouping of the axes, rather than an individual group's positions. |

- A partition groups all axes of an interface \(P\). Represent it as:

  \[
  \pi=\bigl(P,\langle(G_1,p_1),\ldots,(G_m,p_m)\rangle\bigr).
  \]

  Each pair \((G_i,p_i)\) gives one group and its positions in \(P\). An interface may have several partitions because different operations group its axes differently.

- Applying a partition changes how the axes are grouped for reasoning. It does not change the tensor's shape, rank, or axis order. The entries in \(M\) locate the resulting subgroups; \(\Pi\) identifies which subgroups together cover the interface exactly once.

> **Example**
>
> Consider `T = Add(X, Y)` with axis group \(S=\langle i,j\rangle\). Addition preserves both axes, so \(P_T=S\). The state includes these entries for `T`:
>
> | Component | Example | Meaning |
> | --- | --- | --- |
> | \(V\) | \(V[T]=(P_T,\operatorname{Add}(X,Y))\) | `T` has axes \(P_T\) and computes values by addition. |
> | \(M\) | \((P_T,S,\langle0,1\rangle)\) | Group \(S\) occupies positions `0` and `1` in `T`. |
> | \(\Pi\) | \((P_T,\langle(S,\langle0,1\rangle)\rangle)\) | The single group \(S\) covers all axes of `T`. |
>
> Suppose a consumer splits \(S\) into \(I=\langle i\rangle\) and \(J=\langle j\rangle\). If `T` has shape \(\langle d_i,d_j\rangle\), that shape stays the same: the partition only describes its two axes separately. \(M\) gains mappings locating \(I\) at position `0` and \(J\) at position `1`. \(\Pi\) gains the information that \(I\) and \(J\) form a complete, disjoint partition of \(P_T\), alongside the existing partition containing only \(S\).

**Invariants: what must remain true**

- Every mapping entry identifies the stated group in the stated order:

  \[
  (P,G,p)\in M
  \quad\Longrightarrow\quad
  \operatorname{gather}(P,p)=G.
  \]

- Every partition member satisfies this same relationship. Its concatenated position sequence

  \[
  p_1\mathbin{+\!\!+}\cdots\mathbin{+\!\!+}p_m
  \]

  is a permutation of \(\rho(P)\). In other words, the partition covers every axis exactly once. Groups within one partition are disjoint; groups from different partitions may overlap.

> **Example**
>
> The addition entry above selects the correct axes:
>
> \[
> \operatorname{gather}(\langle i,j\rangle,\langle0,1\rangle)
> =\langle i,j\rangle=S.
> \]
>
> Its partition also covers every axis exactly once, using positions \(\langle0,1\rangle\). A sequence such as \(\langle0,0\rangle\) would repeat the first axis and omit the second, violating the partition invariant.

**Algorithm 1: Build symbolic axis mapping**

```python
def build_symbolic_axis_mapping(graph):
    V = initialize_inputs(graph)
    M = set()       # Entries (interface, group, positions)
    partitions = {}

    for op in producer_first(graph):
        inputs = [V[t] for t in op.inputs]
        groups, bindings = resolve_bindings(op, inputs)
        aligned = []

        for (P, expression), operand_bindings in zip(inputs, bindings):
            Q = build_interface(P, operand_bindings)

            for group, positions in operand_bindings:
                M.add((Q, group, positions))

            refine_and_register(Q, operand_bindings, M, partitions)
            expression = transport(expression, P, Q, M, partitions)
            aligned.append((Q, expression))

        Pout = evaluate(op.output_axes, groups)
        output_bindings = list(output_positions(op, groups))

        for group, positions in output_bindings:
            M.add((Pout, group, positions))

        refine_and_register(Pout, output_bindings, M, partitions)
        V[op] = (Pout, compose(op.value, aligned, groups))

    return V, M, partitions
```

**How the main operations work**

- `initialize_inputs` gives each graph input an initial axis interface and an operand-read expression. `producer_first` visits operations after their producers, so each input description is available when needed.
- `resolve_bindings` evaluates the operation's symbolic position expressions and establishes the required axis-validity and pairing conditions. It reuses producer coordinates where applicable and introduces local groups otherwise. Each operand's bindings give pairs \((G_i,p_i)\) covering its axes.
- `build_interface` reuses \(P\) if its groups already occur at the required positions. Otherwise, it constructs the interface described in Section 5.2.1:

  \[
  Q=\operatorname{gather}\left(
    G_1\mathbin{+\!\!+}\cdots\mathbin{+\!\!+}G_m,
    \operatorname{inverse}(p_1\mathbin{+\!\!+}\cdots\mathbin{+\!\!+}p_m)
  \right).
  \]

  Concatenation collects the groups; the inverse position sequence arranges their axes in tensor dimension order.

- `refine_and_register` adds the new partition to \(\Pi\) and compares it with the previously established partitions of the same interface. For overlapping groups, it uses `matchPositions` to add mappings for their shared portions, as in Section 5.2.2.
- `transport` updates the producer's operand-read coordinates from \(P\) to \(Q\) and carries their mappings and partitions into that description. If the interfaces are equal, no coordinate update is needed. These updates apply to the current consumer's input expressions.
- `evaluate` and `output_positions` obtain the result interface and group positions from the declared output order. `compose` combines the updated operand expressions using the operation's value definition. The resulting pair is stored in \(V\) for later consumers.

**Deriving a mapping when needed**

- If a requested mapping is missing, follow established entries and compose their position sequences:

  \[
  (P,G,p)\in M
  \;\land\;
  (G,H,q)\in M
  \quad\Longrightarrow\quad
  \operatorname{gather}(P,\operatorname{gather}(p,q))=H.
  \]

- The derived positions \(\operatorname{gather}(p,q)\) may be returned by lookup or added as the entry \((P,H,\operatorname{gather}(p,q))\). Some mappings are propagated during construction; others are derived when queried. There is no need to store every possible relationship in advance.
- Local mappings follow from the operation bindings and output order. Overlap mappings follow from paired position matches, and composed mappings follow from the gather composition identity. These steps preserve the mapping invariant. The completed rewrite still requires the correctness checks in Section 5.3.

### 5.3 Attribute Program Synthesis

- Section 5.2 establishes which logical axes occur in each source tensor and where they occur. How can these relationships determine the missing attributes of a replacement? We first normalize the value expressions, determine which axes each replacement operation must preserve or reduce, and then use the mapping table to construct its attributes.
- Continue the reassociation example, under an arithmetic policy that permits the required algebraic laws:

  \[
  \underbrace{\operatorname{dot}(\operatorname{dot}(X,Y),Z)}_{\text{LHS}}
  \;\longrightarrow\;
  \underbrace{\operatorname{dot}(X,\operatorname{dot}(Y,Z))}_{\text{RHS}}.
  \]

- The rule supplies the replacement graph. Synthesis fills its attribute holes. Write `T = dot(X, Y)` for the original intermediate, `U = dot(Y, Z)` for the replacement intermediate, and `O` and `O'` for their respective final results.

#### 5.3.1 Value Expression Normalization

- The two graphs combine their operands in different orders. Before constructing attributes, how can we identify the same computation within both expressions?
- From Section 5.2, the original inner dot reduces \(K\), and the outer dot reduces \(L\). With fixed output coordinates and operand indices omitted, its value expression is:

  \[
  e_{\mathrm{LHS}}=\sum_L\left(\sum_K XY\right)Z.
  \]

  Here, each sum ranges over the coordinates of its axis group. The operand reads retain the coordinate mappings established in Section 5.2; the abbreviated notation only hides them.

- We use a sum-of-products representation similar to the R-polynomial form of SPORES. Each term stores a coefficient, indexed operand reads, and symbolic reduction axes. SPORES likewise describes weighted sums of terms, each summing a product of indexed atoms over bound variables. [SPORES, Section 2.3, Definition 2.4](https://www.vldb.org/pvldb/vol13/p1919-wang.pdf#page=4)

  > **Normalized terms: syntax and meaning**
  >
  > A normalized term has the syntax
  >
  > \[
  > t=(c,[r_1,\ldots,r_m],R).
  > \]
  >
  > | Field | Meaning |
  > | --- | --- |
  > | \(c\) | A scalar coefficient multiplying the term. |
  > | \([r_1,\ldots,r_m]\) | Operand reads multiplied together. Each read retains its tensor identity and coordinate expression. |
  > | \(R\) | The ordered sequence of symbolic axes over which the product is summed. |
  >
  > The term denotes:
  >
  > \[
  > \operatorname{value}(t)=\sum_R c\prod_{i=1}^{m}r_i.
  > \]
  >
  > An empty reduction sequence, written `[]`, means no summation. An empty read list has product \(1\), allowing a term to represent a constant.
  >
  > A normalized expression is a list of terms \(N=[t_1,\ldots,t_n]\), whose values are added:
  >
  > \[
  > \operatorname{value}(N)=\sum_{j=1}^{n}\operatorname{value}(t_j).
  > \]
  >
  > Thus, reads within one term are multiplied; separate terms are added. An empty term list represents zero.
  >
  > **Examples.** Here, `X`, `Y`, and `Z` abbreviate indexed operand reads with their coordinates omitted.
  >
  > | Expression | Normalized term list |
  > | --- | --- |
  > | \(XY\) | `[(1, [X, Y], [])]` |
  > | \(X(Y+Z)=XY+XZ\) | `[(1, [X, Y], []), (1, [X, Z], [])]` |
  > | \(2XY-Z\) | `[(2, [X, Y], []), (-1, [Z], [])]` |
  > | \(\sum_K XY\) | `[(1, [X, Y], K)]` |

- Under the permitted multiplication and reduction laws, normalize the example as follows:

  \[
  \sum_L\left(\sum_K XY\right)Z
  =\sum_L\sum_K XYZ
  =\sum_{L,K}XYZ.
  \]

  Normalization collects the operand reads into one product and concatenates their reduction axes. The inner reduction contributes reads `X` and `Y` with reduction axes \(K\). Multiplication adds the read `Z`, after checking that it does not depend on \(K\). The outer reduction then prepends \(L\), after checking that \(L\) and \(K\) are disjoint.

  The resulting term stores reads `X`, `Y`, and `Z` with the single reduction sequence \(L\mathbin{+\!\!+}K\):

  ```text
  Value expression:
  Reduce(L,
    Product(
      Reduce(K, Product(X, Y)),
      Z))

  Normalized term:
  t = (1, [X, Y, Z], L ++ K)
  ```

- The RHS attributes are still unknown, so its reduction axes and operand coordinates cannot yet be determined. Instead, normalize its **arithmetic skeleton**: expand the declared value operations while temporarily omitting axis information. For `X(YZ)`, this produces a term with coefficient `1`, reads `[X, Y, Z]`, and an empty reduction sequence. The empty sequence belongs only to the skeleton; it does not assert that the completed RHS performs no reductions.
- Match each RHS read of `X`, `Y`, or `Z` with the same operand's read on the LHS, then reuse its coordinates. If there are several terms, first match terms containing the same operands. If a read has multiple possible coordinate assignments and they cannot be shown to agree, synthesis remains unresolved.
- The result is a correspondence between operand reads in the two graphs. It supplies the coordinates needed to construct the RHS, while reduction domains, coefficients, and full value equivalence remain to be checked after its attributes are known.

#### 5.3.2 Determine Which Axes Each Operation Preserves

- After matching operand reads, we know which logical axes occur in each RHS input. However, the replacement combines these inputs through new operations. Which axes should each operation reduce, and which should it preserve in its output? We must assign the known axes to the groups declared by each replacement operation.
- Consider dot reassociation, `(XY)Z` → `X(YZ)`. On the LHS, the first dot contracts \(K\), and the second contracts \(L\). Section 5.2.2 splits \(L\) into \(I_{\mathrm{sel}}\) and \(J_{\mathrm{sel}}\), while the final output retains \(B,I_{\mathrm{rem}},J_{\mathrm{rem}},F\).
- The source mappings also tell us which groups occur in the inputs of the new intermediate `U = dot(Y, Z)`: `Y` contains \(B,K,J_{\mathrm{sel}},J_{\mathrm{rem}}\), and `Z` contains \(B,I_{\mathrm{sel}},J_{\mathrm{sel}},F\). These lists identify which groups occur in each operand; they do not specify physical axis order. Which groups should become `U`'s batch, contraction, left-free, and right-free axes? Their roles can change: for example, \(K\) was contracted by the original inner dot, but `U` must keep it so the final dot can pair it with `X`.


**General Rule for Preserving Axes**

- Which axes must an intermediate tensor keep for later computations? Keep an axis used to compute the intermediate if it is also needed outside that computation or in the final output.
- Let \(t\) be the intermediate RHS tensor, such as `U = dot(Y, Z)`. The computation producing \(t\), including its input reads, is its replacement subtree. For `U`, reads of `Y` and `Z` are inside this computation, and the read of `X` is outside it.

  | Symbol | Meaning |
  | --- | --- |
  | \(\operatorname{Axes}(x)\) | The set of logical axes in tensor \(x\)'s interface, ignoring their order. |
  | \(\operatorname{Axes}_{\mathrm{in}}(t)\) | Axes used by operand reads inside the computation producing \(t\), in the matched product. |
  | \(\operatorname{Axes}_{\mathrm{out}}(t)\) | Axes used by operand reads outside that computation, in the same matched product. |
  | \(O\) | The original final output tensor, whose axes the replacement must preserve. |
  | \(\operatorname{KeepAxes}(t)\) | The set of axes that must remain in \(t\)'s output. |

- Using these symbols, the problem is:

  \[
  \begin{aligned}
  \text{Given:}\quad &t,\quad
    \operatorname{Axes}_{\mathrm{in}}(t),\quad
    \operatorname{Axes}_{\mathrm{out}}(t),\quad
    \operatorname{Axes}(O).\\
  \text{Find:}\quad &\operatorname{KeepAxes}(t).
  \end{aligned}
  \]

- First collect the axes needed outside \(t\) or in the final output. Then keep those that also occur inside the computation producing \(t\):

  \[
  \operatorname{KeepAxes}(t)=\operatorname{Axes}_{\mathrm{in}}(t)
      \cap\bigl(\operatorname{Axes}_{\mathrm{out}}(t)\cup\operatorname{Axes}(O)\bigr).
  \]

  Here, \(\cup\) combines the axes from either set, and \(\cap\) keeps only axes present in both sets. These are sets of logical axes; their order is determined when constructing the operation's bindings and output interface. For multiple matched products, the required sets must agree.

- The equation identifies which coordinates must remain available through \(t\). Summing over an axis removes that coordinate from the intermediate. If an outside read uses the same axis, it still needs to pair its values with \(t\)'s values at matching coordinates. If the axis belongs to the final output, its coordinate must remain available to distinguish output elements. Either reason requires preserving the axis, which gives the union \(\operatorname{Axes}_{\mathrm{out}}(t)\cup\operatorname{Axes}(O)\).
- However, \(t\) only needs to carry those required axes that its own reads use. An axis used only outside this computation does not need to pass through \(t\). Intersecting with \(\operatorname{Axes}_{\mathrm{in}}(t)\) selects exactly the axes that occur inside and are still needed later. Axes used only inside can be reduced there, provided the operation's declared semantics permit that reduction.

- For \(t=U\), the inside reads are `Y` and `Z`, and the outside read is `X`. Thus, \(\operatorname{Axes}_{\mathrm{in}}(U)=\operatorname{Axes}(Y)\cup\operatorname{Axes}(Z)\) and \(\operatorname{Axes}_{\mathrm{out}}(U)=\operatorname{Axes}(X)\). Using the groups from Section 5.2.2:

  | Group present in `Y` or `Z` | Needed outside `U`? | Action in `U` |
  | --- | --- | --- |
  | \(B\) | Used by `X` and present in `O` | Preserve as batch axes |
  | \(K\) | Used by `X` | Preserve for the final contraction |
  | \(I_{\mathrm{sel}}\) | Used by `X`; its partners occur in `Z` | Preserve for the final contraction |
  | \(J_{\mathrm{sel}}\) | Neither used by `X` nor present in `O` | Contract between `Y` and `Z` |
  | \(J_{\mathrm{rem}}\), \(F\) | Present in `O` | Preserve as free axes |

- Explicitly, the three sets are as follows. Here, \(\{B,K,\ldots\}\) is shorthand for all axes belonging to the listed groups, ignoring their order:

  \[
  \begin{aligned}
  \operatorname{Axes}_{\mathrm{in}}(U)
    &=\{B,K,I_{\mathrm{sel}},J_{\mathrm{sel}},J_{\mathrm{rem}},F\},\\
  \operatorname{Axes}_{\mathrm{out}}(U)
    &=\{B,K,I_{\mathrm{sel}},I_{\mathrm{rem}}\},\\
  \operatorname{Axes}(O)
    &=\{B,I_{\mathrm{rem}},J_{\mathrm{rem}},F\}.
  \end{aligned}
  \]

  Applying the preservation rule gives:

  \[
  \operatorname{KeepAxes}(U)
    =\{B,K,I_{\mathrm{sel}},J_{\mathrm{rem}},F\}.
  \]

- For example, \(K\) belongs to both the inside and outside sets, so `U` preserves it. \(J_{\mathrm{sel}}\) belongs only to the inside set, so it can be contracted in `U`. \(I_{\mathrm{rem}}\) is needed by the final output but does not occur inside `U`; it reaches the final dot through `X` instead.

- Thus `U` must retain the axes in \(B,K,I_{\mathrm{sel}},J_{\mathrm{rem}},F\). Only \(J_{\mathrm{sel}}\) can be reduced there. At the final dot, \(K\) and \(I_{\mathrm{sel}}\) are shared by `X` and `U` but absent from `O`, so they are reduced together.

**From required axes to operation groups**

- We know which axes must remain in the result, but we still need to assign axes to the operation's declared groups. Which axes belong to each group? We determine this from where they occur in the operands and whether they must be preserved.
- Consider an operation producing \(t\) from operands \(x_1,\ldots,x_m\). Its definition specifies which operands each group binds and whether the group is reduced. For the groups bound to operands, the problem is:

  \[
  \begin{aligned}
  \text{Given:}\quad &\operatorname{Axes}(x_1),\ldots,
    \operatorname{Axes}(x_m),\quad\operatorname{KeepAxes}(t).\\
  \text{Find:}\quad &\operatorname{Axes}_g
    \quad\text{for each declared group }g\text{ bound to operands}.
  \end{aligned}
  \]

  Here, \(m\) is the number of operands, and \(\operatorname{Axes}_g\) is the set of logical axes assigned to group \(g\). This assigns axes to groups; their ordered positions are determined afterward.

- This reasoning follows the operation definition. A dot's batch and contraction groups occur in both operands; its free groups occur in only one. The required output set distinguishes the shared axes that survive from those that are reduced.
- More generally, let \(\operatorname{Axes}_i\) be operand \(i\)'s axis set, and let \(E_g\) be the nonempty set of operands bound to a declared group \(g\). First find axes present in exactly those operands:

  \[
  D_g=\left(\bigcap_{i\in E_g}\operatorname{Axes}_i\right)
      \setminus\left(\bigcup_{i\notin E_g}\operatorname{Axes}_i\right).
  \]

  Then let \(\operatorname{Axes}_g\) be the axes assigned to group \(g\), selected according to its declared role:

  \[
  \operatorname{Axes}_g=
  \begin{cases}
  D_g\setminus \operatorname{KeepAxes}(t),&\text{if the value definition reduces }g,\\
  D_g\cap \operatorname{KeepAxes}(t),&\text{otherwise.}
  \end{cases}
  \]

- For a dot with operands `a` and `b`, these rules give batch axes \((\operatorname{Axes}_a\cap \operatorname{Axes}_b)\cap \operatorname{KeepAxes}(t)\) and contraction axes \((\operatorname{Axes}_a\cap \operatorname{Axes}_b)\setminus \operatorname{KeepAxes}(t)\). The two free groups come from \((\operatorname{Axes}_a\setminus \operatorname{Axes}_b)\cap \operatorname{KeepAxes}(t)\) and \((\operatorname{Axes}_b\setminus \operatorname{Axes}_a)\cap \operatorname{KeepAxes}(t)\).
- The result for reassociation is the following group assignment. Entries show which axes are assigned to each group; free axes still follow their operand's physical order.

  | Replacement operation | Batch | Contracted | Left free | Right free |
  | --- | --- | --- | --- | --- |
  | `U = dot(Y, Z)` | \(B\) | \(J_{\mathrm{sel}}\) | \(K,J_{\mathrm{rem}}\) | \(I_{\mathrm{sel}},F\) |
  | `O' = dot(X, U)` | \(B\) | \(K,I_{\mathrm{sel}}\) | \(I_{\mathrm{rem}}\) | \(J_{\mathrm{rem}},F\) |

#### 5.3.3 Construct Attributes from the Mapping Table

**Attribute construction figure**

- [Storyboard](figures/attribute_construction.yaml): half-width portrait layout with aligned paired queries, a central P_U interface bridge, and a final symbolic attribute table. G abbreviates the ordered group I_sel ++ K; the preserved-group chips show which groups are preserved, without specifying axis order.
- **Caption:** Both operands query the same ordered contraction group. The inner replacement's attributes determine P_U, whose extended mappings then support the final dot's query. The table summarizes both descriptors; resolving the queries into executable attribute programs and verifying the completed rewrite follow afterward.

- The groups above identify which axes each dot uses. Its attributes must instead specify ordered positions in its actual operands. How do we turn a group assignment into those position sequences?
- Write \(\operatorname{positions}(P,G)\) for a mapping-table query that returns \(p\) satisfying:

  \[
  \operatorname{gather}(P,p)=G.
  \]

  The query uses existing entries and their compositions from Section 5.2.3. Both sides of a paired binding query the same ordered group, so corresponding list entries identify paired axes.

**1. Construct the inner replacement dot**

- For `U = dot(Y, Z)`, the source table already locates \(B\) and \(J_{\mathrm{sel}}\) in both operands. Therefore its attributes are:

  \[
  \begin{aligned}
  b_L^U&=b_Y,& b_R^U&=b_Z,\\
  c_L^U&=\operatorname{gather}(j_Y,v_J),& c_R^U&=z_J.
  \end{aligned}
  \]

- Here, \(z_J=\operatorname{gather}(c_R^{\mathrm{outer}},u_J)\) is the paired mapping derived in Section 5.2.2. The two contraction lists select \(J_{\mathrm{sel}}\) in the same order.
- Once these attributes are known, compute the remaining positions in each operand:

  \[
  \begin{aligned}
  f_Y^U&=\operatorname{ordered\_complement}
    \bigl(\rho(P_Y),b_Y\mathbin{+\!\!+}c_L^U\bigr),\\
  f_Z^U&=\operatorname{ordered\_complement}
    \bigl(\rho(P_Z),b_Z\mathbin{+\!\!+}c_R^U\bigr).
  \end{aligned}
  \]

  The declared dot output order then gives:

  \[
  P_U=B
      \mathbin{+\!\!+}\operatorname{gather}(P_Y,f_Y^U)
      \mathbin{+\!\!+}\operatorname{gather}(P_Z,f_Z^U).
  \]

- Extend the mapping table to \(P_U\) using this output layout. This step matters because \(K\) and \(J_{\mathrm{rem}}\) may be interleaved in `Y`, as may \(I_{\mathrm{sel}}\) and \(F\) in `Z`. The interface preserves those orders rather than assuming that each surviving group occupies one contiguous block.

**2. Construct the final replacement dot**

- The final dot contracts \(K\) and \(I_{\mathrm{sel}}\). Filtering the source reduction sequence \(L\mathbin{+\!\!+}K\) to these axes gives \(I_{\mathrm{sel}}\mathbin{+\!\!+}K\): the selected axes from \(I\) occur in \(L\), followed by \(K\). Use this order for both contraction lists. The source table locates it in `X`, and the extended table locates it in `U`:

  \[
  \begin{aligned}
  c_L^{O'}&=\operatorname{gather}(i_X,v_I)\mathbin{+\!\!+}k_X,\\
  c_R^{O'}&=\operatorname{positions}(P_U,I_{\mathrm{sel}})
       \mathbin{+\!\!+}\operatorname{positions}(P_U,K).
  \end{aligned}
  \]

- If a binding already has a fixed position list, that list determines the group order, and inferred paired lists must follow it. Otherwise, the implementation prefers the source reduction order for reduced groups and the source output order for retained groups, adding remaining axes in operand order.
- The resulting attribute table completes both replacement dots:

  | Attribute | `U = dot(Y, Z)` | `O' = dot(X, U)` |
  | --- | --- | --- |
  | Left batch positions | \(b_Y\) | \(b_X\) |
  | Right batch positions | \(b_Z\) | \(\rho(B)\) |
  | Left contraction positions | \(\operatorname{gather}(j_Y,v_J)\) | \(\operatorname{gather}(i_X,v_I)\mathbin{+\!\!+}k_X\) |
  | Right contraction positions | \(z_J\) | \(\operatorname{positions}(P_U,I_{\mathrm{sel}})\mathbin{+\!\!+}\operatorname{positions}(P_U,K)\) |

- Removing the final contraction axes leaves:

  \[
  P_{O'}=B\mathbin{+\!\!+}I_{\mathrm{rem}}
          \mathbin{+\!\!+}J_{\mathrm{rem}}\mathbin{+\!\!+}F=P_O.
  \]

  The replacement therefore has candidate attributes with the required pairings and output order. Their legality and value equivalence are checked next.

#### 5.3.4 Produce and Verify Attribute Programs

- The table still uses symbolic group queries. A rewrite must compute concrete attributes when it matches a graph. How can these queries become executable programs, and how do we know the completed replacement is correct?
- Resolve each query to expressions over source attributes and shapes. For example, the inner replacement's right contraction list is:

  \[
  c_R^U=\operatorname{gather}
    \left(c_R^{\mathrm{outer}},
      \operatorname{matchPositions}
        \left(c_L^{\mathrm{outer}},\lvert B\rvert+\lvert I\rvert+\rho(J)\right)_1
    \right),
  \]

  where the subscript \(1\) selects the first returned sequence, \(u_J\). Group lengths and position ranges are themselves derived from source metadata. This expression computes the required positions for each match without reading tensor elements.
- Queries involving \(P_U\) are resolved in the same way: substitute its constructed layout and compose its mappings until the expression depends only on source metadata. The emitted constructors contain no unresolved RHS attributes or logical-group queries.
- For operations with calculated accesses, position mappings alone may not determine every attribute. The implementation also matches supported index expressions, such as coordinate coefficients and offsets. For example, composing slices gives:

  \[
  \mathrm{start}'=\mathrm{start}_{\mathrm{inner}}
    +\mathrm{stride}_{\mathrm{inner}}\,\mathrm{start}_{\mathrm{outer}},
  \qquad
  \mathrm{stride}'=\mathrm{stride}_{\mathrm{inner}}\,\mathrm{stride}_{\mathrm{outer}}.
  \]

  Such constructors must also satisfy the resulting access bounds. Unsupported coupled or inverse equations remain unresolved.

**Check the completed replacement**

- With attributes filled, the RHS now has a complete value expression. In the reassociation example, `U` reduces \(J_{\mathrm{sel}}\), and the final dot reduces \(K\) and \(I_{\mathrm{sel}}\):

  \[
  e_{\mathrm{RHS}}
  =\sum_{I_{\mathrm{sel}},K}X\left(\sum_{J_{\mathrm{sel}}}YZ\right)
  =\sum_{I_{\mathrm{sel}},K,J_{\mathrm{sel}}}XYZ.
  \]

- The selected groups partition \(L\). Their mappings preserve each coordinate's operand pairing, so the normalized RHS can be compared with \(\sum_{L,K}XYZ\) under the permitted reduction laws. Verification also checks output axis order, shape, dtype, and operation legality.
- After construction, verification repeats these checks in a fresh context using the completed attribute programs. It must establish the required conditions from legal source metadata and the rule's declared conditions; construction obligations do not become extra assumptions.
- The result is a verified collection of source-dependent attribute programs. Synthesis constructs these programs when compiling the rule; applying the rule evaluates them on the matched graph's metadata. If construction or verification cannot establish the required relationships, the rule remains unresolved.

#### 5.3.5 General Synthesis Algorithm

- The example constructs two dots, but the same procedure uses each operation's declared groups, value expression, and output order. How can we state the procedure independently of this example?
- **Input:** A rule with a source graph, a replacement graph containing attribute holes, operation definitions, and declared conditions.
- **Output:** Verified attribute programs, or an unresolved result.
- Retain the state \((V,M,\Pi)\) from Section 5.2.3 and add a table \(C\) of proposed attribute expressions. The pseudocode names the two components of \(V[t]=(P_t,e_t)\) as `.axes` and `.value`. Read correspondence supplies the required output sets \(\operatorname{KeepAxes}(t)\). Process each replacement operation after its inputs, extending the state with its completed meaning and mappings.
- **Construction invariant:** Every completed replacement node has resolved attributes, satisfies its operation's legality conditions, and has an interface and value expression consistent with those attributes. Its mappings satisfy the invariant from Section 5.2.3. Equivalence of the whole rewrite is checked once the replacement is complete.

**Algorithm 2: Synthesize attribute programs**

```python
def synthesize_attribute_programs(rule):
    V, M, partitions = build_symbolic_axis_mapping(rule.lhs)
    source = normalize(V[rule.lhs].value)
    skeleton = normalize(arithmetic_skeleton(rule.rhs))
    match = correspond(source, skeleton)
    required = required_output_axes(match, V[rule.lhs].axes)
    bind_replacement_reads(V, match)
    C = {}

    for op in producer_first(rule.rhs):
        if op.is_input:
            continue
        inputs = [V[t] for t in op.inputs]

        if op.has_attribute_holes:
            groups = select_groups(op.definition, inputs, required[op])
            groups = order_groups(groups, op, source, inputs)

            for binding in op.unresolved_axis_bindings:
                P = inputs[binding.operand].axes
                G = groups[binding.group]
                C[binding.attribute] = positions(P, G, M, partitions)

            complete_other_attributes(op, inputs, groups, match, C)

        completed = substitute_attributes(op, C)
        V[op] = lower_and_check(completed, inputs, M, partitions)

    check_equivalence(V[rule.lhs], V[rule.rhs])
    programs = source_only_expressions(C)
    verify_in_fresh_context(rule, programs)
    return programs
```

- `correspond` matches normalized operand reads. `required_output_axes` computes the sets \(\operatorname{KeepAxes}(t)\), identifying which axes each RHS tensor must preserve for computations outside it or for the final result.
- `select_groups` applies the rules from Section 5.3.2 based on which operands contain each axis and whether it must be reduced. Groups with an explicit output domain are handled separately: their axes come from the required output axes absent from all operands, and their sizes determine the domain attributes. Ambiguous group assignments remain unresolved.
- `order_groups` respects fixed bindings and otherwise uses source and operand orders. `positions` retrieves or derives mappings for these ordered groups; the notation stands for the implementation's mapping lookup and position-construction steps.
- `complete_other_attributes` handles supported index coefficients, offsets, output domains, and output permutations. `lower_and_check` checks the completed operation and publishes its interface and mappings for consumers.
- `check_equivalence` compares the fully lowered values, reduction domains, output order, shape, and dtype. `source_only_expressions` resolves the proposed constructors to source metadata. The final verification reconstructs the completed rule in a fresh context, as described in Section 5.4.
- The pseudocode omits failure handling: unsupported or unproved steps return an unresolved result. On success, its output is a program for each missing attribute, ready to evaluate when the rule matches.

### 5.4 Correctness and Scope

- Attribute synthesis proposes how to construct the replacement. Verification must establish that this construction preserves the original computation. This section explains the checks, the intended guarantee, and the limits of the method.

#### 5.4.1 Checking the Completed Rewrite

- A mapping table can locate the requested axes, but that alone does not establish correctness. Are the resulting attributes valid, and does the completed RHS compute the same tensor as the LHS?
- After synthesis produces an attribute program \(f\), the verifier substitutes its expressions into the RHS and checks the completed rule in a fresh context. It reconstructs the tensor meanings and coordinate relationships instead of treating synthesis's intermediate conclusions as established facts.
- The verification checks are:

  | Check | What must be established |
  | --- | --- |
  | Attribute computation | Each constructor depends only on source information \(s\) and can be evaluated whenever the rule applies. No attribute holes remain. |
  | RHS legality | Constructed attributes satisfy the operation definitions, including valid position lists, paired axis sizes, reduction domains, and access bounds. |
  | Value equivalence | Normalized LHS and completed RHS terms agree in coefficients, indexed operand reads, and reduction domains. |
  | Output agreement | Both sides produce the same ordered output axes and shape. |

- Read matching against the RHS arithmetic skeleton helps establish coordinates, but does not prove value equivalence. The value check uses the fully lowered RHS, with its constructed attributes and actual reduction axes.
- The verifier may use source legality and the rule's declared guard. Any additional condition required by construction must follow from those assumptions; it cannot be added as a new assumption merely to make verification succeed.
- The result is either an accepted completed rule or an explanation of what could not be established. Explicit attribute programs are also subject to semantic verification.

#### 5.4.2 Correctness Guarantee

- What should successful verification guarantee beyond the checks above? It should establish the contract from Section 4.2 for every source instance covered by the rule, rather than only for particular tensor shapes or values.

**Formal Statement**

- The intended soundness guarantee is: if semantic verification accepts the completed rule, then for every legal source description \(s\) satisfying \(\operatorname{Guard}(s)\):

  1. The attribute program \(f(s)\) can be evaluated.
  2. Its attributes make the RHS operations legal.
  3. For every compatible input value \(X\),

     \[
     \operatorname{LHS}(s,X)
     =\operatorname{RHS}(s,f(s),X).
     \]

- Equality includes ordered output shape and element values, with element types fixed as in Section 4. The guarantee uses the selected arithmetic policy. For example, algebraic reassociation does not by itself establish equality under strict floating-point evaluation.
- **Proof structure:** Establish that axis mappings preserve ordered coordinates and that lowering preserves the declared operation meanings. Then prove normalization by induction on value expressions: each case preserves values under its reduction-domain conditions. Finally, show that matched normalized terms have equal coefficients, coordinate accesses, and summation domains, so their sums are equal. Together with attribute evaluability, RHS legality, and output agreement, these results imply the contract above.
- The [soundness proof draft](soundness_proof.md) states these lemmas and proves a relative soundness theorem for the algebraic verifier. Its assumptions explicitly cover correct operation definitions, sound primitive metadata and index checks, admissible coordinate frames, and faithful evaluation of partial attribute programs. These assumptions must hold for the theorem to apply to an implementation.
- The fresh verification context separates checking from construction, but both use shared semantic infrastructure. The proof draft therefore distinguishes the mathematical verifier from the remaining implementation obligations, including checked arithmetic, coordinate transport, and guard evaluation. It does not claim that the current C++ implementation has been formally proved sound.

#### 5.4.3 Scope and Limitations

- What does failure mean? A rule that cannot be proved may still be correct. The method is incomplete: it accepts a semantic rewrite only when the supported reasoning establishes all required conditions.
- The main limits are:

  | Limitation | Example |
  | --- | --- |
  | Ambiguous correspondence | A replacement read has several possible source coordinate assignments that cannot be shown to agree. |
  | Unsupported attribute construction | A missing attribute requires solving a coupled or inverse equation outside the supported fragment. |
  | Unsupported semantic reasoning | An index computation or guard can execute, but the checker cannot establish the required relationship. |
  | Arithmetic policy | A proposed reassociation requires laws unavailable under the selected policy. |
  | Bounded proof work | Normalization or coordinate reasoning reaches an implementation limit. |

- Failure reports should distinguish malformed rules, unresolved or unsupported reasoning, and exhausted proof limits. The method does not claim to find every valid attribute program or the weakest condition under which a rewrite is correct.
- This guarantee applies to semantically verified rules. Structural rules receive structural checks and do not carry the same tensor-equivalence guarantee.
- Thus, acceptance means the supported semantic checks succeeded under the declared conditions; failure to prove a rule is not, by itself, a counterexample to the rule.

## 6. Implementation

### 6.1 High-Level Architecture

**Figure 3 — TEPL architecture**

- [Storyboard](figures/tepl_architecture.yaml): half-width overview led by three equally prominent input cards, Rule, Dialect, and Graph. Rule and Dialect feed the compiler; Graph bypasses compilation and feeds the TEPL engine directly. Compiled JSON contains patterns, guards, and attribute programs, while a dashed adapter branch shows the extension point for additional engines.
- **Caption:** TEPL takes three inputs: a rule specifying the rewrite, a dialect defining operations and their semantics, and a graph to rewrite. The compiler lowers and normalizes the semantics, synthesizes missing attributes, verifies completed rules, and exports reusable rewrite programs. At runtime, the TEPL engine matches patterns, checks guards, evaluates attribute programs, and applies replacements to the input graph. E-graph engines integrate through generated code for Rust/egg and C++/egg-c; the dashed branch denotes engine integration.

- TEPL is implemented in **C++20** and uses **ANTLR** to parse operation definitions and rewrite rules. The system separates rule compilation from rewrite execution.
- During compilation, the compiler resolves names and checks types, then constructs missing attributes and verifies semantic rules. It lowers the completed rules into an executable representation that can be exported as JSON.

  ```text
  Rule compilation
  TEPL source
    → ANTLR parsing
    → Name resolution and type checking
    → Attribute synthesis and semantic verification
    → Executable rewrite program
    → JSON export
  ```

- Type-checked rules may still contain attribute holes. Executable rules contain completed attribute programs, guards, and replacement patterns. JSON serializes this executable representation so it can be loaded without repeating compilation or semantic verification.
- The rewrite engine loads and validates the compiled program, then searches an input graph for matching LHS patterns. For each match, it evaluates the rule's guard, computes the replacement attributes, and constructs and applies the RHS graph.

  ```text
  Rewrite execution
  Compiled JSON + input graph
    → Load and validate
    → Find an LHS pattern match
    → Evaluate the guard
    → Compute replacement attributes
    → Construct and apply the RHS
  ```

- A failed guard or invalid attribute evaluation skips the match. The engine applies rewrites greedily, repeating the search until no rule changes the graph or an execution limit is reached.

### 6.2 Internal Representations

- Symbolic axis groups and position expressions are represented by immutable, typed expression nodes. Their types distinguish axes, positions, extents, integers, and Boolean conditions. Operations such as `gather`, concatenation, and inverse permutation remain symbolic until their inputs are known.
- Expression nodes can share subexpressions. Structural keys identify expressions with the same structure, and each proof context interns these keys to accelerate repeated comparisons. Hash values accelerate lookup; structural comparison resolves collisions.
- Value expressions use nodes for constants, operand reads, binary addition, binary multiplication, and reduction. Calculated operand coordinates use a separate index-expression representation supporting coordinate selection, concatenation, scaling, and offsets.
- Normalization produces a vector of terms. Each term stores an integer coefficient, a list of operand reads, and a symbolic reduction-axis sequence, implementing the representation introduced in Section 5.3.1.

### 6.3 Mapping Storage and Lookup

- The mapping table stores entries `(interface, group, positions)`. Separate indexes support lookup by interface and group, and reverse lookup by interface and position sequence.
- Partitions refer to mapping entries through stable identifiers. Each partition retains its own position sequences when they differ from the mapping table's preferred sequence. Overlap records retain the selected subgroup and its positions in both parent groups.
- Lookup first checks existing mappings. When a direct mapping is unavailable, it follows established relationships and composes their position sequences. Concatenated groups are handled by combining component mappings; traversal through registered origins uses breadth-first search and avoids revisiting the same origin.
- Lookup results are cached. Registering new mappings invalidates the mapping cache, while changes to established facts invalidate the corresponding reasoning caches. This allows later queries to use newly discovered relationships.

### 6.4 Rule Execution

- The engine searches reachable graph nodes for LHS pattern matches. Matching checks operation identities and operand structure, then records captured graph nodes, operation attributes, and shape bindings. Repeated captures must refer to the same node.
- For each match, the evaluator processes the guard and then the replacement-attribute expressions using those bindings. Boolean conditions preserve short-circuit evaluation. A false guard or evaluation failure skips the match.
- The builder reuses captured nodes and constructs new RHS operations with the computed attributes. Construction takes place on a candidate graph, allowing the engine to validate operation contracts and output compatibility before committing the replacement.
- After a successful replacement, the engine restarts its search. Rewriting stops when no candidate changes the graph, a configured limit is reached, or execution is cancelled.

## 7. Limitations and Future Work

**Coordinate semantics for automatic inference.** TEPL supports coordinate
transformations such as static slice and reverse, but its current semantic
language cannot describe every tensor access pattern. Dynamic slice requires
tensor-valued indices; pad and concatenate require coordinate-dependent
branches; and general reshape requires coordinate flattening and unflattening
using division and modulo. Without these operation semantics, TEPL cannot
automatically infer attributes or semantically verify rewrites involving them.
Extending the coordinate language and its associated reasoning is future work.

These restrictions concern the semantic verification and inference machinery.
Users can still express structural rewrites for these operations, specify
applicability conditions, and construct RHS attributes explicitly using TEPL's
attribute expressions. Manual rewrite authoring therefore remains useful
beyond the automatic inference fragment: TEPL can match patterns, evaluate
guards and attribute programs, and build replacement graphs. Such structural
rewrites do not carry the tensor-equivalence guarantee of semantically verified
rules.

**No completeness guarantee.** TEPL does not guarantee that it can verify every
valid rewrite or discover every valid attribute construction. A correct rule
may remain unproved because coordinate correspondence is ambiguous or a
required metadata relationship lies outside the supported reasoning. Likewise,
verifying a manually supplied constructor does not guarantee that synthesis
can discover it. The correctness argument concerns accepted rules; a proof
of completeness is outside this paper's scope. Future work can expand the
supported reasoning and establish completeness for a precisely defined
fragment.

**Reasoning cost and scalability.** Long rewrite compositions and overlapping
axis relationships can increase the number of symbolic partitions and the
size of normalized expressions. Verification and inference use finite proof
budgets, so a rule may remain unresolved when the required reasoning exceeds
those budgets. The XLA evaluation measures coverage. The dedicated
[scaling experiment](../labs/attribute_inferences/reasoning_scalability/README.md) measures
rank-polymorphic dot-chain and additive-term compilation, plus execution of
compiled metadata programs as rank increases. These controlled families do not
establish scalability across arbitrary compositions or axis layouts. Future
work includes more efficient symbolic representations and proof search.

## 8. XLA Evaluation

We evaluate TEPL on 209 concrete rewrite cases adapted from XLA's algebraic
simplifier. The evaluation measures operation-semantic coverage, verification
with explicit RHS attributes, and automatic attribute inference. Counts refer
to concrete cases, which may specialize broader catalog rules.

| Stage | Result |
|---|---:|
| Semantic coverage | 118/209 |
| Explicit verification | 86/118 |
| Auto inference (axis/extent) | 20/20 (100%) |

**Semantic coverage.** A case is eligible for verification when every required
operation has a semantic definition in TEPL. This holds for 118 of the 209
cases; the remaining 91 lack at least one required definition. TEPL combines
axis bindings, scalar expressions, reductions, and calculated operand reads
to describe both value computations and supported coordinate transformations.
For example, transpose permutes axes, dot sums products over contracting axes,
and static slice computes input coordinates from starts and strides.

Supported operations include elementwise arithmetic, transpose, axis-inserting
broadcast, sum reduction, dot, reverse, and static slice with positive sizes
and strides. Convolution support uses fixed layouts, positive strides and
dilations, and no padding or groups. General reshape is unsupported because
it requires coordinate flattening and unflattening using division and modulo.
Pad and concatenate require coordinate-dependent branches, while dynamic slice
requires tensor-valued indices. These capabilities are outside the current
semantic support.

**Explicit verification.** We first supply RHS attributes through captured
descriptors or handwritten attribute constructions, without using `auto`.
TEPL checks that these constructions are evaluable, the RHS operations are
legal, and both sides agree in output shape and values under the declared
guards and arithmetic policy. The recorded full-corpus sweep verifies 86 of
the 118 eligible cases.

The other 32 cases remain unproved. This does not establish that their rewrites
are incorrect: TEPL may lack the required shape, axis, index, or scalar
reasoning, or the rule may need an additional applicability condition.
Successful verification is relative to the declared operation semantics and
arithmetic policy, rather than a claim about every aspect of XLA's numerical
behavior.

**Automatic inference.** Of the 86 verified cases, 63 have no RHS attributes
to infer. Another three require scalar settings, such as comparison direction.
The remaining 20 require only axis or extent lists, including permutations,
reduction dimensions, broadcast dimensions and extents, and slice parameters.

For these 20 cases, we replace the RHS attribute specifications with `auto`
and remove the corresponding handwritten constructions while retaining the
same patterns, guards, and operation semantics. TEPL constructs all required
attributes and independently verifies each completed rule. All 20 succeed
without RHS attribute hints, including both descriptor reuse and computed
attribute programs.

The 100% result measures inference conditional on explicit verification:
these 20 cases are the verified subset of 38 eligible axis/extent cases.
The verification count comes from the recorded full-corpus sweep; the
inference result comes from the current compiler recheck.

## 9. Related Work and Novelty Positioning

- **TensorRight:** Aggregated axes and arbitrary-rank rewrite verification are prior work; axis grouping alone is not a novelty claim. [POPL 2025](https://doi.org/10.1145/3704865)
- **Tensor rewrite synthesis and equality saturation:** Compare TASO/TENSAT on generated artifacts, semantic assumptions, and treatment of attributes; examine primary papers before asserting distinctions.
- **Semantics-driven synthesis:** Compare synthesis targets, parameterization, and correctness contracts with superoptimization and rewrite-synthesis systems.
- **Index-notation and affine compiler IRs:** Explain how TEPL reconstructs operation attributes from symbolic coordinate correspondence.
- **Rewrite DSLs and host callbacks:** Compare declarative authoring with manually implemented attribute transformations.
- Candidate distinction to substantiate: compositional synthesis of reusable, rank-polymorphic attribute programs from user-defined tensor semantics.
- Avoid “first” claims until a systematic literature review supports them.

## 10. Conclusion

- Reiterate the central result to establish: semantic coordinate correspondence can generate executable attribute transformations.
- Connect this result to the language: users write rewrite patterns and conditions, then supply attribute changes explicitly or ask the compiler to infer them.
- Summarize only guarantees and practical benefits supported by proofs and measurements.

## Appendix / Artifact

- Formal IR grammar, denotational semantics, mapping invariants, and algebraic side conditions.
- Full synthesis pseudocode and proof details.
- Benchmark definitions, guards, explicit and synthesized constructors, and failure classifications.
- Reproduction commands, versioned environments, and independent reference tests.
