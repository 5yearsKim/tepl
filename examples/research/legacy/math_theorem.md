# Research Proposal: Finite-Rank Verification of Tensor Rewrites with Automatic Attribute Inference

## 1. Problem Definition

Goal: Prove that verifying tensor rewrites up to a finite rank \\(K\\) is sufficient to guarantee that valid RHS attributes exist for arbitrary-rank tensors.

For example, in TEPL:

```
LHS: (outer: dot (inner: dot A B) C)
RHS: (dot A (dot B C))
```

The LHS attributes are known, while RHS attributes must be inferred automatically.

Let:

- \\(S\\): Tensor rewrite rule
- \\(p\\): Legal tensor configuration (ranks, shapes, LHS attributes)
- \\(\beta\\): Inferred RHS attributes
- \\(\operatorname{Valid}(p,\beta)\\): Both sides are mathematically equivalent for all tensor values, including shape and axis ordering

We want to derive a finite \\(K\\) satisfying:

\\[ \boxed{ \left( \forall p:\operatorname{rank}(p)\le K,\ \exists\beta:\operatorname{Valid}(p,\beta) \right) \implies \left( \forall p,\ \exists\beta:\operatorname{Valid}(p,\beta) \right) } \\]

For multiple inputs, the rank bound applies componentwise.

The challenge is to derive \\(K\\) mathematically from operation semantics, rather than selecting it experimentally.

## 2. Restricting TensorLang Semantics

Arbitrary user-defined tensor computations cannot generally guarantee a computable finite-rank bound.

We therefore define a restricted semantic language, tentatively called Axis-Regular TensorLang.

| Feature       | Supported                                          | Initially excluded             |
| ------------- | -------------------------------------------------- | ------------------------------ |
| Tensor rank   | Arbitrary                                          | —                              |
| Index access  | Select, preserve, omit coordinates                 | Computed indices such as `i+j` |
| Axis matching | Explicit logical correspondences                   | Unrestricted unknown pairings  |
| Axis ordering | Fixed group ordering, filtering, reversal          | Arbitrary permutations         |
| Reduction     | Finite sums over selected axes                     | Data-dependent reduction       |
| Computation   | Exact sums/products, independent single-use inputs | Arbitrary scalar programs      |
| Shapes        | Symbolic dimensions and equality                   | General shape arithmetic       |
| Guards        | Finite-state-expressible conditions                | Arbitrary predicates           |

The goal is to support useful operations such as:

- `dot_general`, including restricted batching and contracting
- `reduce_sum`
- Restricted `transpose`
- Elementwise multiplication

These operations must lower into a common semantic representation with exact, finitely describable axis relationships.

## 3. Mathematical Theorems to Derive

### Theorem 1 — Exact Axis Representation

Prove that every legal tensor configuration and RHS attribute assignment in the restricted language can be represented using finite words.

Each word position represents a logical axis, annotated with finite information such as:

```
Axis:       b       m       k       n
Role:      batch    free   contract  free
```

The representation must preserve all legal axis relationships, shapes, and output ordering.

### Theorem 2 — Semantic Equivalence

Prove that equivalence in the proposed axis representation is equivalent to actual tensor equality:

\\[ \boxed{ \operatorname{AxisEquivalent}(L,R) \iff \forall X,\ L(X)=R(X) } \\]

This is potentially the hardest theorem.

It requires showing that the abstraction preserves numerical computations, not merely tensor shapes.

For the initial exact sum/product fragment, one possible proof technique is comparing polynomial expansions and their index-identification relationships.

### Theorem 3 — Finite-State Attribute Existence

Prove that the set of legal configurations admitting some valid RHS attributes is recognizable by a finite automaton:

\\[ G_S(p)\iff\exists\beta:\operatorname{Valid}(p,\beta) \\]

A possible mathematical foundation is monadic second-order logic over finite words (MSO), which admits translation to finite automata.

The difficult obligation is proving that the restricted tensor semantics and attribute-existence relation can be expressed exactly in this logic.

### Theorem 4 — Finite Counterexample Bound

Construct an automaton recognizing configurations where no RHS attributes work:

\\[ \operatorname{Bad}\_S = \operatorname{Legal}\_S\setminus G_S \\]

If this automaton has \\(N\\) reachable states, then every nonempty failure language has a counterexample represented by a word of length at most \\(N-1\\).

Assuming each input rank is bounded by its encoded word length:

\\[ \boxed{K=\max(1,N-1)} \\]

Thus:

\\[ \boxed{ \text{All configurations up to }K\text{ valid} \implies \text{All configurations valid} } \\]

This theorem follows from standard automata theory once Theorems 1–3 establish an exact encoding.

## 4. Expected Verification Framework

TEPL Rewrite + Operation Semantics

User-defined operations and `auto` attributes

Restricted Semantic Analysis

Check supported fragment and extract axis constraints

Finite-State Verification

Construct validity/failure automata and derive sufficient K

No failures

Arbitrary-rank validity proven

Failures exist

Counterexample or applicability guard

If the exact automata are available, emptiness checking can decide universal validity directly; deriving \\(K\\) is an additional mathematical guarantee.

For accepted configurations, TEPL can recover or infer concrete RHS attributes at rewrite-matching time.

## 5. Expected Research Contributions

| Contribution                        | Expected result                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| Restricted tensor semantic language | Expressive, rank-independent representation supporting practical ML operations |
| Exact semantic translation          | Proof that finite-state constraints preserve tensor equivalence                |
| Automatic finite-rank bound         | Derive a sound \\(K\\) from rewrite semantics                                  |
| Attribute synthesis                 | Generate RHS attributes and exact applicability guards                         |
| Experimental evaluation             | Measure supported rewrites, verification cost, and bound sizes                 |

## 6. Important Limitations

The framework would not automatically support arbitrary TensorLang programs. Unsupported operations must return `unknown` rather than claiming soundness.

Moreover, deriving \\(K\\) does not guarantee it will be small enough for practical bounded verification. Automaton construction can also be computationally expensive.

Finally, some rules, including unrestricted `dot_general` reassociation, are mathematically invalid for particular attribute configurations. The framework should detect those failures or generate suitable guards—not incorrectly prove unconditional validity.

## Central Research Question

Can we define a sufficiently expressive subset of tensor semantics for which arbitrary-rank rewrite validity and RHS attribute existence are exactly reducible to finite-state verification, enabling automatic derivation of sound finite-rank bounds?

The main mathematical work is establishing Theorems 1–3. Once those hold, Theorem 4 follows from established automata theory.

That distinction is important: the potential research contribution is not the finite-automaton cutoff theorem itself, but the generic, sound translation from tensor semantics to a decidable finite-state representation.