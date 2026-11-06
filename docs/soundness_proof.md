# Soundness of Semantic Rewrite Verification

This draft gives a mathematical soundness proof for the algebraic verification procedure described in Sections 4 and 5 of [the paper outline](paper_outline.md). It proves the core normalization and comparison arguments and states the contracts required from operation definitions, coordinate handling, and metadata checking.

The main theorem is a **relative soundness theorem**: it assumes that the primitive metadata checker and the implementation of the specified lowering rules satisfy the contracts below. It does not establish that every branch of the current C++ implementation satisfies those contracts. The final section identifies that remaining implementation work.

## 1. Statement and scope

Let a rule consist of an LHS computation, an RHS computation, and a declared guard. Use the notation from Section 4:

- \(s\) is the source information: shapes and operation attributes.
- \(X\) assigns tensor values to the captured inputs, with the shapes specified by \(s\).
- \(f\) is a candidate program computing the RHS attributes from \(s\).
- \(\operatorname{Guard}(s)\Downarrow\mathrm{true}\) means guard evaluation succeeds and returns `true`.
- \(f(s)\Downarrow a\) means attribute evaluation succeeds and returns \(a\).

The theorem concerns the following contract:

\[
\begin{aligned}
&\operatorname{Accept}(\operatorname{LHS},\operatorname{RHS},f,\operatorname{Guard})\\
&\quad\Longrightarrow\quad
\forall s.\ \bigl(\operatorname{LegalLHS}(s)
\land\operatorname{Guard}(s)\Downarrow\mathrm{true}\bigr)\\
&\qquad\Longrightarrow\quad
\exists a.\ f(s)\Downarrow a
\land\operatorname{LegalRHS}(s,a)
\land\forall X.\ \operatorname{LHS}(s,X)=\operatorname{RHS}(s,a,X).
\end{aligned}
\]

Tensor equality means the same ordered shape and the same value at every output coordinate. Element types are fixed in this formulation.

For the algebraic fragment, scalar values belong to a commutative ring \(\mathbb K\). Thus addition and multiplication are associative and commutative, multiplication distributes over addition, zero annihilates multiplication, and integer coefficients have their usual ring interpretation. This covers exact algebraic semantics; it does not establish the same identities for strict floating-point execution. Metadata integers and their checked arithmetic are separate from scalar tensor values.

All tensor dimensions are finite and nonnegative. Their sizes may depend on \(s\), but not on tensor element values or on reduction coordinates. Zero-sized dimensions are allowed. The theorem quantifies over arbitrary finite ranks and sizes, without fixing a rank bound.

## 2. Meanings of expressions and normalized terms

Fix legal source information \(s\). Each logical axis \(u\) has a finite coordinate domain

\[
D_s(u)=\{0,\ldots,d_s(u)-1\}.
\]

For a sequence \(R\) of distinct axes, \(D_s(R)\) is the set of assignments to those axes. An empty axis sequence has one assignment: the empty assignment. If any axis in \(R\) has size zero, \(D_s(R)\) is empty.

An environment \(\eta\) supplies coordinates for the free axes. A read \(\operatorname{Read}(A,\alpha)\) accesses captured tensor \(A\) at the ordered coordinate tuple computed by index expression \(\alpha\). Write its value as \(X_A[\alpha(s,\eta)]\). Validity requires the tuple to have the correct rank and be within the tensor's bounds.

The scalar value-expression language is:

\[
e ::= c\mid\operatorname{Read}(A,\alpha)
\mid\operatorname{Add}(e,e)
\mid\operatorname{Product}(e,e)
\mid\operatorname{Reduce}(R,e).
\]

Constants, reads, addition, and multiplication have their ordinary meanings in \(\mathbb K\). Reduction means:

\[
\llbracket\operatorname{Reduce}(R,e)\rrbracket_{s,X,\eta}
=\sum_{\rho\in D_s(R)}
\llbracket e\rrbracket_{s,X,\eta[R\mapsto\rho]}.
\]

Bound coordinates are scoped; renaming them must avoid capture. The notation \(\eta[R\mapsto\rho]\) replaces the coordinates of \(R\) with \(\rho\).

A normalized term is \(\tau=(c,[r_1,\ldots,r_m],R)\), with meaning:

\[
\llbracket\tau\rrbracket_{s,X,\eta}
=\sum_{\rho\in D_s(R)}
c\prod_{j=1}^{m}
\llbracket r_j\rrbracket_{s,X,\eta[R\mapsto\rho]}.
\]

A normalized expression \(N=[\tau_1,\ldots,\tau_k]\) denotes the sum of its terms. An empty product equals \(1\); an empty term list equals \(0\). The read list preserves multiplicity: two occurrences of the same read contribute two factors.

Let \(\operatorname{Deps}(\tau)\) be all axes occurring in its read coordinates. Its free dependencies are \(\operatorname{Deps}(\tau)\setminus R\). Treat these as conservative dependency sets: absence of an axis guarantees independence from that coordinate; presence need not imply a genuine value dependence after cancellation.

## 3. Primitive contracts and valid premises

Let \(\Gamma\) contain source legality, the successfully evaluated declared guard, and facts derived from them. Write \(\Gamma\models\phi\) when every source instance and compatible coordinate assignment satisfying \(\Gamma\) satisfies \(\phi\).

The proof uses these contracts:

1. **Declared operation semantics.** Each operation definition gives its legality conditions, output interface and shape, and a value expression. For legal operands, these definitions describe the operation's mathematical meaning. Connecting this meaning to a backend implementation is a separate specification obligation.
2. **Primitive reasoning.** If the metadata or index checker establishes \(\Gamma\vdash\phi\), then \(\Gamma\models\phi\). This includes equalities, bounds, disjointness, selector validity, and equality of calculated index expressions. Equal axis sizes alone do not imply equal axis identities.
3. **Defined evaluation.** Proving that a constructor is well-defined establishes successful evaluation for every admissible \(s\). The metadata evaluator implements the specified primitive operations, including checked arithmetic and partial operations. Referencing only source fields is necessary, but does not by itself prove evaluability.
4. **Admissible coordinates.** Every captured input frame maps logical coordinates to its physical dimensions in the stated order, with matching sizes and distinct axes. Coordinate changes and index substitution preserve this interpretation and avoid capturing bound coordinates. The mapping rules proved below justify the corresponding sequence operations.

These are explicit interfaces to the rest of the verifier, not an assumption that the completed rewrite is equivalent. In particular, the proof below establishes normalization and normalized-term comparison without assuming their conclusions.

**Premise discipline.** Source legality may be assumed because the theorem is restricted to legal source instances. A completed RHS operation must establish its legality before publishing facts based on it. Each other fact added to \(\Gamma\) must follow from facts already justified. Candidate constructors and candidate guards do not justify their own correctness.

**Lemma 1 — Preservation of valid premises.** If initial premises hold for every admissible source instance, and each added fact follows from existing premises by a valid rule, every fact in the resulting context holds for every such instance.

**Proof.** Induct on the number of facts added. The base case is the definition of admissibility. For the induction step, the earlier facts hold by the induction hypothesis, and the new fact follows by validity of its derivation. A failed obligation adds no assumption and cannot lead to acceptance. This also explains why recording a failed obligation for diagnostics is harmless only when verification stops. ∎

## 4. Coordinate-mapping lemmas

All statements in this section are about evaluated finite sequences. Applying them to symbolic sequences means they hold for every admissible evaluation of \(s\).

**Lemma 2 — Gather composition.** For valid position sequences \(p\) and \(q\),

\[
\operatorname{gather}(\operatorname{gather}(P,p),q)
=\operatorname{gather}(P,\operatorname{gather}(p,q)).
\]

**Proof.** At position \(j\), both sides equal \(P[p[q[j]]]\). Selector validity makes each access defined. Both sides have length \(|q|\), so they are equal as ordered sequences. ∎

**Lemma 3 — Reconstructed layouts.** Let \(C=G_1\mathbin{+\!\!+}\cdots\mathbin{+\!\!+}G_m\) contain distinct axes. Let \(p=p_1\mathbin{+\!\!+}\cdots\mathbin{+\!\!+}p_m\) be a permutation of \(\{0,\ldots,|C|-1\}\), with \(|p_i|=|G_i|\). Define \(Q=\operatorname{gather}(C,\operatorname{inverse}(p))\). Then \(\operatorname{gather}(Q,p_i)=G_i\) for every \(i\).

**Proof.** By Lemma 2, \(\operatorname{gather}(Q,p)=C\), since selecting the inverse permutation at positions \(p\) gives the identity sequence. Restrict both sides to the consecutive positions occupied by \(p_i\) in \(p\). The corresponding segment of \(C\) is \(G_i\). ∎

**Lemma 4 — Partition refinement.** Suppose \(G=\operatorname{gather}(P,p)\), \(H=\operatorname{gather}(P,q)\), and \((u,v)=\operatorname{matchPositions}(p,q)\). Then

\[
\operatorname{gather}(G,u)=\operatorname{gather}(H,v).
\]

**Proof.** Each matched pair identifies the same position in \(P\): \(p[u[j]]=q[v[j]]\). Selecting \(P\) at those positions gives the same ordered axes. The definition of `matchPositions` includes every shared position and retains its paired locations. Lemma 2 transports the relationship through any earlier mappings. ∎

For a valid selector, its ordered complement contains exactly the omitted positions. The selected positions and complement are disjoint and cover the interface. They need not concatenate in physical axis order. Comparing two complete partitions can therefore introduce overlapping subgroups without changing the tensor or its coordinate domain.

**Mapping invariant.** Starting from valid operation bindings, every mapping generated by identity selection, composition, reconstructed layout, concatenation with valid offsets, or the refinement above satisfies \(\operatorname{gather}(P,p)=G\).

**Proof.** Induct on the mapping derivation. The nontrivial cases are Lemmas 2–4; concatenation and offsets follow by indexing each component. A consistent, size-preserving coordinate renaming carries both sides to the corresponding renamed equality. This establishes the invariant for these mathematical rules; any additional implementation shortcut must satisfy the same invariant. ∎

## 5. Lowering preserves meanings

**Lemma 5 — Compositional lowering.** Under the primitive contracts, lowering a legal operation graph produces an ordered output interface \(P\), shape \(S\), and expression \(e\) such that evaluation of \(e\) at each output coordinate equals evaluation of the graph at that coordinate.

**Proof.** Induct over the graph in producer-first order.

- For an input tensor, its admissible frame lists coordinates in physical dimension order. The generated read therefore accesses that same tensor element.
- Suppose the claim holds for an operation's inputs. Its definition specifies the result by combining those inputs using scalar operations, reductions, and possibly calculated reads. Replacing each operand by its lowered expression preserves its value by the induction hypothesis.
- For an ordinary operand read, the mapping invariant ensures that selecting or renaming coordinates uses the same operand positions. For a calculated read, capture-avoiding substitution evaluates the producer expression at exactly the requested index tuple. The index contract and access obligations establish that the tuple is defined and in bounds.
- Addition and multiplication preserve equality of their operands. Reduction preserves equality at each assignment of its bound coordinates, and therefore preserves the finite sum. Its domain is the domain declared by the operation.
- The declared output layout determines the ordered interface and shape. Established legality ensures the output positions and sizes are valid.

Thus the lowered expression and the operation graph agree at every output coordinate. The induction applies to the source using its assumed legality and to the completed replacement after its legality obligations have been established. ∎

This argument is relative to the declared operation semantics. Incorrect declarations cannot be repaired by a rewrite-equivalence proof.

## 6. Normalization preserves values

Define \(N(e)\) recursively, following the normalizer:

| Expression | Normalized result |
| --- | --- |
| Constant \(c\) | `[(c, [], [])]` |
| Read \(r\) | `[(1, [r], [])]` |
| \(\operatorname{Add}(e_1,e_2)\) | Concatenate \(N(e_1)\) and \(N(e_2)\). |
| \(\operatorname{Product}(e_1,e_2)\) | For each pair of terms, multiply coefficients and concatenate reads and reductions, subject to the conditions below. |
| \(\operatorname{Reduce}(R,e)\) | Prepend \(R\) to each term's reductions, after checking disjointness. |

For product terms \(\tau_1=(c_1,F_1,R_1)\) and \(\tau_2=(c_2,F_2,R_2)\), require:

\[
\begin{aligned}
R_1\cap R_2&=\varnothing,\\
R_1\cap(\operatorname{Deps}(\tau_2)\setminus R_2)&=\varnothing,\\
R_2\cap(\operatorname{Deps}(\tau_1)\setminus R_1)&=\varnothing.
\end{aligned}
\]

Here set operations ignore sequence order. These conditions ensure independent bound coordinates and prevent one term's reduction from binding a free coordinate used by the other term. The result is

\[
(c_1c_2,F_1\mathbin{+\!\!+}F_2,R_1\mathbin{+\!\!+}R_2).
\]

For the reduction case, require \(R\cap R_\tau=\varnothing\) for each normalized child term. A failed condition or coefficient-overflow check produces no successful normalization result.

**Lemma 6 — Normalization.** If normalization succeeds and its side conditions are established, then

\[
\llbracket e\rrbracket_{s,X,\eta}
=\llbracket N(e)\rrbracket_{s,X,\eta}.
\]

**Proof.** By structural induction on \(e\).

1. **Constant and read.** Each becomes a one-term expression with an empty reduction sequence. The empty coordinate assignment contributes once. For constants, the empty product is \(1\). The meanings therefore agree directly.
2. **Addition.** By induction, each operand equals the sum of its normalized terms. Concatenating the term lists adds those two sums, preserving the value.
3. **Multiplication.** By induction and distributivity, the product of two normalized expressions is the sum over all pairs of terms. For one pair, the side conditions give

   \[
   \left(\sum_{\rho_1\in D_s(R_1)}c_1\prod F_1\right)
   \left(\sum_{\rho_2\in D_s(R_2)}c_2\prod F_2\right)
   =\sum_{(\rho_1,\rho_2)\in D_s(R_1)\times D_s(R_2)}
      c_1c_2\left(\prod F_1\right)\left(\prod F_2\right).
   \]

   The free-dependency checks ensure that extending the coordinate assignment with \(R_2\) does not change the first factor, and vice versa. Disjointness identifies the product domain with \(D_s(R_1\mathbin{+\!\!+}R_2)\). Associativity and commutativity of multiplication give exactly the constructed term. Summing these equalities proves the case.
4. **Reduction.** By induction and distributivity of finite sums over addition, reducing a normalized expression equals the sum of its individually reduced terms. For each term, disjointness gives a bijection

   \[
   D_s(R)\times D_s(R_\tau)
   \cong D_s(R\mathbin{+\!\!+}R_\tau).
   \]

   The nested sums therefore equal the single sum in the new term. The outer reduction is allowed to bind free dependencies of the child: that is precisely the original reduction's meaning.

All arguments use finite sums. They also hold when a reduction domain is empty, because the corresponding sum is zero. ∎

## 7. Term collection and comparison are sound

Two terms have the same monomial when their reduction axes describe the same logical coordinate domain and their read lists admit a bijection pairing the same captured tensors at equal calculated coordinates. Read multiplicities must match. Equal axis sizes without coordinate identity are insufficient.

**Lemma 7 — Term comparison.** If two terms have equal coefficients and the same monomial under the established coordinate relationships, their values are equal.

**Proof.** Equal logical reduction axes with consistent sizes give a bijection between the reduction assignments, even if the axis sequences use different orders or partitioned descriptions. At corresponding assignments, each matched pair of reads accesses the same captured tensor at the same physical coordinate tuple. The products are equal by commutativity, with every repeated factor retained by the read bijection. Equal coefficients preserve that equality. Summing over corresponding assignments proves the claim. ∎

**Lemma 8 — Collection and expression comparison.** Removing zero-coefficient terms and combining terms with the same monomial preserve values. If the comparator finds a coefficient-preserving bijection between the resulting terms, the two expressions have equal values.

**Proof.** A zero-coefficient term contributes zero. For terms with a common monomial \(m\) and domain \(D\), distributivity gives

\[
\sum_D c_1m+\sum_D c_2m=\sum_D(c_1+c_2)m.
\]

Thus each collection step preserves the sum. For comparison, Lemma 7 makes each matched pair equal; adding all pairs proves equality of the expressions. The bijection prevents omitting or duplicating terms. Overflow during coefficient collection must reject rather than wrap the coefficient. ∎

For example, the source in dot reassociation normalizes to reductions \(L\mathbin{+\!\!+}K\). The constructed RHS may normalize to \(I_{\mathrm{sel}}\mathbin{+\!\!+}K\mathbin{+\!\!+}J_{\mathrm{sel}}\). Refinement establishes that \(I_{\mathrm{sel}}\) and \(J_{\mathrm{sel}}\) partition \(L\), and their mappings preserve paired accesses. Lemma 7 therefore applies despite the different sequence order.

## 8. Soundness of acceptance

Define the abstract verifier to accept only after it has:

1. Checked that every candidate attribute expression refers only to source information and is evaluable on every admissible source instance.
2. Lowered the LHS and completed RHS using admissible coordinate frames, establishing all RHS legality obligations.
3. Successfully normalized and collected both complete value expressions.
4. Established term equivalence by Lemmas 7–8 and equality of the ordered output interfaces and shapes.

Read matching against the arithmetic skeleton may propose coordinate frames. Acceptance still requires the admissibility and complete-expression checks above. No lemma claims that the skeleton has the same value as the original expression.

**Theorem — Relative soundness of semantic verification.** Under the contracts in Section 3, acceptance by this verifier implies the contract in Section 1.

**Proof.** Fix any source information \(s\) such that the LHS is legal and the declared guard evaluates successfully to `true`. Lemma 1 ensures every established premise is valid for this instance.

The first acceptance check and the defined-evaluation contract give some \(a\) with \(f(s)\Downarrow a\). The second check and primitive reasoning establish \(\operatorname{LegalRHS}(s,a)\).

Now fix any compatible tensor values \(X\). Output-interface and shape agreement give the same physical output coordinate domain on both sides. At any coordinate \(o\) in that domain, choose the corresponding logical environment \(\eta_o\). Let \(e_L\) and \(e_R\) be the lowered complete expressions. Then:

\[
\begin{aligned}
\operatorname{LHS}(s,X)[o]
&=\llbracket e_L\rrbracket_{s,X,\eta_o}
&&\text{by Lemma 5},\\
&=\llbracket\operatorname{collect}(N(e_L))\rrbracket_{s,X,\eta_o}
&&\text{by Lemmas 6 and 8},\\
&=\llbracket\operatorname{collect}(N(e_R))\rrbracket_{s,X,\eta_o}
&&\text{by accepted term comparison},\\
&=\llbracket e_R\rrbracket_{s,X,\eta_o}
&&\text{by Lemmas 6 and 8},\\
&=\operatorname{RHS}(s,a,X)[o]
&&\text{by Lemma 5}.
\end{aligned}
\]

Thus the tensors have equal values at every coordinate and the same ordered shape. If the output has no elements, value equality is vacuous and shape agreement still holds. Since \(s\) and \(X\) were arbitrary, the quantified contract follows. ∎

**Corollary — Accepted synthesis results are correct.** Let a synthesis procedure propose any \(f\), and publish it only if the verifier accepts. Every published result satisfies the theorem, under its assumptions.

**Proof.** Apply the theorem to the accepted candidate. The argument does not depend on how synthesis chose the candidate. ∎

Consequently, proving that the `KeepAxes` heuristic always finds a solution is not required for this soundness theorem. That would concern completeness. A mistaken candidate must fail verification unless it independently meets all acceptance conditions. This conclusion relies on the verifier checking admissible coordinates and side conditions, not merely running in a fresh context.

## 9. Connection to the implementation and remaining obligations

The code has the structure used by the theorem:

| Proof component | Implementation |
| --- | --- |
| Binary scalar expressions and tensor meanings | [`graph.h`](../src/core/semantics/graph.h) |
| Graph composition, index substitution, and operation obligations | [`graph.cc`](../src/core/semantics/graph.cc) |
| Mapping composition and partition refinement | [`axis_bindings.cc`](../src/core/semantics/index_maps/axis_bindings.cc), [`axis_maps.cc`](../src/core/semantics/index_maps/axis_maps.cc) |
| Coordinate transport | [`axis_transport.cc`](../src/core/semantics/index_maps/axis_transport.cc) |
| Normalization, collection, and bijective term matching | [`normalize.cc`](../src/core/semantics/symbolic/normalize.cc) |
| Calculated-coordinate comparison | [`index_expr.cc`](../src/core/semantics/symbolic/index_expr.cc) |
| Source-scope validation, definedness checks, and failing obligations | [`context.cc`](../src/core/semantics/context.cc) |
| Fresh verification and final acceptance checks | [`verify.cc`](../src/core/semantics/proof/verify.cc) |
| Verification of synthesized candidates before returning success | [`prove_rule.cc`](../src/core/semantics/prove_rule.cc) |

This correspondence is evidence that the mathematical procedure models the intended implementation. It is not a proof of full implementation refinement. In particular, these obligations remain before claiming an unconditional theorem about the current compiler:

- **Metadata reasoning:** justify every successful inference, canonicalization, learned fact, and cache reuse in the primitive checker. A fresh context does not establish this automatically.
- **Coordinate interpretation:** establish that source-frame construction, repeated input occurrences, transport, partition aliases, and RHS read anchors always satisfy the admissibility contract, including calculated accesses and output-only domains.
- **Partial evaluation:** show that every constructor accepted as well-defined evaluates successfully with the actual runtime integer representation. Include intermediate overflow, indexing, inverse permutations, and division preconditions; source-only syntax is insufficient.
- **Guard semantics:** connect declared-guard lowering to successful runtime evaluation, preserving ordered local bindings and short-circuit behavior. Algebraic treatment of Boolean expressions alone does not justify evaluation safety.
- **Operation specifications:** justify any claim that the declared semantics describes the actual target operators. The relative theorem applies to the declarations themselves.
- **Executable lowering:** if the claim covers application by generated code, prove that emitted attribute and guard programs implement the checked programs, including their failure behavior.

The paper can state the relative theorem with these contracts explicit and include the core proofs above in an appendix. It should not describe the C++ implementation as formally proved sound until the applicable implementation obligations have been discharged. Bounded proof search and conservative rejection affect completeness; they do not affect the theorem when an unresolved attempt never becomes acceptance.
