# Complete attribute synthesis for algebraic tensor rewrites

**Mathematical proof draft, October 22, 2026.** This is the unified proof for
contraction, permutation, unary summation, and a restricted additive fragment.
The [paper version](complete_attribute_synthesis_paper.md) gives the compact
statement. The proof concerns the specified mathematical generator; current
compiler coverage and remaining implementation obligations are in Section 9.

## 1. Semantics and supported fragment

All tensors have exact real entries. Input shapes have arbitrary finite ranks
and physical extents at least two; rank-zero tensors are allowed. Tensor
inputs are independently variable, without symmetry, sparsity, or other
value assumptions.

The operation contracts are:

| Operation | Descriptor, legality, and meaning |
| --- | --- |
| Permutation | A full permutation \(\pi\) of input positions; output position \(j\) reads input position \(\pi_j\). |
| Unary sum | An in-range duplicate-free list \(d\); sum over precisely those axes, remove them, and retain the other axes in operand order. |
| Contraction | Four ordered lists \((b_\ell,b_r,k_\ell,k_r)\), each in range and duplicate-free. Batch and contraction lists are disjoint on each operand; corresponding lists have equal lengths and paired extents. Multiply operand reads and sum over the paired contraction axes. Output order is batch-list order, followed by unselected left and right axes in operand order. |
| Addition | Elementwise addition of operands with identical ordered shapes; no attributes or broadcasting. |

Empty descriptor lists and scalar outputs are allowed. These are the entire
legality contracts: there are no extra layout or attribute restrictions.
They justify finite distributivity and changes of summation order under
exact arithmetic, not strict floating-point reassociation.

A rule specification \(\Sigma\) records the source and fixed RHS trees,
named inputs, operation contracts, captured source-field references, and
RHS attribute holes. The source may contain addition anywhere. Expand it by
linearity and bilinearity to \(L=\sum_{j\in J}L_j\). Require:

1. Each \(L_j\) is a contraction/permutation/unary-sum tree over a nonempty
   input set \(U_j\), using each input once.
2. The sets \(U_j\) are distinct. They may overlap, so an input can occur in
   several additive components.
3. The actual RHS is an explicit sum \(R=\sum_{j\in J}R_j\), where \(R_j\)
   is such a read-once tree over exactly \(U_j\).
4. Every RHS descriptor is an independent unrestricted unknown. There are
   no pins, explicit descriptors, shared unknowns, constants, or other
   operations in this fragment.

These are finite syntactic admission checks, independent of whether
replacement attributes exist. In particular, RHS normalization must not
copy one original unknown into independent branch-local unknowns.
Source expansion may copy *captured* descriptors; all copies retain their
original values. Expansion preserves legality and the source output shape:
linearity preserves values, and substituting an equally shaped summand does
not change an enclosing operation's descriptor legality or output shape.

Metadata \(p\) supplies the input shapes and all captured source attributes,
including any sharing between source occurrences. Define

\[
\begin{aligned}
\operatorname{Valid}_\Sigma(p,\beta)\iff{}&
 \operatorname{Legal}_R(p,\beta)\\
&\land\operatorname{Shape}_R(p,\beta)=\operatorname{Shape}_L(p)\\
&\land\forall X:\ \llbracket R\rrbracket_{p,\beta,X}
                    =\llbracket L\rrbracket_{p,X}.
\end{aligned}
\tag{1}
\]

Legality includes every intermediate operation; shapes are ordered vectors.
Attributes depend on metadata, not tensor contents: the quantifier order is
\(\exists\beta\,\forall X\). All subsequent metadata quantifiers assume legal
source operations and the extent domain above. A single summand gives the
non-additive fragment.

## 2. An exact axis representation

Analyze one source component. Give every physical input axis a named port
\((c,i)\). Replay its descriptors: merge paired batch or contraction ports,
reorder exposed ports at permutations, and remove exposed ports at unary
sums. Retain the classes of removed ports. Write

\[
N=(P,\sim,F),\qquad
\mathcal A=P/{\sim},\qquad
I_c=(a(c,0),\ldots,a(c,r_c-1)),
\]

where \(F\) is the ordered output-class sequence and \(d_a\) is class \(a\)'s
extent. Classes are sets of named input ports, not dimension sizes.

Read-once trees have a useful wire property: each class has at most one port
per input and at most one live occurrence at any node. Binary pairings join
components from disjoint input subtrees; a removed component cannot reappear.
Two exposed components from the same child can never be joined later.
Unary sums remove wires without merging them. Every class contains an input
port, including classes reduced from only one input.

Expanding the finite sums and products gives the network denotation

\[
\mathcal T_N(X)[y]=
\sum_{\substack{t_a\in[0,d_a)\ (a\in\mathcal A)\\
                 t_{F_i}=y_i\ \text{for every }i}}
\prod_c X_c[(t_{a(c,0)},\ldots,t_{a(c,r_c-1)})].
\tag{2}
\]

**Lemma 1 (Axis exactness).** Two legal read-once components over the same
physical inputs are equal for every input value, with the same ordered
output shape, if and only if their port partitions and ordered output
attachments agree.

**Proof.** Equal representations make (2) identical up to dummy-index
renaming. Conversely, first reject different output shapes. Attach a named
port to every output position. If the resulting partitions differ, choose
two ports joined in one partition but separated in the other. Assign their
separated classes coordinates 0 and 1, and all other classes 0. These
coordinates exist because every extent is at least two.

Set each input to one at its resulting coordinate tuple and zero elsewhere.
At the corresponding output coordinate, the separating network returns one:
every summation coordinate is fixed by an input port, so exactly one term
contributes. The identifying network returns zero because it requires the
chosen 0 and 1 to agree. Scalar inputs contribute their single entry, chosen
as one. Thus different representations cannot agree for every input.
\(\square\)

This lemma concerns the full partition, including reduced classes and
output attachments. Equal output-axis sets or shapes alone are insufficient.

## 3. Canonical live axes

For a class \(a\), let \(\sigma(a)\) be its supporting inputs. At RHS node
\(v\), let \(U_v\) be the inputs below it. The mandatory axes are

\[
M_v=\{a:\sigma(a)\cap U_v\ne\varnothing
 \land(a\in\operatorname{set}(F)\lor\sigma(a)\nsubseteq U_v)\}.
\tag{3}
\]

A final-output class or one needed outside the subtree cannot disappear.
Other classes may remain exposed until a later unary sum. Therefore
\(M_v\) is not generally the actual live set.

Instead compute canonical live sets \(P_v\) bottom-up:

\[
\begin{array}{ll}
P_v=\operatorname{set}(I_c),&v\text{ is input }c,\\
P_v=P_u,&v\text{ permutes child }u,\\
P_v=P_u\cap M_v,&v\text{ sums child }u.
\end{array}
\tag{4}
\]

For a contraction with children \(\ell,r\), use

\[
\begin{aligned}
B_v&=P_\ell\cap P_r\cap M_v,&
K_v&=(P_\ell\cap P_r)\setminus M_v,\\
LF_v&=P_\ell\setminus P_r,&
RF_v&=P_r\setminus P_\ell,\\
P_v&=B_v\cup LF_v\cup RF_v.&&
\end{aligned}
\tag{5}
\]

Thus a dot contracts every completed shared non-output class; a unary sum
removes every completed non-output class still present. An unshared class
must remain free at a dot, even if no longer needed. Direct induction gives
\(M_v\subseteq P_v\), with each class in \(P_v\) supported inside \(U_v\).

**Lemma 2 (Canonicalization preserves existence).** If a valid assignment
has intermediate layouts \(W_v\), another valid assignment has exactly
\(W_v|_{P_v}\) at every node. In particular,
\(P_{\mathrm{root}}=\operatorname{set}(F)\).

Here \(w|_A\) filters a sequence by membership in \(A\).

**Proof.** By Lemma 1, a valid RHS has the source partition. Hence every
mandatory class is exposed where needed. We show by induction that
\(P_v\subseteq\operatorname{set}(W_v)\) and the projected layouts can be
realized:

- Inputs are unchanged; a full permutation can realize any ordering of its
  projected child.
- A unary sum retains the mandatory classes in its child. These also occur
  in the original output; removing other projected axes produces precisely
  \(W_v|_{P_v}\), since sums preserve survivor order.
- At a dot, a class shared by both canonical children must be joined here.
  If left separate, its components would arrive through the same child at
  every later ancestor and could never join. A shared mandatory class must
  be batched in the original assignment. A shared completed class may have
  been batched or contracted; contracting it now removes only an unnecessary
  output axis.

  A canonical left-free class cannot occur in the original right child.
  If it did, support in both children would make it mandatory in the right
  child, hence present in \(P_r\), a contradiction. Likewise for right-free
  classes. Consequently survivors keep their batch/free roles. Filter the
  original batch order to \(B_v\), contract \(K_v\) in any common order,
  and select their positions in the projected children. The output is
  exactly \(W_v|_{P_v}\).

All descriptors are legal: selected axes are present once, paired extents
agree, and batch/reduction sets are disjoint. These changes join no distinct
source classes and eliminate a class only after all its supporting ports
have joined. At the root, \(M_{\mathrm{root}}=\operatorname{set}(F)\) and
\(W_{\mathrm{root}}=F\), so the inclusions force
\(P_{\mathrm{root}}=\operatorname{set}(F)\). The projected expression has the
same network and final order, hence is equivalent by Lemma 1. \(\square\)

This is an early-elimination *normal form*, not a claim that every valid
assignment has the same intermediate sets. Without unary sums on either
side, the canonical sets reduce to the original mandatory boundaries.

## 4. Exact order constraints and a constructor

First require the root-set check
\(P_{\mathrm{root}}=\operatorname{set}(F)\).
Propagate required subsequences, rather than choosing complete layouts:

- Set \(Q_{\mathrm{root}}=F\).
- At a permutation, pass the empty sequence.
- At a unary sum, pass \(Q_v\) unchanged.
- At a dot, pass \(Q_v|_{LF_v}\) left and \(Q_v|_{RF_v}\) right.

Let \(\preceq\) denote subsequence and \(\Vert\) concatenation. Define

\[
\phi(p)=
[P_{\mathrm{root}}=\operatorname{set}(F)]
\land\bigwedge_{\text{inputs }c}[Q_c\preceq I_c]
\land\bigwedge_{\text{dots }v}
[Q_v=(Q_v|_{B_v})\Vert(Q_v|_{LF_v})\Vert(Q_v|_{RF_v})].
\tag{6}
\]

These are source-metadata computations, not existential intermediate orders.

**Necessity.** Any valid assignment has a canonical realization by Lemma 2.
Its root order is \(F\). At a dot, output has batch/left-free/right-free
blocks, so every required subsequence respects this block order and passes
the appropriate free subsequence to each child. Batch lists can select child
positions in any order, imposing no batch-order requirement on children.
Unary sums preserve the order of surviving axes. A permutation imposes no
child requirement; an input has its fixed physical order. Top-down induction
therefore gives every check in (6).

**Construction and sufficiency.** On success, construct layouts bottom-up.
An input retains \(I_c\). A permutation puts \(Q_v\) first and then its
remaining child axes. A unary sum removes the positions of \(P_u\setminus P_v\).
At a dot, enumerate \(B_v\) by a list \(b\) beginning with \(Q_v|_{B_v}\);
append remaining batch classes in a fixed class order. Enumerate \(K_v\)
by any fixed list \(k\). For child layouts \(W_\ell,W_r\), emit

\[
\begin{aligned}
\beta_v={}&(\operatorname{pos}(W_\ell,b),\operatorname{pos}(W_r,b),
           \operatorname{pos}(W_\ell,k),\operatorname{pos}(W_r,k)),\\
W_v={}&b\Vert(W_\ell|_{LF_v})\Vert(W_r|_{RF_v}).
\end{aligned}
\tag{7}
\]

Position lookup lists the positions of the given classes. A permutation
uses the same lookup to realize its chosen layout; a sum uses it for the
removed classes.

Induction gives \(\operatorname{set}(W_v)=P_v\) and \(Q_v\preceq W_v\):
(6) supplies the required block order, while sums preserve the required
subsequence. Hence all lookups are defined and every descriptor is legal.
Every source class spanning both children is present in both canonical
sets, since it is mandatory there; the dot joins its occurrences. No
distinct source classes merge. Elimination occurs only after support is
complete. Inductively the partial partition is the source partition
restricted to the subtree's ports. At the root, the set check and
\(F\preceq W_{\mathrm{root}}\) force \(W_{\mathrm{root}}=F\); all non-output
classes have disappeared. Lemma 1 proves equivalence.

Consequently (6) holds exactly when a valid component assignment exists,
and (7), with the unary cases, constructs one.

## 5. Additive decomposition

Scale each independent input \(X_c\) by a real scalar \(z_c\). Read-once
multilinearity, including unary summation, gives

\[
L_j((z_cX_c)_c)=\left(\prod_{c\in U_j}z_c\right)L_j(X),
\]

and the same equation for \(R_j\). Distinct \(U_j\) yield distinct monomials.
At any output coordinate, uniqueness of polynomial coefficients therefore gives

\[
\forall X:\ L(X)=R(X)
\quad\Longleftrightarrow\quad
\forall j\;\forall X:\ L_j(X)=R_j(X),
\tag{8}
\]

for legal, equally shaped expressions.

For component \(j\), let \((\phi_j,f_j)\) be the checker and constructor of
Section 4. Define

\[
\Phi_\Sigma(p)=\bigwedge_{j\in J}\phi_j(p),\qquad
f_\Sigma(p)=\bigcup_{j\in J}f_j(p)
\quad\text{on success}.
\tag{9}
\]

The union assigns disjoint descriptor identifiers. Components use separate
axis partitions: addition aligns output coordinates but does not identify
input-port classes from different terms. All successful components have the
source shape, so their sum is legal.

**Theorem (Complete constructive attribute synthesis).** For every admitted
specification and legal in-domain metadata,

\[
\boxed{
\begin{aligned}
\Phi_\Sigma(p)&\iff\exists\beta\;
                    \operatorname{Valid}_\Sigma(p,\beta),\\
\Phi_\Sigma(p)&\Rightarrow
                    \operatorname{Valid}_\Sigma(p,f_\Sigma(p)).
\end{aligned}
}
\tag{10}
\]

**Proof.** A valid full assignment makes every component valid by (8).
Section 4 implies each \(\phi_j\). Conversely, if every check holds, the
component constructors yield valid assignments. Independence permits their
union; equal output shapes make the sum legal, and (8) gives full equality.
The single-term case proves completeness for contraction/permutation/sum
trees. Omitting unary sums gives the original contraction/permutation case.
\(\square\)

## 6. Uniform generation of metadata programs

The theorem yields finite programs, not an enumerated table of ranks.
Metadata instructions have their mathematical meanings: source-field reads,
finite partitions and merges, filtering, set operations, concatenation,
equality, subsequence checks, position lookup, bounded loops, and conditionals.

A syntax-directed generator precomputes subtree input sets and emits:

~~~text
GENERATE(Sigma):
    expand source and match terms by their distinct input sets
    for each component j:
        emit source replay: input ports, pairings, removals, ordered interfaces
        emit postorder computation of M, P, B, K, LF, RF
        emit root-set check and preorder propagation of Q
        emit the input and dot order checks
        emit postorder construction using (7) and the unary cases
    emit Phi(p): return conjunction of component checks
    emit f(p):
        if not Phi(p): return NotApplicable
        return union of the constructed component assignments
~~~

Source replay normalizes all recorded ports through the completed source
partition. Reduced classes remain recorded. Static component scopes prevent
one source occurrence copied by expansion from becoming a fresh metadata
choice, or logical classes from different components being merged.

Induction over emitted traversals establishes four invariants: replay equals
the source network; bottom-up sets equal (3)--(5); propagated requirements and
the Boolean equal (6); successful construction equals (7) and its unary cases.
Loop-prefix invariants give the corresponding statements for variable-length
axis lists. The successful branch has all lookup preconditions by Section 4;
a false check bypasses construction. Thus evaluation implements (9)--(10).

Generation traverses finite syntax, including finite source expansion.
Execution traverses finite trees and metadata collections. Both terminate;
ranks determine loop lengths, not program structure. Expansion can be
exponential in the original expression size; this is not a general
polynomial-time normalization claim.

## 7. From declared TEPL information to a generated condition

The symbolic specification is extracted as

\[
\Sigma=E(T)=(\mathcal C,\mathcal G_L,\mathcal G_R,
             \mathcal D,\mathcal R,\mathcal H),
\]

recording named captures, operand trees, operation contracts, source-field
references, and ownership of independent RHS holes. Variable-length input
axis sequences are references evaluated using \(p\), not fixed-rank lists.

Admission matches declarations structurally against Section 1's templates:

- Permutation: a full-input axis group, one permutation field, permuted output,
  and transported operand reads.
- Unary sum: selected reduction axes, the ordered complement as output,
  one reduction-list field, and a sum of the operand over the selected group.
- Contraction: four distinct fields, paired batch/reduction groups, ordered
  free complements, the three output blocks, and reduction of operand products.
- Addition: aligned full input interfaces, their shared ordered shape, and
  the sum of operand values.

Allow consistent renaming, concatenation reassociation, equivalent ordering
of exclusions in complements, and reversal of scalar product factors.
These finite checks exclude additional assertions or fields and do not call
a general semantic-equivalence oracle. Check read-once components, distinct
input sets, and descriptor ownership after the specified source expansion.

A local field adapter \(\rho_v\) maps declaration fields to canonical roles.
It is a bijection because the template fields are distinct and exhaustive.
Source adapters read the same captured values, retaining all sharing.
The product of RHS adapters is a bijection on complete assignments.
Structural induction over templates preserves legality, ordered shapes,
and values; source expansion preserves the same properties. Thus declaration
validity is equivalent to canonical validity, and no possible attribute
assignment is lost in translation.

For a metadata language providing the exact analysis queries and finite-list
operations, emit (6) and (9) as an explicit condition \(W_T\). Interpret every
query in its component and occurrence scope. Queries abbreviate the finite
procedures in Section 6, not calls to an attribute-existence oracle.
Structural induction over the
emitted expressions gives

\[
\operatorname{Eval}(W_T,p)=\Phi_{E(T)}(p).
\tag{11}
\]

Combining (10)--(11) with the assignment adapters proves the surface claim:
the specified translation produces a checker that accepts exactly when
valid attributes exist, and a constructor that returns them on success.

The mathematical feasibility program uses the source replay and RHS structural
requirements above. In executable TEPL, users instead write source metadata
restrictions that the compiler proves sufficient. For example:

~~~tepl
(source: dot (inner: dot A B) C) => (dot A (dot B C))
where {
    source.attrs.lhs_batching_dimensions == inner.axes.B.positions
}
~~~

This is a sufficient restriction, not a presentation of the complete feasibility
formula. Labels identify operation occurrences without adding tensor captures.
Source logical-axis identities differ from numeric positions and extents.
Runtime inferred-RHS queries and `axis_domain()` have been removed from the
language's executable path. The generalized mathematical generator still needs
canonical live/root-set analysis and component scopes; those definitions do not
claim existing public query support.

An additional defined, terminating condition \(C(p)\) gives mathematical-solver applicability
\(C(p)\land\Phi_\Sigma(p)\). Guaranteeing that every match satisfying \(C\)
is feasible requires the separate implication \(C\Rightarrow\Phi_\Sigma\)
within the source domain. This proof does not decide arbitrary implications,
nor require every equivalent spelling of a generated condition to be
recognized by the current compiler.

## 8. Examples, obstructions, and restrictions

**Unary summation.** For \(A:[M,K]\), \(B:[K,N]\),

\[
\sum_{i,j}(AB)_{ij}
=\sum_k\left(\sum_i A_{ik}\right)\left(\sum_j B_{kj}\right).
\]

The canonical RHS removes \(M,N\) in its two unary sums and contracts \(K\)
at the dot. All three descriptors are inferred together. Conversely,
a source summing an input axis cannot be replaced by permutations alone:
the unwanted axis remains in \(P_{\mathrm{root}}\), failing the set check.

**Distribution and LoRA.** Distribution has components \(\{A,C\}\) and
\(\{B,C\}\); copying the source contraction descriptor supplies a witness.
LoRA, \(X(W+AB)\to XW+(XA)B\), has components \(\{X,W\}\) and \(\{X,A,B\}\).
The first is always realizable; the second is an exact ordered-reassociation
problem. Some generalized-dot matches fail it, despite the scalar identity.

**Output order.** For independent vectors, a transposed outer product has
order right-free then left-free. A single dot without a permutation must
output left-free then right-free. Equal extents do not repair this mismatch.

Failures have checkable structural witnesses: an unwanted canonical root
axis, an input-order inversion, or a pair of required axes reversing the
dot's batch/left-free/right-free order. Recomputing the analysis validates
the witness; (10) excludes every assignment in that component and full rule.
This is an impossibility claim only on legal in-domain metadata.

The essential boundaries are:

| Restriction | Reason |
| --- | --- |
| Independent read-once inputs per term | Exact wire representation and one-hot separation. |
| Distinct term input sets | Separates terms by independent capture scaling. Same-degree sums need additional reasoning. |
| Extents at least two | Distinguishes different partitions using coordinates 0 and 1. Empty or singleton extents are outside the theorem. |
| Independent unrestricted RHS descriptors | Canonical projection and branch witnesses can be realized without coupled constraints. |
| Exact operation and arithmetic contracts | Early elimination, finite distribution, and reordered reductions preserve meaning. |

For example, \(A+\operatorname{transpose}(B)\) cannot generally be realized as
\(\operatorname{transpose}_p(A)+\operatorname{transpose}_p(B)\) with one shared
unknown \(p\): the components separately require identity and swapping.
Independent component feasibility is not sufficient when descriptors share.
Likewise, two terms with the same input set cannot be paired independently
merely by the scaling argument.

No completeness claim is made for subtraction/cancellation, repeated inputs
within a component, other reducers, keepdims, general scalar functions,
convolution, or extra backend restrictions.

## 9. Current implementation and evidence

The standalone native exact solver covers the original read-once
contraction/permutation specialization. Executable TEPL uses independently
verified symbolic constructors under written source conditions. The generalized
unary-sum and additive complete algorithms above are mathematical extensions
requiring integration; general proofs of some such rules do not imply complete
attribute search.

| Component | Existing implementation and remaining work |
| --- | --- |
| Admission and declaration roles | [contract.cc](../../src/core/semantics/synthesis/ordered/contract.cc) recognizes contraction/permutation contracts. Extend templates, descriptor adapters, and admission for sums and additive components. |
| Runtime analysis and construction | [evaluate.cc](../../src/inference/evaluate.cc) and [axis_analysis.cc](../../src/rewrite/axis_analysis.cc) implement the original source replay, forced boundaries, orders, and constructor. Generalization requires (3)--(7), unary instructions, and the root-set check. |
| Compilation | [compile.cc](../../src/core/semantics/synthesis/ordered/compile.cc) emits the original schedules. Add component expansion/scopes and the generalized schedules. |
| Source restrictions | [restrictions.cc](../../src/core/semantics/restrictions.cc) searches a bounded vocabulary and verifies sufficient source restrictions. Complete generalized feasibility analysis remains a separate extension. |

The general semantic verifier already proves some additive/reduction rules;
that does not establish complete attribute search for them. Fully explicit
RHS validation checks the supplied assignment rather than existential
attribute existence.

Finite tests support implementation correspondence, not arbitrary-rank
correctness. A separate small exhaustive descriptor experiment checked the
generalized live-set analysis on 1,224 source/RHS cases, including delayed
reductions and reassociation: all decisions agreed and all 629 constructed
solutions had the source network. This corroborates canonicalization but
does not replace its proof or independently prove the network semantics.

Formal refinement remains necessary for a theorem about the actual C++
frontend, symbolic normalizers, metadata queries, generator, and runtime.
The declared operation contracts must also match the intended backend.

Finally, resource/domain failures must remain distinct from infeasibility.
The mathematical programs terminate over finite metadata with sufficient
resources. Machine executions additionally need representable indices,
checked arithmetic, and sufficient memory/work budget. Resource exhaustion,
unsupported schemas, or outside-domain data do not prove that attributes
cannot exist.

## 10. Practical TEPL acceptance and restriction suggestions

The completeness theorem concerns the specified mathematical generator and its
admitted domain. It is a specialized decision-and-construction procedure, not
a restriction on every semantic rule expressible in TEPL. The practical compiler
also uses general candidate construction and independent symbolic verification.
No completeness claim is made for that extended search.

Let \(L(p)\) denote source legality, \(C(p)\) the written rule conditions, and
\(V_f(p)\) mean that the generated metadata program \(f\) is defined and makes
the replacement legal, equally shaped and typed, and equivalent under the
selected arithmetic policy. A semantic rule is accepted only on a proof of

\[
\forall p:\quad L(p)\land C(p)\Rightarrow V_f(p).
\]

Within the completeness domain, proving \(C\Rightarrow\Phi_\Sigma\) under
source legality is sufficient for the theorem's constructor. Domain assumptions
must also be established; they cannot silently become match-time restrictions.
Outside this domain, direct verification can establish the same acceptance
contract without a complete feasibility decision. In particular, singleton and
zero extents, mixed explicit/automatic descriptors, and repeated-capture
identity cases may be verified by the general path. Failure there is Unknown,
not an impossibility certificate. Strict floating-point equivalence is not
inferred from an ideal-arithmetic proof.

The implementation proposes copied descriptors for identical operation trees
and verifies operation, operand, and attribute identity directly. Other candidates
are checked in a fresh context using source legality and the written conditions.
All constructor definedness and legality obligations must be discharged before
materialization. Unresolved obligations cannot become hidden runtime guards.
The old runtime inferred-applicability compilation path has been removed.

If the original rule cannot be proved, `check --suggest-where` searches a bounded
vocabulary of source-group equalities and identity-permutation predicates.
For a proposed conjunction \(R\), fresh synthesis and independent verification
must establish

\[
\forall p:\quad L(p)\land C(p)\land R(p)\Rightarrow V_f(p).
\]

A separate concrete metadata search must provide a legal source witness for
\(L\land C\land R\), including source dtype legality. This witness excludes
vacuous recommendations; it is not evidence for the universal implication.
The current search tries individual predicates and pairs, followed by deletion
checks. It may fail to find a restriction or witness even when one exists. A
recommendation is verified sufficient; neither global minimality nor weakest
applicability is claimed. Suggestions do not accept the original rule or alter
its executable conditions: the user reviews and adds them to source, then
normal compilation checks the resulting rule again.

The mathematical theorem, the practical acceptance contract, and implementation
correspondence are separate claims. Unit, CLI, constructor-corruption, and
serialization tests provide implementation evidence, not a formal refinement
proof of the compiler or verifier.
