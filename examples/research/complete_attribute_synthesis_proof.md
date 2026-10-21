# Completeness of ordered attribute synthesis

**Status: mathematical proof draft, October 21, 2026.** This document gives a
self-contained hand proof of the concrete structural algorithm implemented in
[`synthesis/ordered`](../src/core/semantics/synthesis/ordered/README.md). It proves
necessity, constructive sufficiency, termination, and certificate properties for
the fragment defined below. Section 12 adds a specified abstract program
generator and proves that its generated feasibility checks and constructors
implement this algorithm uniformly over arbitrary finite ranks. It is not a
proof-assistant development or a formal verification of the C++ implementation.
Section 13 defines the supported declarative TEPL fragment and proves that its
semantic information suffices to generate an exact feasibility checker and a
correct constructor. Section 10 records the implementation correspondence and
remaining boundaries.

The central result is

\[
\boxed{
\exists\beta\;\operatorname{Valid}(S,p,\beta)
\quad\Longleftrightarrow\quad \Phi_S(p)
\quad\Longleftrightarrow\quad
\operatorname{Infer}(S,p)=\operatorname{Solved}(\beta_0),
}
\]

where the returned \(\beta_0\) is itself valid. The predicate \(\Phi_S(p)\)
depends only on source metadata and the replacement tree. Its definition does
not quantify over, or require knowing, replacement attributes.

The main generated-program result (Theorem 3) strengthens this to a terminating
generation procedure \(\operatorname{Generate}(S)=(\Phi_S,f_S)\): the generated
predicate decides attribute existence exactly, and the generated constructor
terminates with valid attributes whenever that predicate holds. Here a generated
predicate denotes a finite metadata program, not an enumerated table of ranks.

Theorem 4 connects that abstract result to the information expressed by
supported TEPL declarations. It concerns the specified translation and
generator; implementation of this path in the current compiler remains open.

## 1. Domain, legality, and quantifiers

A schema \(S=(L,R)\) consists of a source expression and a fixed replacement
skeleton. Both are finite trees over the same nonempty finite set of named
captures \(\mathcal C\), with each capture occurring exactly once per tree.
Captures denote independently variable tensors. Each internal node is either
the contraction or permutation specified below. Every operation in \(R\) has
its own unrestricted unknown descriptor; there are no pins, shared unknowns,
or additional attribute restrictions.

Concrete metadata \(p\) gives the physical shape of each capture and every
source descriptor. Each physical extent is a finite integer at least two.
Rank-zero tensors, empty batch/contract lists, and scalar outputs are allowed.
All source operations are legal. Tensor scalars are exact real numbers;
finite sums and products have their mathematical meanings.

### 1.1 Operation contracts

A permutation on a rank-\(r\) operand has a descriptor \(\pi\) that is a
permutation of \([0,r)\). Output axis \(j\) is input axis \(\pi_j\).
Coordinate values are transported accordingly; no axis is created or reduced.

A contraction has four ordered position lists \((b_L,b_R,k_L,k_R)\):

- Each list is in range and has no duplicates.
- \(b_L\) and \(k_L\) are disjoint; \(b_R\) and \(k_R\) are disjoint.
- \(|b_L|=|b_R|\), \(|k_L|=|k_R|\), and each corresponding pair has equal extent.

Let \(f_L\) and \(f_R\) be the unselected positions of their operands, in
operand order. The output axes are, in order, the paired batch axes in list
order, then \(f_L\), then \(f_R\). At output coordinates \((u,v,w)\),

\[
D[u,v,w]=\sum_z A[\iota_L(u,v,z)]B[\iota_R(u,w,z)],
\]

where \(\iota_L\) places \(u,v,z\) in positions \(b_L,f_L,k_L\), and
\(\iota_R\) places \(u,w,z\) in positions \(b_R,f_R,k_R\). The sum ranges
over the Cartesian product of the paired contraction extents. An empty product
of coordinate domains contains the single empty assignment.

These are the entire mathematical legality contracts. Backend dtype,
precision, or layout restrictions require separate treatment. In particular,
this theorem does not claim strict floating-point reassociation is valid.

### 1.2 Valid replacement

For fixed legal \(p\), define

\[
\begin{aligned}
\operatorname{Valid}(S,p,\beta)\iff {}&
\operatorname{Legal}(R,p,\beta)\\
&\land\operatorname{Shape}(R,p,\beta)=\operatorname{Shape}(L,p)\\
&\land\forall X:\ \llbracket R\rrbracket_{p,\beta,X}
                         =\llbracket L\rrbracket_{p,X}.
\end{aligned}
\]

Legality includes every intermediate operation. Shapes are ordered vectors.
The assignment \(\beta\) may depend on \(p\), but must work for **all** tensor
values \(X\). The quantifier order is \(\exists\beta\,\forall X\).
No assumptions about tensor values, such as symmetry or sparsity, are used.

The fragment is defined by these contracts before running inference. Admission
does not mean that inference succeeds. In particular, unrealizable replacement
trees remain members of the fragment.

### 1.3 Explicit supported fragment and metadata domain

Call the abstract schema fragment above \(\mathcal F_0\). Its expression
grammar is

\[
E ::= c\mid\operatorname{Perm}_{a}(E)
          \mid\operatorname{Dot}_{a}(E,E),
\]

where \(c\) is a named capture and \(a\) is a descriptor reference. The
grammar has the following additional, independently checkable restrictions:

| Schema property | Restriction |
| --- | --- |
| Operations | Exactly the axis, order, value, and legality contracts of Section 1.1. |
| Input use | The same named captures occur exactly once per side; their values vary independently. |
| Source descriptors | Captured metadata references; their values are supplied at each match. Source references may be shared, and such sharing is preserved. |
| Replacement descriptors | A distinct unrestricted unknown per operation, with no pinned fields, derived expressions, or sharing with other RHS or source descriptors. |
| Arithmetic | The ideal exact-real interpretation of Section 1. |
| Extra features | No other operators, tensor literals, coordinate arithmetic, or additional attribute restrictions. |

Schema admission is independent of ranks, descriptor values, or feasibility.
For an admitted schema, define the separate metadata-domain predicate

\[
D_S(p)\iff\text{every physical input extent in }p\text{ is finite and at least two}.
\]

Source legality is checked separately. All theorem quantifiers over metadata
are restricted to \(\operatorname{Legal}_L(p)\land D_S(p)\). This domain
allows arbitrary finite ranks and scalars. A singleton/empty dimension lies
outside the theorem; it is not evidence that no replacement attributes exist.

Section 13 defines \(\mathcal F_T\), a declarative TEPL fragment that translates
into \(\mathcal F_0\). Membership is determined by explicit semantic templates
and rule structure, not by invoking synthesis or attempting to prove an arbitrary
operator equivalent to a contraction. User metadata conditions restrict the
matches under consideration; they are not premises used to define membership
by successful inference.

## 2. Exact source representation

Give every physical capture axis a named port \((c,i)\). Let \(P\) be the set
of ports, retaining their extents and physical order within each capture.
Replay the legal source:

1. A leaf exposes its ports in physical order.
2. A permutation rearranges its exposed ports.
3. A contraction identifies each paired batch or contraction axis. A batch
   exposes one representative; a contraction exposes none. Unpaired axes
   retain their operand order.

Keep all identifications, including those for axes that have been reduced.
The resulting equivalence classes \(\mathcal A=P/{\sim}\) are logical axes.
Write \(a(c,i)\) for a port's class, \(d_a\) for its common extent, and
\(F\) for the ordered sequence of classes exposed at the source root. Let
\(\bar F\) denote the set of entries of \(F\). The network is
\(N=(P,{\sim},F)\), with port shapes implicit in \(P\).

### Lemma 1. Wire structure

For every legal expression in the fragment:

1. Each logical class contains at most one port from any one capture.
2. Each intermediate exposes at most one wire of each final logical class.
3. Each class has a nonempty set of input ports. It terminates either in one
   final output position or in one contracting pair. A reduced class contains
   ports from at least two captures.

**Proof.** Initially each axis is a distinct component with one exposed wire.
At a binary node, the two subtrees use disjoint captures. Legal descriptors
select each exposed wire at most once. Pairing therefore merges two distinct
components, one from each child; it never merges two components of the same
child. A batch leaves exactly one exposed wire, and a contraction leaves none.
Unpaired components leave one exposed wire each. A component without an exposed
wire can never be selected by an ancestor. Permutations only rearrange exposed
wires.

Inductively every component is a tree of merges whose leaves are input ports
and whose only possible continuing wire is its root. Distinct exposed
components of an intermediate cannot later merge with each other: at every
ancestor they arrive through the same child, while each pairing selects one
component from each of two children. Thus later merges cannot invalidate
distinctness at earlier intermediates. Disjoint capture sets at each merge
give statement 1, single continuing wires give statement 2, and termination
of a merge tree gives statement 3. A contraction necessarily merges two
nonempty components. ∎

### Lemma 2. Network denotation

At output coordinate \(y\), the expression denotes

\[
\mathcal T_N(X)[y]=
\sum_{\substack{t_a\in[0,d_a)\ (a\in\mathcal A)\\
                 t_{F_j}=y_j\ \text{for all }j}}
\prod_{c\in\mathcal C}
 X_c[(t_{a(c,0)},\ldots,t_{a(c,r_c-1)})].
\tag{1}
\]

In this notation the free coordinates are fixed, so the sum enumerates only
the non-output classes. No coordinate variable is absent from all reads.

**Proof.** Induct over the expression. The leaf case is its ordinary tensor
read. A permutation changes only which exposed class attaches to each output
position. A binary node multiplies the two child expressions, identifies their
selected batch and contraction coordinates, and sums the latter. Previously
reduced child classes are disjoint from each other and from all exposed
classes, by Lemma 1. Finite distributivity combines their sums with the new
contraction sum to give (1). Each capture contributes one factor. Every class
contains an input port, so every coordinate occurs in a read. ∎

### Lemma 3. Semantic exactness

For two legal expressions over the same physical captures and shapes,

\[
\bigl(\text{same ordered output shape and equal tensors for every }X\bigr)
\quad\Longleftrightarrow\quad
\bigl(\text{same port partition and same ordered output classes}\bigr).
\]

Class names are irrelevant: classes are compared as sets of named ports.

**Proof, sufficiency.** The right side makes (1) identical up to renaming
summation coordinates. Corresponding class extents agree because the port
shapes are fixed. Hence both output shapes and every output value agree.

**Proof, necessity.** Different output ranks or shapes already violate tensor
equality. Otherwise add one named port \(o_j\) for each output position and
attach it to class \(F_j\). Equality of the resulting partitions on
\(P\cup\{o_j\}\) is equivalent to equality of the original partition and
ordered output attachments. Every class still contains an input port.

Suppose these augmented partitions differ. There exist two ports identified
in one partition, say \(N_2\), and separated in the other, \(N_1\). Assign
coordinate 0 to one separated \(N_1\) class, coordinate 1 to the other, and
coordinate 0 to all remaining \(N_1\) classes. This is in range on every port
because all extents are at least two. It specifies an input coordinate tuple
\(x_c\) for each capture and an output coordinate \(y\).

Choose independent one-hot inputs: \(X_c[x_c]=1\) and every other entry of
\(X_c\) equals zero. For \(N_1\), precisely one summation assignment can
contribute a nonzero product: each class is fixed by an input port. It exists
by construction, so \(\mathcal T_{N_1}(X)[y]=1\). For \(N_2\), a nonzero
product would force the same input tuples and output coordinate, including
both 0 and 1 on an identified pair. This is impossible, so
\(\mathcal T_{N_2}(X)[y]=0\). This contradicts equality for every \(X\).
Scalar captures contribute their single entry, chosen to be 1, and cause no
exception. ∎

This proves necessity at fixed extents, even when all extents are equal. It
does not rely on choosing generic dimension sizes or assuming that distinct
networks are distinguishable. The one-hot construction establishes that fact.

## 3. Forced axis sets and pairings

For a source class \(a\), define its capture support
\(\sigma(a)=\{c:(c,i)\in a\text{ for some }i\}\). Different classes may
have the same support; their port identities remain distinct.

For a replacement subtree with captures \(U\subseteq\mathcal C\), define

\[
\partial U=\{a\in\mathcal A:\sigma(a)\cap U\ne\varnothing
        \land(a\in\bar F\ \lor\ \sigma(a)\nsubseteq U)\}.
\tag{2}
\]

### Lemma 4. Forced boundary

Every valid replacement exposes exactly one copy of each class in
\(\partial U\) at that subtree, and no other classes.

**Proof.** By Lemma 3, a valid replacement has the source's complete network.
Consider a class touching \(U\). If it also touches a capture outside \(U\),
its component inside \(U\) must remain exposed to join the outside component.
Once reduced, it has no continuing wire and cannot be recovered. If the class
is a final output, it must likewise remain exposed. Lemma 1 ensures that at
most one continuing wire exists. This proves inclusion of \(\partial U\).

Conversely, let \(a\notin\bar F\) with \(\sigma(a)\subseteq U\). It has
at least two supporting captures by Lemma 1. Let \(v\) be their lowest common
ancestor in the replacement tree. This is binary and lies within the subtree
on \(U\); both children contain support. The two components must be joined at
\(v\): if left separate, they travel through the same child of every later
ancestor and can never be joined. They must be contracted there. Batching
would leave a wire, but there is no supporting capture outside \(v\) with
which it could later be contracted. Pairing it with another class would change
the final partition. Permutations cannot eliminate it. Therefore \(a\) is
absent at the boundary of \(U\). ∎

For a binary node \(v\), write \(L_v=\partial U_\ell\),
\(R_v=\partial U_r\), and \(P_v=\partial U_v\). Define

\[
\begin{aligned}
B_v&=L_v\cap R_v\cap P_v,&
K_v&=(L_v\cap R_v)\setminus P_v,\\
LF_v&=L_v\setminus R_v,&
RF_v&=R_v\setminus L_v.
\end{aligned}
\tag{3}
\]

### Lemma 5. Boundary identities and forced roles

The sets \(B_v,LF_v,RF_v\) are disjoint and

\[
P_v=B_v\mathbin{\dot\cup}LF_v\mathbin{\dot\cup}RF_v.
\tag{4}
\]

Every valid replacement must batch precisely \(B_v\), contract precisely
\(K_v\), and leave precisely \(LF_v,RF_v\) unpaired. Pairings always match
the two occurrences of the same source class.

**Proof.** Formula (2) gives
\(P_v\subseteq L_v\cup R_v\). If an axis is in \(L_v\setminus R_v\)
but not in \(P_v\), its support is inside \(U_v\), is not final, and is not
entirely inside \(U_\ell\). It must therefore touch \(U_r\), which would put
it in \(R_v\), a contradiction. The symmetric argument applies on the right.
This proves (4) and disjointness follows from (3).

At a leaf, (2) gives every physical axis: an axis supported by that capture
alone must be final by Lemma 1. At a unary node the boundary is unchanged.

By Lemma 4, a shared class has two incoming wires that must become one if
retained, and none otherwise. These are exactly batching and contracting.
Leaving the pair unmerged cannot be repaired above this node, as in Lemma 4.
Distinct classes cannot be paired by Lemma 3. An axis appearing in only one
child consequently has to remain unpaired. ∎

Thus axis roles and pairings are determined before any positional attributes
are chosen. Only their permitted orders remain to be realized.

## 4. Attribute-free ordering constraints

Write \(q\preceq w\) for the subsequence relation, \(w|_A\) for filtering
sequence \(w\) to entries in set \(A\), and \(\Vert\) for concatenation.
All layouts below list distinct logical classes. Let \(I_c\) be capture
\(c\)'s physical layout expressed in source classes.

Assign a required sequence \(Q_v\) top-down:

- At the root, \(Q_{\mathrm{root}}=F\).
- At a permutation, pass the empty sequence to its child.
- At a contraction, pass \(Q_v|_{LF_v}\) left and \(Q_v|_{RF_v}\) right.

Induction using (4) shows that each \(Q_v\) has distinct entries drawn from
\(P_v\). In particular, these are subsequence requirements, not complete
intermediate layouts.

Define \(\Phi_S(p)\) as the conjunction of these finite checks:

\[
\begin{array}{ll}
Q_v\preceq I_c, &\text{at a capture }c;\\[2pt]
Q_v=(Q_v|_{B_v})\Vert(Q_v|_{LF_v})\Vert(Q_v|_{RF_v}),
 &\text{at a contraction};\\[2pt]
\mathrm{true}, &\text{at a permutation}.
\end{array}
\tag{5}
\]

The binary check says that the roles in \(Q_v\) are nondecreasing under
\(B<LF<RF\). Neither (2), (3), nor (5) mentions unknown RHS positions.

### Lemma 6. Necessity of ordering constraints

If \(\operatorname{Valid}(S,p,\beta)\), then \(\Phi_S(p)\).

**Proof.** Let \(W_v\) be the actual layout under \(\beta\), interpreted
in source classes by Lemma 3. We prove \(Q_v\preceq W_v\) top-down and
derive every check. At the root, \(W_v=F=Q_v\). At a binary node, Lemma 5
and the operation contract give

\[
W_v=b\Vert(W_\ell|_{LF_v})\Vert(W_r|_{RF_v}),
\]

where \(b\) enumerates \(B_v\) in the chosen batch-list order. Any
subsequence of this concatenation has nondecreasing block roles, giving the
binary check in (5). Its left-free subsequence occurs in \(W_\ell\), and
its right-free subsequence occurs in \(W_r\), giving the induction hypotheses
for the children. No batch-order requirement on either child follows or is
needed: positional batch lists may select child axes in any order.

At a permutation the child's empty requirement is automatically satisfied.
At a leaf \(W_v=I_c\), giving the leaf check. ∎

## 5. Constructing legal positional attributes

### Lemma 7. Sequence primitives

Suppose \(q\) is duplicate-free with entries in a finite set \(A\), and
\(w\) lists every member of \(A\) exactly once, possibly among other
distinct entries. Define

\[
\operatorname{extend}(q,A,w)=q\Vert(w|_{A\setminus\operatorname{set}(q)}).
\tag{6}
\]

This lists exactly \(A\), without duplicates, and contains \(q\) as a
subsequence. If \(a\) is any duplicate-free sequence drawn from a
duplicate-free layout \(W\), define

\[
\operatorname{pos}(W,a)=[\text{the unique position of }a_i\text{ in }W]_i.
\tag{7}
\]

It is an in-range duplicate-free list satisfying
\(\operatorname{gather}(W,\operatorname{pos}(W,a))=a\). For disjoint
sequences' underlying sets, their position lists are disjoint. If \(a\)
enumerates all of \(W\), the position list is a full permutation.

**Proof.** In (6), the two concatenated sets are disjoint and partition \(A\).
In (7), existence follows from membership and uniqueness from distinctness.
Different axes cannot occupy the same position. The remaining statements
follow immediately from these properties. ∎

After all checks (5) pass, construct bottom-up:

1. **Capture:** return \(W_v=I_c\).
2. **Permutation:** construct \(W_u\) with its empty requirement. Set
   \(W_v=\operatorname{extend}(Q_v,P_v,W_u)\) and emit
   \(\pi=\operatorname{pos}(W_u,W_v)\).
3. **Contraction:** construct both child layouts. Choose a fixed enumeration
   \(e_B\) of \(B_v\), and set
   \(b=\operatorname{extend}(Q_v|_{B_v},B_v,e_B)\). Choose any fixed
   enumeration \(k\) of \(K_v\). Emit

\[
\begin{aligned}
b_L&=\operatorname{pos}(W_\ell,b),&
b_R&=\operatorname{pos}(W_r,b),\\
k_L&=\operatorname{pos}(W_\ell,k),&
k_R&=\operatorname{pos}(W_r,k),\\
W_v&=b\Vert(W_\ell|_{LF_v})\Vert(W_r|_{RF_v}).
\end{aligned}
\tag{8}
\]

The concrete implementation uses ascending canonical class IDs for \(e_B\)
and \(k\). The proof permits other enumerations, including the symbolic
constructor's child traversal and source reduction order. The two sides of
each pairing must use the **same** logical sequence.

### Lemma 8. Local legality and order preservation

If (5) holds, construction at every node returns a legal subtree with a layout
listing exactly \(P_v\), and \(Q_v\preceq W_v\).

**Proof.** Induct bottom-up. Leaves have the required set by Lemma 5 and the
required subsequence by their check. At a permutation, parent and child
boundaries agree; Lemma 7 gives the full legal permutation and required order.

At a contraction, both children contain all axes in \(B_v\cup K_v\).
Lemma 7 gives valid position lists. The batch and contract sets are disjoint,
so the selected lists are disjoint in each operand. Corresponding lists have
equal lengths and pair the same class; the class has a common extent from
legal source extraction. This proves all contraction legality conditions.
Its unselected axes are exactly \(LF_v\) and \(RF_v\), giving precisely
the output in (8). Equation (4) shows that output is a duplicate-free
enumeration of \(P_v\).

The batch block contains \(Q_v|_{B_v}\) by Lemma 7. The left-free block
contains \(Q_v|_{LF_v}\) by the left induction hypothesis and filtering;
the symmetric argument holds on the right. The binary check (5) puts these
three subsequences in the required order. Hence \(Q_v\preceq W_v\). ∎

### Lemma 9. Semantic preservation of construction

The constructed root denotes the source network (1).

**Proof.** We prove a stronger subtree invariant. For captures \(U\), put

\[
\mathcal A_U=\{a:\sigma(a)\cap U\ne\varnothing\},\qquad
J_U=\mathcal A_U\setminus\partial U.
\]

At a boundary-coordinate assignment \(t\), the constructed subtree denotes

\[
H_U(X,t)=\sum_{(t_a)_{a\in J_U}}
\prod_{c\in U}X_c[(t_{a(c,i)})_{i=0}^{r_c-1}],
\tag{9}
\]

with boundary coordinates placed in its constructed layout. Each sum uses
the corresponding finite source extent.

For a leaf, \(J_U\) is empty and (9) is its read. A permutation preserves
(9) while rearranging its boundary. At a binary node, the definitions imply
the disjoint union

\[
J_{U_v}=J_{U_\ell}\mathbin{\dot\cup}J_{U_r}\mathbin{\dot\cup}K_v.
\tag{10}
\]

Indeed, a class closed inside one child cannot touch the other. A class closed
at the parent but not inside either child must touch both children and no
outside capture, and cannot be final; these are precisely \(K_v\). All
other shared classes are \(B_v\). The emitted positions identify exactly
these same-class coordinates, batch \(B_v\), and sum \(K_v\). Substituting
the two child instances of (9) into the contraction definition and using finite
distributivity gives (9) for the parent via (10).

At the root, \(U=\mathcal C\), \(\partial U=\bar F\), and
\(J_U=\mathcal A\setminus\bar F\), so (9) is (1). Lemma 8 gives
\(F\preceq W_{\mathrm{root}}\); both sequences enumerate exactly
\(\bar F\), so they are equal. Thus the output order and shape also agree. ∎

For certificate checking, a structural form of the same induction is useful:
at each subtree, the constructed partition is the source partition restricted
to its input ports. Classes touching both children are joined exactly once at
that node; classes inside one child are unchanged; distinct source classes
are never joined. Every non-final class is reduced at the lowest common
ancestor of its support, while final classes remain exposed. Consequently
the complete replayed port partition and ordered output equal the source's.

## 6. Termination and the completeness theorem

### Lemma 10. Termination and exhaustive decision

On every finite in-domain instance, the mathematical algorithm terminates and
returns either constructed attributes or a failed check in (5).

**Proof.** Source replay, boundary derivation, order propagation, and
construction each recurse on strict children of finite trees. All sets and
sequences are drawn from the finite port set or node set. Equivalence closure,
set operations, filtering, position lookup, and order comparisons terminate
on these finite objects. Lemmas 5 and 7 establish the preconditions of every
construction lookup, so successful checks cannot lead to an undefined
position. Each check is a decidable comparison of concrete finite sequences.
Either one fails or every check passes; in the latter case construction
terminates by the preceding argument. No rank cutoff, numerical tensor
evaluation, or search over attribute permutations is used. ∎

### Theorem 1. Complete constructive attribute synthesis

For every schema and legal source configuration in Section 1, the algorithm
terminates, and

\[
\exists\beta\;\operatorname{Valid}(S,p,\beta)
\iff\Phi_S(p)
\iff\operatorname{Infer}(S,p)\text{ returns attributes}.
\]

Every returned assignment is valid. A failed check proves that no valid
assignment exists.

**Proof.** Existence of a valid assignment implies (5) by Lemma 6; its use of
Lemmas 3–5 is what excludes semantically equivalent alternative pairings or
axis roles. Conversely, (5) makes construction legal by Lemma 8 and equivalent
to the source by Lemmas 2 and 9. Thus it supplies a valid witness. Lemma 10
establishes termination and exhaustiveness of the two outcomes. If a check
fails, the contrapositive of Lemma 6 excludes every valid assignment. ∎

The theorem is uniform over all finite ranks. It is a **pointwise** statement
about concrete metadata: \(\forall p\), inference decides
\(\exists\beta\,\forall X\). It does not say that every admitted skeleton
is realizable for every legal \(p\).

## 7. Checkable positive and negative results

### Lemma 11. Two-axis certificates of failure

Whenever (5) fails, a failing node and two adjacent entries of its required
sequence suffice to certify failure.

**Proof.** At a contraction, label required entries by \(0,1,2\) for
\(B,LF,RF\). A finite sequence is nondecreasing if and only if every
adjacent pair is nondecreasing. A failing sequence therefore contains an
adjacent descent. At a leaf, map each required axis to its unique position
in \(I_c\). These positions are distinct. The requirement is a subsequence
exactly when the position sequence is strictly increasing; otherwise some
adjacent pair descends. These are all possible failed checks. ∎

The verifier must rederive the required sequence and forced roles from the
source and RHS skeleton. A pair with asserted role labels alone is not a
certificate. This result concerns certificate size; it does not by itself
establish a rank cutoff for arbitrary guarded schemas.

### Theorem 2. Certificate soundness and acceptance

Consider the following checker on an in-domain instance:

1. Replay source descriptors and recompute (2)–(5), comparing any supplied
   source network and requirement records with those results.
2. For success, check positional legality by independently replaying the
   supplied RHS descriptors. Compare the entire partition, ordered output,
   and supplied intermediate layouts with the replay results.
3. For impossibility, recompute a failed local check and compare the supplied
   descending pair with the selected witness. The implementation selects the
   first witness in node-ID/sequence order.

Then every accepted success has valid attributes, every accepted impossibility
has no valid attributes, and every result produced by the mathematical solver
is accepted by the checker.

**Proof.** Accepted success has legal descriptors and equal networks by replay;
Lemma 3 gives semantic equality and output-shape equality. Accepted failure
establishes a failed instance of (5); Lemma 6 excludes every assignment.

For acceptance of generated results, source extraction and the requirement
definitions agree by construction. Lemma 8 ensures emitted descriptors pass
legality replay. The structural induction following Lemma 9 ensures replay
recovers exactly the source network and recorded layouts. For a negative
result, producer and checker deterministically select the same failed check
and descending pair. The generated negative result has no constructed
attributes or layouts. Thus each generated certificate is accepted. ∎

The success replay is independent of the constructor's attribute choices.
The implementation shares source replay and requirement routines between
producer and checker; this is not implementation diversity or formal
verification of those shared routines.

## 8. Worked positive and negative examples

Let \(A\) have physical axes \([i,j]\), and \(B\) axis \([k]\), with
all extents at least two. The source forms their outer product and permutes
its output to \([j,i,k]\). Consider

```text
dot[auto](transpose[auto](A), B)
```

At the root, \(LF=\{i,j\}\), \(RF=\{k\}\), and \(B=K=\varnothing\).
The required role word for \([j,i,k]\) is \(LF,LF,RF\), which passes.
The left transpose receives \([j,i]\) and sends an empty requirement to
\(A\). Construction emits permutation \([1,0]\) and empty dot lists,
yielding the exact required order. Attributes were constructed after the
attribute-free constraints passed.

For the same source with replacement `dot[auto](A, B)`, the root check still
passes, but the leaf requires \([j,i]\preceq[i,j]\), which fails. The
certificate is the leaf and adjacent pair \((j,i)\). By Theorem 1, no other
batching or contracting attributes can repair it: they would change the
source's port partition or output interface.

For a failure at a binary node, use vector captures and source
\(L[j,i]=A[i]B[j]\), with replacement `dot[auto](A, B)`. The root has
\(LF=\{i\}\), \(RF=\{j\}\), and requires \([j,i]\), whose roles
\(RF,LF\) descend. This remains impossible when both extents are two.

## 9. Conditions, resources, and limits of the theorem

For a metadata-only user condition \(C(p)\), Theorem 1 restricts immediately
to legal in-domain instances satisfying \(C\). To promise that the original
rule applies at **every** source match satisfying \(C\), separately establish

\[
\forall p:\operatorname{Legal}_L(p)\land C(p)
\Longrightarrow D(S,p)\land\Phi_S(p),
\tag{11}
\]

where \(D\) records this proof's domain. Neither the solver nor this proof
automatically establishes (11) for arbitrary symbolic conditions. Silently
adding \(\Phi\) to the user's condition would change the coverage claim.

The mathematical theorem assumes sufficient resources. For the C++ API with
finite `max_work`, the corresponding contract is: for each in-domain instance
there is a finite mathematical work count; if that count and all required
indices fit the implementation's types, an execution with a sufficient budget
and machine memory/stack returns the mathematical decision, subject to
implementation correspondence. A smaller budget may
return `ResourceLimit`, which carries no impossibility claim. Even the default
`size_t` maximum is a finite budget, not mathematical infinity. This proof
does not assert that arbitrary finite ranks fit in one physical machine.

| Restriction | Where it is used |
| --- | --- |
| Independent read-once captures | Wire merging, independently chosen one-hot inputs, and subtree factorization. |
| Extents at least two | Coordinates 0 and 1 distinguish different augmented partitions in Lemma 3. |
| Full permutations and the exact contraction contract | Forced boundaries and the three output blocks. Unary reductions would change the last possible elimination point. |
| Independent unrestricted RHS descriptors | Local batch and permutation choices can be combined without coupling. |
| Exact arithmetic | Finite distributivity and reordering of products and reductions. |
| Legality exactly as specified | Every descriptor produced in Lemma 8 is allowed. |

Outside-domain inputs are not negative instances of this theorem. Empty or
singleton extents may hide semantic differences; repeated captures or
value-side assumptions invalidate the one-hot independence argument. Extra
operators, pins, shared descriptors, and backend restrictions need additional
proofs or a different characterization.

## 10. Correspondence to the current implementation

This table identifies the code implementing each mathematical step. It is a
review map, not a mechanically checked refinement proof.

| Proof component | Implementation | Correspondence to audit |
| --- | --- | --- |
| Section 1 operation/schema domain | [`contract.cc`](../src/core/semantics/synthesis/ordered/contract.cc), `recognize`, `admit` | Recognized group bindings, output, and scalar value have exactly the two operation meanings; no extra legality restriction is accepted. Captures are read-once and RHS descriptors independent. |
| Lemmas 1–2 source extraction | [`evaluate.cc`](../src/inference/evaluate.cc), `Replay` | Validate physical lists and extents, union both batch and contract pairs, retain reduced classes, normalize all interfaces. Canonical representative is the least physical port ID. |
| Lemmas 4–5 | `Analysis::boundaries` | Compute support using capture IDs, retain class identity separately, implement (2) and (3). |
| Lemma 6 | `Analysis::ordering`, `Analysis::obstruction` | Root requirement is the source output; transpose resets it; free-axis filtering propagates it; all local checks are exact. |
| Lemmas 7–9 | `extend`, `positions`, `Analysis::construct` | Same logical lists on both sides of each pairing, total position lookup, legal permutations, preserved child free-axis order. |
| Lemma 10 / Theorem 1 | `inference::evaluate`, `Work` | Every in-domain instance reaches construction or an order inversion when resources suffice. Other statuses are not logical impossibility. |
| Theorem 2 | `inference::check`, `Replay` | Success replays actual attributes; failure rederives the deterministic inversion. Claimed source/requirements/layouts are not trusted. |
| Theorems 3–4 program generation | [`compile.cc`](../src/core/semantics/synthesis/ordered/compile.cc), [`inference.h`](../src/shared/metadata/inference.h) | Emit completed schedules, field adapters, and subtree capture sets without evaluating source metadata. |
| Generated-program execution | [`evaluate.cc`](../src/inference/evaluate.cc), [`rewrite/inference.cc`](../src/rewrite/inference.cc) | Execute the five passes once per match, or replay fully explicit attributes; keep resource/domain failures distinct. |
| Symbolic versions of (3), (5), (8) | [`requirements.cc`](../src/core/semantics/synthesis/ordered/requirements.cc), [`construct.cc`](../src/core/semantics/synthesis/ordered/construct.cc) | Requires exact symbolic source correspondence and faithful metadata primitive evaluation. The proof permits their different deterministic enumeration choices. |

`recognize` inspects axis/value templates rather than operation names. The
semantic adapter follows Section 1: permutation output selects the all-axis
group by a full permutation and reads its operand; contraction group bindings
identify the two selected lists, free groups are ordered complements, output
is batch/left-free/right-free, and the value reduces the product over the
contract group. Swapping scalar product operands is allowed by exact real
arithmetic. Alternative equivalent definition spellings may be rejected;
completeness is for the stated admitted templates, not all possible DSL
definitions. This also assumes the normal frontend has supplied a well-formed
plan and that the declared operation semantics match the intended backend.

The default compiler uses symbolic correspondence, bounded reasoning,
and a fresh general verifier before accepting a candidate. Their completeness
does not follow from Theorem 1. Section 12 proves a uniform generated-program
theorem for a separately specified abstract compiler:

\[
\operatorname{Eval}(\operatorname{Compile}(S),p)
\simeq\operatorname{Infer}(S,p)
\]

for all admitted metadata, including exact applicability checks and defined
evaluation of every emitted operation (comparing decisions and attributes,
without certificate payloads). An explicit `--infer-applicability` mode now
compiles and executes completed inference schedules through the native C++
engine. It uses the same mathematical passes rather than general symbolic
implication search. This implementation is not a mechanically checked
refinement: verification of the actual frontend, instruction semantics, and
compiler/evaluator correspondence remains open. Universal condition coverage
and Rust code-generation/runtime correspondence remain separate. Rust work
remains deferred.

## 11. Validation and paper statement

The dedicated [C++ test](../tests/ordered_synthesis_test.cc) compares the
solver with exhaustive positional-attribute enumeration on small instances
and checks positive and negative certificates. It also exercises equal
extents, multiple pairings, scalar cases, high ranks, internal permutations,
invalid metadata, and resource outcomes. Such tests corroborate the code;
Lemmas 1–10 provide the argument for arbitrary finite ranks. The network-based
enumeration oracle does not independently validate Lemma 3's tensor semantics.

Run the focused check with:

```sh
bazel test //tests:ordered_synthesis_test
```

Suggested paper theorem:

> **Exact feasibility generation and constructive attribute synthesis.** For read-once
> contraction/permutation trees over independently variable tensors, with
> independent unrestricted replacement descriptors, exact real arithmetic,
> and finite input extents at least two, a terminating generator produces a
> feasibility program and an attribute constructor. On every legal source
> configuration, the feasibility program terminates and returns true exactly
> when a legal equivalent replacement attribute assignment exists. Whenever it
> returns true, the constructor terminates with such an assignment. The same
> generated programs handle arbitrary finite ranks.

This is a theorem about the specified algorithm and abstract generator. The
concrete structural implementation is supported by tests and code review;
its native generated-program integration is covered by regression tests. Do not describe the
whole TEPL compiler as formally verified or complete on the basis of this hand
proof. Theorem 2 additionally supplies certificates for concrete outcomes.

Related repository documents: [refactoring and proof plan](complete_attribute_synthesis_plan.md),
[general verification soundness](soundness_proof.md), and the earlier
[ordered-contraction research draft](../examples/research/ordered_contractions.md).

## 12. Generation of feasibility and attribute programs

This section supplies the fourth proof step: translation correctness. The input
is the abstract schema of Section 1, with fixed operation contracts and source
descriptor references. No TEPL-specific syntax is needed. The generator below
is an independent mathematical compiler specification. The native inferred-
applicability path now implements the same passes using specialized, validated
operation schedules; the default compilation path retains its earlier proof
behavior. This section does not formally verify either C++ path.

### 12.1 Metadata-program language and semantics

A program operates on a store of finite sequences, finite sets, maps, finite
partitions, natural numbers, and Booleans. It has assignment, sequencing,
conditionals, returns, and `for each` loops over finite snapshots of collections.
There is no unbounded search instruction. Expressions include metadata reads,
indexed selection, concatenation, filtering, set operations, equality, extent
comparison, sequence length, and the operations defined below.

- `new_port(c,i)` records the named physical input port and its extent.
- `merge(a,b)` replaces the two blocks containing its arguments in a finite
  partition with their union; other blocks are unchanged.
- `representative(a)` returns the least numbered port in its block.
- `positions(W,q)` implements (7), defined when every entry of `q` occurs
  uniquely in `W`.
- `extend(q,A,w)` implements (6).
- `subsequence(q,w)` returns the ordinary finite subsequence test.

Port numbering uses increasing capture ID and then physical position. Set
enumerations use increasing canonical port IDs. These conventions fix the
otherwise arbitrary choices made by the mathematical constructor.

The operations are given their mathematical meanings, not postulated to solve
attribute synthesis. They can all be expanded into finite loops: merging scans
the finite partition, filtering scans its input, positions scans the layout,
and subsequence checking advances through a finite sequence. Metadata reads
are defined on the source domain of Section 1. The proof below establishes
preconditions of partial operations such as selection and position lookup.
Proving that a particular C++ primitive implements these definitions remains
an implementation refinement obligation.

### 12.2 A syntax-directed generator

The schema fixes the source and target trees, capture identities, and descriptor
field references. The generator computes finite traversal schedules from these
trees and emits the following blocks. All loops over axes or attribute lists
are emitted as loops; their bounds are read at execution time.

**Block A: source analysis.** For each capture, emit a loop that creates one
port per physical axis and stores its ordered input layout. Initialize the
partition to singleton ports. Visit source operations in postorder:

- For a permutation, emit selection of the child's layout using the referenced
  source permutation field.
- For a contraction, emit loops over the two batch lists and two contract
  lists, merging corresponding selected child ports. Store the output as the
  left batch selection followed by the two ordered unselected child sequences.

After all source operations, normalize every port and recorded layout through
`representative`. Retain the full partition, including reduced classes. Store
the source root layout as `F`. Compute each class's capture support by scanning
its member ports.

**Block B: RHS boundaries.** Compute each RHS subtree's capture set statically
from the tree. In RHS postorder, emit a loop over source classes implementing
(2). At binary nodes, emit the intersections and differences in (3). Nothing
in this block requires an RHS descriptor value.

**Block C: required orders and feasibility.** Store `Q[root] = F`. In RHS
preorder, emit the child assignments of Section 4. Initialize `ok = true`.
For every leaf, conjoin `subsequence(Q[v], I[c])` into `ok`. For every
contraction, conjoin the test

```text
Q[v] == concat(filter(Q[v], B[v]),
               filter(Q[v], LF[v]),
               filter(Q[v], RF[v]))
```

Permutations add no check. A conjunction over no checks is true.

**Block D: guarded construction.** Emit `if !ok: return NotApplicable` before
any construction block. Then, in RHS postorder, emit exactly (6)–(8): leaf
layouts, permutation extension and positional conversion, or paired batch and
contract lists and ordered free blocks. Return `Solved(attributes)`.

Let `A; B; C` denote the shared analysis prefix. Define the generated predicate
program and guarded constructor program by

\[
\begin{aligned}
\operatorname{Generate}(S)&=(\Phi_S,f_S),\\
\Phi_S&=A;B;C;\operatorname{return}\ ok,\\
f_S&=A;B;C;D.
\end{aligned}
\]

Here \(\Phi_S(p)\) abbreviates evaluation of the predicate program. The
constructor returns either `NotApplicable` or `Solved(beta)`; the attribute
function on feasible metadata is the payload of its successful result.
Recomputing the prefix makes both programs self-contained. An implementation
may share its store within a single match, provided the source metadata is
unchanged and the store is the one produced by that prefix.

### Lemma 12. Generation terminates and is uniform in rank

For every schema in Section 1, the specified generator terminates and produces
finite programs without knowing any input rank or source descriptor value.

**Proof.** Capture-set computation and traversal scheduling recurse on strict
subtrees. Each node emits a fixed finite template, containing loops for
metadata-dependent sequences. No generation step unfolds such a loop according
to an unknown rank, enumerates descriptor assignments, or asks whether a
symbolic condition is valid. Thus a finite tree produces a finite program.
The same program accepts metadata lists of every finite length. ∎

### Lemma 13. Correctness of generated analysis and checks

For every legal in-domain metadata configuration, Blocks A–C terminate and
produce the same source network, boundaries, required sequences, and Boolean
as Sections 2–4 of the concrete algorithm.

**Proof.** For Block A, induction on the capture/axis loops gives exactly the
physical port set and initial layouts. Induction on the source postorder gives
the replay invariant: after a processed subtree, its selected wires, accumulated
identifications, and stored output equal source replay at that subtree. A
permutation selects the same positions. For a binary node, an induction on
the pairing loop shows that after its first \(j\) pairs, exactly those
\(j\) equalities have been added. The output concatenation has the operation's
declared order. Source legality guarantees all selections are defined.
Normalizing representatives after all merges gives the same complete network
and canonical labels as Section 2. The support scan computes precisely
\(\sigma\).

For Block B, static subtree capture sets are correct by tree induction.
During a boundary loop, the stored set consists exactly of already visited
classes satisfying (2). At loop completion this is \(\partial U\). The
subsequent set operations therefore compute exactly (3).

For Block C, induction on preorder establishes that each assigned `Q[v]`
equals Section 4's requirement: the root starts with `F`, permutations reset
the child to empty, and binary nodes filter by the proved free-axis sets.
After each local check, `ok` is the conjunction of the checks already visited.
The primitive expression meanings make each check precisely its instance of
(5). Consequently the final Boolean equals the mathematical \(\Phi_S(p)\).

Each loop traverses a finite snapshot and each schedule is finite. All reads
are from initialized entries: postorder initializes children before parents,
and preorder initializes parent requirements before children. Hence the blocks
terminate with defined values. ∎

### Lemma 14. Correctness of generated construction

If the generated check returns true, Block D terminates with the same layouts
and attributes as the mathematical constructor using the same enumeration
conventions. If the check returns false, Block D returns `NotApplicable`
without evaluating construction operations.

**Proof.** The false branch follows from the conditional's execution semantics.
For the true branch, Lemma 13 gives the exact mathematical requirements and
establishes (5). Induct over RHS postorder. A leaf stores the same input
layout. At a permutation, the induction hypothesis supplies the child's exact
layout; `extend` and `positions` therefore compute the sequence and permutation
of Section 5. At a contraction, both child layouts agree, and the fixed
enumerations of \(B_v\) and \(K_v\) agree. Every emitted expression in (8)
therefore returns the same list or layout. Lemmas 7–8 ensure every positional
lookup is defined and every emitted descriptor is legal. Independent RHS
descriptors ensure that storing one node's fields cannot overwrite another's.
All sequences and schedules are finite, establishing termination. ∎

### Theorem 3. Exact feasibility generation and constructive synthesis

For every schema \(S\) in Section 1, generation terminates with finite
metadata programs \((\Phi_S,f_S)\). For every legal source configuration
\(p\) in that domain, both programs terminate and

\[
\operatorname{Eval}(\Phi_S,p)=\mathrm{true}
\quad\Longleftrightarrow\quad
\exists\beta\;\operatorname{Valid}(S,p,\beta).
\]

Moreover,

\[
\operatorname{Eval}(\Phi_S,p)=\mathrm{true}
\Longrightarrow
\operatorname{Eval}(f_S,p)=\operatorname{Solved}(\beta)
\land\operatorname{Valid}(S,p,\beta),
\]

and a false check yields `NotApplicable` and rules out every valid assignment.
These properties hold for arbitrary finite ranks using the same generated
programs for each fixed schema.

**Proof.** Lemma 12 establishes generation and rank uniformity. Lemma 13 equates
the generated Boolean with the concrete feasibility predicate; Theorem 1
equates that predicate with attribute existence. Lemma 14 equates successful
generated construction with the mathematical constructor, whose correctness
is established by Lemmas 8–9. Its false branch and Theorem 1 give exact
inapplicability. The same lemmas establish termination. ∎

Equivalently, after projecting away certificate payloads and fixing identical
enumeration conventions, executing the guarded generated program yields the
same decision and attributes as the mathematical concrete solver. The solver's
`Impossible` corresponds to the program's `NotApplicable` on this domain.

Section 13 establishes preservation of the Section 1 operation contracts and
attribute choices for a specified translation of a restricted declarative TEPL
fragment. It remains to establish that the actual frontend, compiler, and
evaluator implement that translation and the abstract generation scheme.
The theorem assumes valid metadata in its domain; a production entry point
must check that domain and distinguish unsupported inputs and resource limits
from the false result of the exact feasibility predicate. An explicit inferred
guard may combine an independently evaluated user condition \(C(p)\) with
this predicate. No completeness claim about proving arbitrary implications
from \(C\) is required.

## 13. Supported TEPL declarations suffice for exact checker generation

This section answers the language-level question: what information must a
rule expose so that an exact feasibility checker can be generated? Define a
restricted declarative language, give its translation, and prove that the
translation preserves both tensor meanings and the available attribute
assignments. This strengthens the abstract result without claiming that the
current C++ compiler has already implemented the resulting generator.

### 13.1 Declarative fragment \(\mathcal F_T\)

Let \(T\) be a well-formed TEPL rule with resolved operation declarations and
named references to captures and descriptor fields. The semantic declaration
of every operation must match one of the following templates. These are
schematic descriptions of the checked semantic representation, not new parser
syntax. Operation, group, and field names may be renamed consistently.

| Template | Required semantic information |
| --- | --- |
| Permutation | One operand; one position-list field \(\pi\); an axis group covering all operand axes; output `permute(group, pi)`; ordinary operand value access with the corresponding coordinate transport. Legality is exactly full-permutation legality. |
| Contraction | Two operands; four distinct position-list fields \(b_L,b_R,k_L,k_R\); paired groups selecting batches and contractions from the respective operands; free groups given by ordered complements of those selections; output `batch ++ left-free ++ right-free`; value equal to the product of the operand reads, summed over the contraction group. Legality is exactly Section 1.1. |

For syntactic admission, allow association/flattening of concatenations,
reordering of the excluded lists inside an ordered complement, consistent
renaming, and reversal of the two scalar product factors. These transformations
have their ordinary exact semantics. No other normalization is needed to
recognize the templates. Permit only the ordinary list-validity and full-rank
checks of the permutation contract; reject extra assertions. The contraction
template has the stated group-legality checks and no additional assertions.
An alternative equivalent definition spelling may be outside this initial
fragment. No general semantic-equivalence oracle is part of admission.

The two expression trees must satisfy Section 1.3's capture and descriptor
restrictions. Source field references name captured metadata; RHS descriptors
have exactly the template fields and every field is unrestricted `auto`.
Value assumptions, such as symmetry of an input, are excluded. The declaration
semantics is the ideal axis/value semantics used here; additional backend
dtype or precision restrictions are not included in this theorem.

Optional conditions \(C\) depend only on source metadata and are evaluated
separately. The checker-generation theorem below does not analyze or evaluate
\(C\), and does not require proving a universally quantified implication
from it. When composing a complete match procedure, condition evaluation must
be defined and terminating on the matches where it is invoked.

This definition distinguishes the information required for inference:

| Information in a declaration or match | Use in the generated analysis |
| --- | --- |
| Named captures and physical axis positions | Preserve input-port identities; equal extents alone do not identify axes. |
| Paired group bindings and source field references | Recover which ports are identified at the actual match. |
| Reduction group and retained groups | Distinguish reduced classes from exposed classes. |
| Ordered output expressions and permutation references | Recover the source output sequence and RHS ordering restrictions. |
| Scalar product/reduction value expression | Justify exact network denotation and equivalence for all input values. |
| Unrestricted independent RHS fields | Ensure that all choices made by the constructive algorithm can be expressed. |

The logical partition need not be known at compilation: references specify
how to compute it from runtime metadata. The claim is sufficiency of this
information, not necessity of TEPL's particular surface notation.

### 13.2 Translation and attribute adapters

Define a translation \(\tau\) recursively. Preserve each capture and operand
connection. Replace a matched permutation template by `Perm`, and a matched
contraction template by `Dot`. Read the template's semantic field bindings to
record, for each operation \(v\), a field-order adapter \(\rho_v\): the
permutation field, or the ordered tuple \((b_L,b_R,k_L,k_R)\). Because the
fields are distinct and exhaust the descriptor, this adapter is a bijection
between declaration-field tuples and canonical tuples.

The translation returns an abstract schema \(S=\tau(T)\) and two adapters:

- \(\alpha_T(p)\) preserves capture shapes and supplies source fields to their
  canonical roles through the recorded references. A shared source reference
  remains shared; reading it at two occurrences does not create two choices.
- \(h_T(\beta)\) rearranges each RHS descriptor into canonical role order.
  Since every RHS descriptor is independent, the product of the local
  bijections is a bijection on complete RHS assignments. Its inverse writes
  constructed fields back to the declaration's field order.

These adapters perform finite named-reference accesses and field rearrangements.
They neither enumerate ranks nor change the lists stored in the fields. The
generator emits these accesses for execution on \(p\); it does not require
source attribute values while compiling \(T\).

If two source occurrences interpret one descriptor in different canonical
field orders, use separate canonical views computed from that same descriptor.
These views are determined by \(\alpha_T\), not independent metadata inputs.
Define \(D_T(p)=D_{\tau(T)}(\alpha_T(p))\); since shapes are preserved, it is
exactly the physical-extent condition of Section 1.3.

### Lemma 15. Decidable admission and total translation

Membership in \(\mathcal F_T\) is decidable by finite structural checks. For
every member \(T\), translation terminates with \(\tau(T)\in\mathcal F_0\)
and the adapters above.

**Proof.** Normalize only the finite syntactic variations explicitly permitted
above, then compare each finite declaration against the two templates. Count
capture occurrences and check descriptor ownership, field distinctness,
operand arities, and source/RHS references. These checks do not inspect input
values or ask whether attributes exist. Translation visits strict subtrees,
and each operation emits one canonical node and a finite field map. The
templates give the Section 1.1 contracts; structural checks give Section
1.3's remaining requirements. Thus the output lies in \(\mathcal F_0\).
Neither translation nor admission depends on the outcome of feasibility. ∎

### Lemma 16. Operation, legality, and assignment preservation

Let \(\operatorname{Valid}_T\) denote validity under these declared TEPL
axis/value semantics. For every admitted \(T\), legal in-domain \(p\), and
complete RHS assignment \(\beta\),

\[
\operatorname{Valid}_T(T,p,\beta)
\iff
\operatorname{Valid}\bigl(\tau(T),\alpha_T(p),h_T(\beta)\bigr).
\]

Consequently,

\[
\exists\beta\;\operatorname{Valid}_T(T,p,\beta)
\iff
\exists\gamma\;\operatorname{Valid}\bigl(\tau(T),\alpha_T(p),\gamma\bigr).
\]

**Proof.** First inspect each template. The permutation adapter selects exactly
the declared positions from the full input interface; its legality, ordered
shape, and coordinate transport equal those of `Perm`. In the contraction
template, the four fields select precisely the two paired groups; their
complements select precisely the free axes in operand order. The output
expression gives the canonical three blocks. The value expression is exactly
the sum of products in Section 1.1. Renaming changes no meaning; reversing
scalar product factors preserves exact real multiplication. The explicit
legality restrictions make positional legality equivalent in both directions.

Induction over each tree now preserves legality, ordered shapes, and values
at every node for every input tensor assignment. Source adapters retain
actual metadata values and all sharing. RHS adapters neither fix fields nor
identify distinct unknowns. This establishes the first equivalence, including
the universally quantified equality of source and replacement values.

For the second equivalence, map a declaration assignment through \(h_T\)
in the forward direction and an abstract assignment through \(h_T^{-1}\)
in the reverse direction. The inverse exists because descriptors contain
exactly the template fields and are independent. Hence translation loses no
possible solution and introduces no unattainable solution. ∎

Preserving only completed-expression values would not suffice: a translation
could preserve every retained candidate while silently discarding another
valid assignment. The bijection is what excludes that source of false negatives.

### Theorem 4. Exact checker generation from declared TEPL information

There is a terminating generation procedure for every \(T\in\mathcal F_T\)
that returns a feasibility program \(\Psi_T\) and a guarded attribute program
\(g_T\). For every source metadata configuration \(p\) satisfying source
legality and \(D_T(p)\), their evaluations terminate, and

\[
\boxed{
\operatorname{Eval}(\Psi_T,p)=\mathrm{true}
\iff
\exists\beta\;\operatorname{Valid}_T(T,p,\beta).
}
\]

If the result is true, \(g_T\) returns a valid declaration-level assignment;
if false, \(g_T\) returns `NotApplicable` and no valid assignment exists.
One generated pair handles every finite rank in the domain for each fixed rule.

**Construction.** Translate \(S=\tau(T)\) and generate
\((\Phi_S,f_S)=\operatorname{Generate}(S)\) as in Section 12. Emit adapters
around these programs:

\[
\begin{aligned}
\Psi_T(p)&=\Phi_S(\alpha_T(p)),\\
g_T(p)&=
\begin{cases}
\operatorname{Solved}(h_T^{-1}(\gamma)),
  & f_S(\alpha_T(p))=\operatorname{Solved}(\gamma),\\
\operatorname{NotApplicable},
  & f_S(\alpha_T(p))=\operatorname{NotApplicable}.
\end{cases}
\end{aligned}
\]

These equations specify composition of generated programs, not compile-time
evaluation on unknown metadata. Both programs use only source metadata and
the fixed declaration/tree information.

**Proof.** Lemma 15 and Theorem 3 establish terminating generation. Lemma 16
makes \(\alpha_T(p)\) legal and in-domain for the abstract schema, so
Theorem 3 applies. Its equivalence and Lemma 16 give

\[
\operatorname{Eval}(\Psi_T,p)=\mathrm{true}
\iff\exists\gamma\;\operatorname{Valid}(S,\alpha_T(p),\gamma)
\iff\exists\beta\;\operatorname{Valid}_T(T,p,\beta).
\]

On success Theorem 3 returns valid \(\gamma\); Lemma 16 makes
\(h_T^{-1}(\gamma)\) valid in the declarations. On failure the equivalence
excludes every declaration-level assignment. Finite adapter operations and
Theorem 3 give evaluation termination. The translation and emitted templates
depend on syntax and references, while ranks determine only finite runtime
list lengths. This establishes rank uniformity. ∎

### 13.3 Meaning of acceptance, conditions, and current implementation

There are three different decisions:

| Decision | Meaning |
| --- | --- |
| \(T\in\mathcal F_T\), decided at compilation | The declarations contain the supported information, so the specified generator can produce a checker and constructor. It does not assert feasibility for all matches. |
| \(\operatorname{Legal}_L(p)\land D_T(p)\), checked or guaranteed at matching | The actual metadata is in the theorem's domain. Failure is not the checker's mathematical claim of infeasibility. |
| \(\Psi_T(p)\), evaluated on an in-domain match | True exactly when a legal equivalent RHS assignment exists. |

For example, consider the admitted schematic rule

```text
transpose[@p](dot[@d](A, B)) => dot[auto](A, B)
```

At two-vector matches with empty batch/contract lists in `@d`, an identity
source permutation is feasible; swapping the two source output axes is
infeasible. In the latter case the target must retain both free axes, and
their declared output order cannot be reversed. This remains true when the
vector lengths are equal. Both metadata cases belong to the same admitted
rule and domain; Theorem 4 requires the generated checker to distinguish them.

An explicitly inferred applicability condition can combine \(C(p)\) with
\(\Psi_T(p)\) after establishing the domain. Exactness concerns
\(\Psi_T\), conditional on that domain. A false user condition, unsupported
metadata, or resource exhaustion must not be reported as proof that RHS
attributes do not exist. Proving that every match satisfying arbitrary \(C\)
is feasible remains separate future work.

The existing `recognize`/`admit` code provides much of the structural template
machinery, but \(\mathcal F_T\) is the mathematical definition above, not an
assumption that these routines or the frontend have been formally verified.
The default compiler still proves obligations under declared conditions.
The explicit inferred-applicability mode now compiles role adapters and
postorder schedules, precomputes capture sets, and evaluates feasibility before
constructing descriptors in the native engine. Fully explicit RHS descriptors
use replay validation, justified by Lemma 3 rather than existential synthesis.
Mixed explicit/automatic synthesis remains outside the implementation's admitted
fragment. The native tests exercise these paths and altered-program rejection;
formally verifying the frontend and primitive/compiler/evaluator correspondence
remains open. The theorem is about the specified mathematical translation and
generator, not a claim of formal verification of the current compiler.
