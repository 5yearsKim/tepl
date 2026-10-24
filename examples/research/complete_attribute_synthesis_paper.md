# Exact Feasibility and Constructive Attribute Synthesis

**Fragment.** Operations are generalized contraction, full permutation,
unary summation, and elementwise addition. A contraction pairs equal-extent
batch and reduction axes using in-range, duplicate-free position lists,
disjoint on each operand. It outputs batches in list order, then left-free
and right-free axes in operand order. Unary summation removes an in-range,
duplicate-free selection of axes and preserves survivor order.
Addition requires identical ordered shapes.
These are the complete contracts, with exact-real sum-of-products semantics.

Let \(\Sigma\) specify source and RHS trees, declared axis semantics, and
attribute references. Linearity expands the source to \(L=\sum_{j\in J}L_j\).
Require an explicit RHS sum \(R=\sum_{j\in J}R_j\), where each pair
\(L_j,R_j\) uses the same nonempty input set \(U_j\), once per input, through
contraction, permutation, and unary summation. The sets \(U_j\) are distinct
but may overlap. Every RHS descriptor is an independent unrestricted unknown;
no other operators or value assumptions are admitted. A single term includes
the original non-additive fragment.

Metadata \(p\) supplies shapes and legal source attributes. Inputs vary
independently; ranks are arbitrary and finite, physical extents are at least
two, and scalars are allowed. Let \(\operatorname{Valid}_\Sigma(p,\beta)\)
mean that attributes \(\beta\) make the RHS legal with the source's ordered
output shape and equal to the source for every input value.

**Theorem (Exact feasibility generation).** A terminating generator produces
metadata programs \((\Phi_\Sigma,f_\Sigma)\) such that, for every legal
in-domain \(p\),

\[
\boxed{
\begin{aligned}
\Phi_\Sigma(p)&\iff\exists\beta\;\operatorname{Valid}_\Sigma(p,\beta),\\
\Phi_\Sigma(p)&\Rightarrow
\operatorname{Valid}_\Sigma(p,f_\Sigma(p)).
\end{aligned}
}
\tag{1}
\]

Both programs terminate; the constructor returns NotApplicable on failure.
The same programs handle all finite ranks.

**Generated constraints.** Analyze each source term separately. Source
pairings identify physical input-axis ports; retain these classes even after
reduction. Let \(I_c\) be input \(c\)'s ordered class sequence, \(F\) the
ordered output, and \(\sigma(a)\) the inputs containing class \(a\).
For a RHS node \(v\) with input set \(V_v\), the mandatory axes are

\[
M_v=\{a:\sigma(a)\cap V_v\ne\varnothing
 \land(a\in\operatorname{set}(F)\lor\sigma(a)\nsubseteq V_v)\}.
\tag{2}
\]

Compute canonical live sets \(P_v\) bottom-up. An input keeps
\(P_v=\operatorname{set}(I_c)\); a permutation keeps its child's set;
a unary sum sets \(P_v=P_{\mathrm{child}}\cap M_v\).
At a contraction, set

\[
\begin{aligned}
B_v&=P_\ell\cap P_r\cap M_v,&
K_v&=(P_\ell\cap P_r)\setminus M_v,\\
LF_v&=P_\ell\setminus P_r,&
RF_v&=P_r\setminus P_\ell,\\
P_v&=B_v\cup LF_v\cup RF_v.&&
\end{aligned}
\tag{3}
\]

Thus completed non-output axes disappear at the first operation that can
remove them. Require \(P_{\mathrm{root}}=\operatorname{set}(F)\).

Propagate a required subsequence \(Q\), starting with \(F\).
A contraction passes \(Q|_{LF}\) and \(Q|_{RF}\) to its children;
a unary sum passes \(Q\) unchanged; a permutation passes the empty sequence.
Here \(Q|_A\) filters by set \(A\), \(\Vert\) concatenates, and \(\preceq\)
denotes subsequence. Check

\[
\begin{cases}
Q\preceq I_c,&\text{at an input }c,\\
Q=(Q|_B)\Vert(Q|_{LF})\Vert(Q|_{RF}),&\text{at a contraction}.
\end{cases}
\tag{4}
\]

Let \(\phi_j\) conjoin the root-set and order checks for term \(j\).
Then \(\Phi_\Sigma=\bigwedge_j\phi_j\), using only source metadata and the
fixed RHS structure.

**Proof.**
*Separate terms.* Scale each input \(X_c\) by an independent real \(z_c\).
Term \(j\) is multiplied by \(\prod_{c\in U_j}z_c\). Distinct input sets give
distinct monomials, so coefficient uniqueness makes equality of the sums
equivalent to equality of every paired term. Source expansion preserves
legality and shapes by the operation contracts.

*Recover axis relationships.* Each read-once term denotes a sum of products,
with one factor per input and every index class observed by an input port.
Equal partitions and ordered output attachments therefore give equal tensors.
Conversely, attach a named port to each output position. If two partitions
differ, assign coordinates 0 and 1 to classes separated in one but identified
in the other. One-hot inputs at the resulting tuples produce 1 versus 0:
every reduction coordinate is fixed by an input port. Hence equality for all
inputs forces the same partition and output order.

*Normalize axis lifetimes.* Every valid replacement retains (2), but may
keep additional axes until a later unary sum. Project its layout at each
node onto \(P_v\). This projection is realizable: a unary sum can remove
completed non-output axes, and a dot can contract a completed shared class
instead of batching it. Every surviving class keeps its batch/free role;
filtering preserves the relative order of survivors. Permutations can realize
any projected order, while input layouts are unchanged. Induction therefore
gives an equivalent realization with exactly the canonical live sets.
Its root must have precisely \(F\), and its block and input orders satisfy
(4). Thus every valid assignment passes the generated checks.

*Construct attributes.* Build layouts bottom-up. Inputs retain their order;
a permutation places \(Q\) first. At a dot, choose a batch order beginning
with \(Q|_B\), any common order of \(K\), and translate those lists into
positions in the child layouts. Append child free axes in their existing
order. At a unary sum, reduce the positions of
\(P_{\mathrm{child}}\setminus P_v\).
Inductively each layout contains exactly \(P_v\) and contains \(Q\) as a
subsequence, by (4). All lookups are defined, paired extents match, and
the descriptors are legal. Only occurrences of the same source class merge;
a class disappears only after its supporting ports have joined.
At the root the layout is exactly \(F\), establishing term equality.

Independent descriptors let the term assignments combine; their output
shapes agree, so the full sum is legal and equal. Finally, finite source
expansion and emitted tree traversals implement (2)--(4) and the construction
using finite metadata operations. Rank affects loop bounds, not program
structure. This proves (1), termination, and uniform generation. \(\square\)

For LoRA, \(X(W+AB)\to XW+(XA)B\), the distinct input sets are
\(\{X,W\}\) and \(\{X,A,B\}\); only the latter needs reassociation analysis.
Unary summation additionally permits
\(\sum_{i,j}(AB)_{ij}=\sum_k(\sum_i A_{ik})(\sum_j B_{kj})\),
with reduction and contraction attributes inferred together.

**Scope.** Additional conditions give mathematical-solver applicability
\(C(p)\land\Phi_\Sigma(p)\). The theorem concerns the specified generator,
not C++ implementation verification. The
[unified proof](complete_attribute_synthesis_proof.md) gives the full argument.

**Practical TEPL integration.** The complete generator is a specialized reasoning
procedure. TEPL's general synthesis path may support additional cases without
complete search, but each accepted semantic rule must be verified for every
legal match satisfying its written conditions. Constructor definedness, RHS
legality, ordered shape and dtype, and equivalence under the selected arithmetic
policy are proof obligations, never hidden applicability filters. Singleton and
zero extents, mixed descriptors, and repeated-capture identity may be supported
through direct verification without broadening this theorem's completeness
claim. Unknown proofs remain failures to establish the rule, not impossibility
results. A bounded restriction recommender verifies each suggestion afresh and
requires a legal metadata witness; users add reviewed restrictions explicitly.
The proof's Section 10 describes the implemented acceptance contract and the
separate implementation-correspondence obligation.
