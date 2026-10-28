# Exact Feasibility and Constructive Axis-Attribute Synthesis

**Revised October 28, 2026.** The general structural extension is stated at the
end of this note and proved in Section 12 of the unified proof. It admits fresh
positive axes and metadata weights while inferring only marked axis fields.
The earlier specialization below remains a corollary with its original domain.

**Signed-axis specialization (Sections 1--11).** Operations are generalized contraction,
full permutation, whole-axis reversal, unary summation, aligned weighted
scalar products, and separated elementwise addition. Tensors have exact real
entries, arbitrary finite ranks, and extents at least two; scalars are allowed.
Broadcast, slice, and reshape are excluded.

Contraction pairs equal-extent batch and reduction axes using in-range,
duplicate-free lists, disjoint on each operand. Its output order is batches,
left-free axes, then right-free axes. Permutation reorders whole axes. Reversal
selects any subset of axes and reads coordinate \(n-1-i\) there. Unary sum
removes selected axes and preserves survivor order. An admitted scalar operation
has a fixed positive finite arity and

\[
h(x_1,\ldots,x_k)=q_h\prod_i x_i,
\qquad q_h\in\mathbb Q\setminus\{0\}.
\tag{S}
\]

Its tensor operation has no attributes, requires identical ordered operand
shapes, and applies the function positionwise without implicit broadcasting.
The rational coefficient is fixed in the declared semantics, not an unknown
field or tensor capture. This single condition includes negation, fixed nonzero
rational scaling, and aligned products of any positive finite arity. A
structural adapter recognizes the rational coefficient and product of all
aligned operand reads; it does not decide equivalence of arbitrary scalar code.
Addition requires identical shapes and retains its restricted role below.

**Attribute boundary of this specialization.** Descriptors select, order, or pair whole axes.
Operations with extent, coordinate-offset/stride, padding, or target-shape
fields are excluded on either side, even if those fields are supplied.
Axis positions identifying which axes to manipulate are permitted; coordinate
indices identifying elements within an axis are excluded. Input extents remain
metadata for legality and fixed reversal semantics. Attribute-free operations
and axis descriptors must still match the exact contracts above; this boundary
does not admit every operation taking an axis parameter.

Let \(\Sigma\) specify source and fixed RHS trees. Expand the source by
linearity/bilinearity to \(L=\sum_{j\in J}L_j\). Require an actual RHS sum
\(R=\sum_{j\in J}R_j\), with each pair using the same nonempty input set
\(U_j\), once per input, through the non-additive operations above. The sets
\(U_j\) are distinct but may overlap. All RHS descriptors are independent
unrestricted unknowns within their operation contracts. There are no unknown
extent fields, shared holes, standalone constant tensors, or input-value
assumptions. Fixed coefficients in (S) are part of operation semantics. Synthesis
changes attributes of the fixed operation tree.

Metadata \(p\) supplies input shapes and legal source attributes. Define
\(\operatorname{Valid}^\times_\Sigma(p,\beta)\) to require a legal RHS,
the source's ordered output shape, and equality for every input tensor value.
The quantifier order is \(\exists\beta\,\forall X\).

**Theorem (Complete constructive axis-attribute synthesis).** A terminating
generator produces metadata programs \((\Phi^\times_\Sigma,f^\times_\Sigma)\)
such that, for every admitted specification and legal in-domain metadata,

\[
\boxed{
\begin{aligned}
\Phi^\times_\Sigma(p)&\iff\exists\beta\;
                     \operatorname{Valid}^\times_\Sigma(p,\beta),\\
\Phi^\times_\Sigma(p)&\Rightarrow
                     \operatorname{Valid}^\times_\Sigma(p,f^\times_\Sigma(p)).
\end{aligned}
}
\tag{1}
\]

The same programs handle all finite ranks. The constructor returns
NotApplicable on failure. With aligned scalar operations the procedure uses
finite semantic-state dynamic programming, potentially with combinatorial
cost. The previous direct constructor remains a specialization for the
fragment without the added scalar operations.

**Normalize scalar coefficients.** For each expanded non-additive component
\(E\), compute \(c_E=\prod_h q_h\) once, and erase those coefficients
to obtain its unweighted network \(N_E\). Interpret an aligned product as
fixed full-batch contractions, preserving every axis in position order; these
fixed lists do not introduce new attribute holes. Unary scaling leaves the
network unchanged. Thus \(E=c_E\mathcal T_{N_E}\).

Lemma 1's signed exactness extends immediately: different networks have a
one-hot witness giving a nonzero coefficient versus zero; equal networks have
a satisfying one-hot input exposing their coefficients. Hence universal
equality is exactly equality of the networks and of \(c_E\). The nonzero
coefficient assumption is essential to this argument.

**Construct and prove completeness.** At each RHS node retain all legally
reachable canonical unweighted networks, including ordered shape, with one
realizing descriptor witness per network. Consider all axis orders at a
permutation, all subsets at reversal/sum, and all legal pairings, roles, and
batch orders at contraction. An aligned scalar node has only its fixed
positionwise alignment, requiring all child shapes to agree. Replay these
connections as in Section 2; reduced physical input ports remain recorded.

Induction shows each table contains exactly the realizable networks. Generated
entries have legal witnesses. Conversely, any legal assignment's child states
occur by induction and its local choice is considered. Equal child networks
have equal shapes and normalized values; their scalar coefficients are fixed
by the subtree. Retaining one witness therefore preserves parent legality and
values. Independent holes allow witnesses to combine.

Accept each component exactly when its coefficient matches the source and its
root table contains the source network; recover the stored witnesses. The
exactness argument proves component completeness. Multilinearity preserves
Section 5's separation of distinct input sets, so the witnesses combine to
prove (1). The actual RHS must already be separated; copying unknowns through
an RHS expansion and same-input-set cancellation remain excluded.

Each finite-rank child has finitely many local choices. Induction gives finite
tables and witnesses, and syntax-directed generation emits these traversals,
root checks, and witness recovery. All coefficients are exact rational constants
computed during generation. This proves termination uniformly across finite
ranks without reading tensor values or enumerating coordinate grids. The search
may be combinatorial. The direct core constructor remains a specialization;
fixed pointwise alignment prevents extending its greedy elimination unchanged.

**Scope and implementation.** This is completeness for the specified
mathematical procedure. Broadcast, slice, reshape, shared holes, repeated
inputs within a component, singleton/zero extents, zero scaling, standalone
constants, scalar functions outside (S),
and strict floating-point reassociation remain outside the theorem. It neither
formalizes the compiler nor establishes a new XLA coverage count. Extra source
conditions give applicability \(C\land\Phi^\times_\Sigma\); feasibility of
all matches under \(C\) requires a separate implication proof. The
[unified proof](complete_attribute_synthesis_proof.md), Section 11, gives the
full scalar extension; Sections 1--7 give the direct core constructor.

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

## General structural extension: only axis slots are unknown

The broader theorem in Section 12 of the unified proof admits
**metadata-weighted signed wiring**: identify whole axes with identity or
reflection, sum entire domains, reorder surviving axes, and introduce fresh
positive axes with supplied extents. A nonzero rational factor may be determined
by the declared semantics, supplied metadata, child shapes, and selected axes.
This factor is computed, not inferred. Local legality depends only on that
metadata, those shapes, and the local axis choice. Axis choices are finite and
independent, and their enumeration respects any supplied axis fields. The
read-once and distinct-input-set additive restrictions remain.

This single contract includes supplied-extent broadcast (including singleton
expansion), unit-dimension insertion/removal, keepdims summation, and full-axis
mean. `axis` fields alone may be `auto`; `extent`, `coordinate`, and general
`u64`/`i64` fields are supplied programs or defaults. Their values and definedness
are preserved. Physical inputs and introduced axes have positive extents;
singletons are allowed, but zero extents are outside this theorem.

Let \(s_\Sigma(p)\) collect the supplied fields, \(\alpha\) the marked axis
slots, and \(D_\Sigma(p,\alpha)\) the completed descriptors. There are uniform
terminating structural procedures such that

\[
\Phi^{\mathrm{wire}}_\Sigma(p)
\iff \exists\alpha\;\operatorname{Defined}(s_\Sigma(p))\land
 \operatorname{Valid}_\Sigma(p,D_\Sigma(p,\alpha)),
\tag{2}
\]

and on success the recovered axis assignment gives a valid completion without
changing supplied fields. The domain and arithmetic assumptions are part of the
statement, not inferred runtime guards.

**Proof idea.** Use the canonical state
\((\mathbf n,\Pi,\varepsilon,\lambda)\): ordered output shape, signed
partition of observable non-unit input/output ports, and exact rational weight.
A unit coordinate is always zero, so its relationships are omitted. A closed
class with no physical input or output port contributes its extent to the
weight; a pure output class carries an axis on which the value is constant.
The remaining partition is exact by the one-hot separation argument, and a
satisfying one-hot input exposes the nonzero weight. Equal states are
interchangeable in a parent because both legality and its declared factor depend
only on the state shapes and fixed local metadata.

Induction therefore proves that the reachable-state tables contain exactly the
realizable states, with one sufficient witness per state. Match source and RHS
root states, then combine independent witnesses using capture-scaling separation
of additive components. All weights must be part of the table key: an eliminated
broadcast axis can compensate a scalar coefficient. The procedure traverses
axis structure and exact metadata, not tensor coordinates or tensor entries.
It may still be combinatorial in axis choices and states.

General convolution, slice, reshape, padding, cumulative sum, and value-dependent
rearrangement do not enter merely by supplying their non-axis attributes: they
need a proved adapter to this wiring contract or a different exact symbolic
representation. The theorem is a mathematical extension, not a completeness
claim for the current bounded C++ synthesizer, a strict floating-point result,
or a new measured XLA coverage count.

**Closure under linear contexts and normalization.** Sections 12.6--12.7 of
the unified proof allow unary linear operations (including broadcast,
permutation, reversal, sum, and mean) around separated additions. Joint tuples
of component states retain one common axis choice at each original node;
distribution never creates independently inferred copies of that choice.
This admits broadcast over addition while preserving the completeness argument.

Checked unit-only reshape normalization uses broadcast to insert size-one
axes and one-element summation to remove them. A restricted exact-real
scalar-division normalization also applies when an independent scalar capture
occurs only through its reciprocal and is nonzero: replace that reciprocal
by a common scalar input. Polynomial identity on all nonzero scalar values
is equivalent to identity on all real scalar values, so axis completeness
transfers. These are proved admission routes, not general reshape/division
support; the nonzero domain, assignment correspondence, and original legality
obligations remain explicit.
