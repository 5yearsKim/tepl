# Proofs for decomposing tensor attribute inference

The decomposition and constraint-solving results below are provable. The hard
part is generating small, efficiently decidable constraints from general tensor
semantics. These proofs do not establish that every operation admits such
constraints, or give an arbitrary-rank cutoff.

These are mathematical arguments, not machine-checked proofs or novelty claims.

## Assumptions

Fix metadata `p`: finite input shapes and known attributes. Inputs are independent
tensors with arbitrary real entries. Use exact arithmetic, tensor reads,
addition, multiplication, finite sums, and indexing independent of tensor values.
Shape and attribute legality also depend only on metadata. Repeated reads are
allowed; floating-point rounding and nonlinear value operations are excluded.

Let `beta` be the unknown RHS attributes. First require legality and equal
ordered output shapes through a predicate `C0(p, beta)`. Elementwise equality
is considered only when this predicate holds.

## Theorem 1: exact decomposition by input degree

Scale each input independently: `Xi -> si * Xi`. For fixed legal attributes,
every expression has a finite expansion

\[
 E(s_1X_1,\ldots,s_nX_n)
   =\sum_{\alpha\in\mathbb N^n}s^\alpha E_\alpha(X),
 \qquad s^\alpha=\prod_i s_i^{\alpha_i}.
\]

Here `E_alpha` is a tensor-valued polynomial component; missing components are
zero. Then

\[
 \forall X:\ L(X)=R(X)
 \iff
 \forall\alpha\;\forall X:\ L_\alpha(X)=R_\alpha(X).
\]

**Proof.** Reads of `Xi` have degree vector `ei`; constants have degree zero.
Addition combines components of the same degree. Multiplication uses
`(FG)_alpha = sum_{u+v=alpha} F_u G_v`. Value-independent indexing preserves
degrees, and finite summation acts on each component separately. Structural
induction therefore constructs the expansion.

If `L = R` for all inputs, fix any `X` and scale it by arbitrary real `s`.
For every output element, their difference is a polynomial in `s` that vanishes
everywhere. Every coefficient is consequently zero. Since `X` was arbitrary,
the corresponding components are equal for all inputs. Conversely, summing
equal components and setting every `si = 1` gives `L = R`. ∎

This works at every finite rank, but does not reduce the number of ranks that
must be considered. It compares whole degree components, not individual terms:
`AB + BA` has one degree component, which must remain together.

## Theorem 2: exact factorization of attribute existence

Let `C_alpha(p, beta)` state equality of the corresponding degree components
for all input values. Include all legality and shape conditions in `C0`. Then

\[
 \exists\beta\;\operatorname{Valid}(p,\beta)
 \iff
 \exists\beta:\ C_0(p,\beta)\land\bigwedge_\alpha C_\alpha(p,\beta).
\]

**Proof.** Theorem 1 gives the equivalence for each legal assignment `beta`.
Conjoin legality and shape equality, then existentially quantify that *same*
assignment on both sides. ∎

If the resulting constraints split into groups using disjoint attribute sets
`beta_1, ..., beta_k`, with independent domains, then additionally

\[
 \exists\beta_1\cdots\beta_k\ \bigwedge_j F_j(p,\beta_j)
 \iff
 \bigwedge_j\exists\beta_j\ F_j(p,\beta_j).
\]

**Proof.** A joint witness restricts to a witness for each group. Conversely,
combine the witnesses for the disjoint groups; no variable or constraint is
shared, so their union satisfies every group. ∎

Shared attributes, shape constraints, and domain restrictions must remain in
the joint problem. Dropping these dependencies would invalidate the result.

### LoRA consequence

For ideal bilinear dot and ordinary elementwise addition,

\[
 X(W+AB)\quad\rightsquigarrow\quad
 \operatorname{dot}_{xw}(X,W)+
 \operatorname{dot}_{out}(\operatorname{dot}_{xa}(X,A),B)
\]

has two degree components: `sX*sW` and `sX*sA*sB`. With known source metadata,
independent RHS descriptors, and each RHS branch required to have the source
output shape, the obligations split into `{xw}` and `{xa, out}`.

If each descriptor has at most `D` candidates, exhaustive search needs at most
`D + D^2` local assignment checks, rather than `D^3` joint checks. This counts
assignments, not the cost of proving each component equality. Additional shared
constraints can prevent this split.

## Theorem 3: solving cost bounded by attribute interaction width

Suppose the exact constraints above are available as decidable predicates with
finite attribute domains of size at most `D`. There are `m` constraints. Supply
a tree of `b` bags of attribute variables satisfying:

1. Every constraint's variables occur together in some bag.
2. Bags containing any given variable form a connected subtree.
3. Each bag contains at most `w + 1` variables.

Then existence and witness construction can be decided by dynamic programming
in

\[
 O\big((b+m)D^{w+1}T\big)
\]

time. Here `T >= 1` bounds the cost of one local predicate evaluation and the
per-assignment bookkeeping on a bag. Construction of the constraints and bag
decomposition is outside this bound.

**Proof.** Root the bag tree and assign every constraint to one containing bag.
For each bag assignment, check its local constraints and whether every child
has a feasible assignment agreeing on the shared variables. A child's message
records feasible assignments to that intersection, so each compatibility check
is a table lookup, not a scan over all child assignments.

The connected-subtree property ensures different child subtrees share no
unrecorded variables. Induction therefore shows that each message records
exactly the assignments extendable throughout its subtree. The root accepts
exactly when a full solution exists. Store witness assignments to reconstruct
one top-down. Each bag enumerates at most `D^(w+1)` assignments; summing local
constraint evaluations and child-message lookups gives the stated bound. ∎

This is an application of established variable-elimination methods; see
[Dechter, Tractable Structures for Constraint Satisfaction Problems](https://ics.uci.edu/~dechter/publications/r139.pdf).

## What remains hard

- `C_alpha` is initially a semantic predicate, not automatically a cheap SMT
  formula. An exact component checker is still required.
- Component expansion can be exponential. Circuit representations may help,
  but these proofs give no general compactness guarantee.
- Shared constraints may produce large width. Entire axis-list domains can
  themselves grow factorially with rank.
- Timeout or incomplete component checking must return `Unknown`; it cannot
  supply an impossibility proof.
- These are fixed-metadata results. They do not prove that attributes exist
  for every shape/rank or for any configuration in a symbolic shape family.

The paper opportunity is to derive this exact factorization from user-defined
semantics, identify useful fragments with cheap component checks and small
interaction width, and demonstrate the reduction in actual inference work.
The algebraic decomposition and generic width theorem alone are not sufficient
novelty claims.
