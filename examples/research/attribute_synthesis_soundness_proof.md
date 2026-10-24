# Soundness of semantically verified attribute synthesis

**Status: mathematical proof draft, October 22, 2026.** This note gives a
modular soundness theorem for a specified verification discipline. Its short
paper statement is below; the supporting obligations follow. It complements
[complete attribute synthesis](complete_attribute_synthesis_proof.md), whose
exact feasibility result applies to the restricted contraction/permutation/
unary-summation fragment and its separated additive components. This note
does not establish that every rule accepted by the current C++ implementation
meets these obligations, or formally verify that implementation.

## 1. Setting and short paper proof

Let \(T=(L,R,C)\) be a rewrite with metadata-only condition \(C\). Metadata
\(p\) contains input shapes, dtypes, and captured source attributes. Fix an
arithmetic policy \(\alpha\) and precise, pure operation semantics. Shapes,
metadata collections, and reduction domains are finite. A compiler
produces an applicability predicate \(G(p)\), an attribute program \(f(p)\),
and a checked derivation. The predicate includes \(C\), any declared domain
restriction, and every additional runtime obligation. Additional conditions
must be exposed as part of the rule's applicability, not silently assumed.
The attribute tuple \(\beta=f(p)\) depends on metadata, not tensor values.

Let \(\mathcal X_L^\alpha(p)\) be the input tensors of the prescribed shapes
and dtypes on which the source is defined. Define

\[
\begin{aligned}
\operatorname{Valid}_\alpha(T,p,\beta)\iff{}&
  \operatorname{Legal}_R(p,\beta)\\
&\land\operatorname{Shape}_R(p,\beta)=\operatorname{Shape}_L(p)\\
&\land\operatorname{Dtype}_R(p,\beta)=\operatorname{Dtype}_L(p)\\
&\land\forall X\in\mathcal X_L^\alpha(p):\quad
  \llbracket R\rrbracket^\alpha_{p,\beta,X}=
  \llbracket L\rrbracket^\alpha_{p,X}.
\end{aligned}
\]

The final line requires the RHS to be defined for each such \(X\).
Legality includes every intermediate operation and all descriptor constraints.
Equality is the observation required by \(\alpha\); exact-real
equality does not establish strict floating-point equality.

**Theorem (Sound guarded attribute synthesis).** Suppose acceptance requires
a derivation using the sound local rules specified in Section 2, faithful
operation lowering, and the metadata-program obligations in Section 3. For
every legal source configuration \(p\), applicability evaluation terminates;
if \(G(p)\) is true, construction terminates with \(\beta=f(p)\) and

\[
\operatorname{Valid}_\alpha(T,p,f(p)).
\]

**Proof.** Fix such a \(p\) with \(G(p)\). By the metadata-program invariant,
every executed constructor instruction is defined, and finite traversals and
bounded loops terminate, yielding \(\beta\). The checked legality derivation
establishes the legality, shape, and dtype obligations for this assignment.
Fix any \(X\in\mathcal X_L^\alpha(p)\). Faithful lowering gives the exact
source and replacement meanings. Induction on the checked equality derivation
preserves definedness and equality: reads use equal coordinates, scalar
congruence preserves equal arguments, reductions obey their checked domain
and reducer laws, and each algebraic step uses a valid law with discharged
side conditions. Consequently the replacement is defined and equals the
source for this \(X\). Since \(X\) was arbitrary, the result holds for every
source-defined input. \(\square\)

This is a soundness theorem, not an existence decision procedure. Candidate
generation may be heuristic. Failure, an unproved obligation, or a resource
limit does not establish that valid replacement attributes do not exist.
For a resource-bounded implementation, the theorem concerns completed
successful checks and executions; a resource failure must not apply a rewrite.

## 2. Local semantic lemmas supporting the proof

These rules describe a mathematical verification discipline. They are not
assumptions that a routine named `verify` is correct. Each implemented
normalization, matcher, and side-condition solver must refine the corresponding
rule before the theorem can be attributed to that implementation.

**Lemma 1 (Faithful lowering).** Suppose each operation declaration specifies
its legality, ordered output shape, dtype, coordinate access maps, scalar
function, and reduction behavior faithfully. Compositional lowering preserves
the expression's denotation and definedness.

*Proof.* Induct over the finite expression tree, or a topological order for a
pure acyclic graph. Captures denote their input tensors. At an operation,
substitute the induction hypotheses for its operands into its declared
semantics. The faithful operation contract gives the required result. Shared
captures retain their identity; this argument requires no read-once assumption.

**Lemma 2 (Sound equality derivations).** A derivation built from the following
rules establishes defined equality under its stated assumptions:

| Rule | Required premise and justification |
| --- | --- |
| Tensor read | The same capture is read at equal, in-bounds coordinate tuples. Its values are therefore equal. Matching extents alone is insufficient. |
| Scalar congruence | Corresponding arguments are equal and the same deterministic function is defined on them. Applying that function preserves equality. |
| Ordered reduction | Equal initial values and the same reducer consume pointwise equal sequences in the same order. Induction on sequence length proves equality. |
| Reduction reindexing | A checked bijection relates the reduction domains, corresponding terms agree, and the reducer permits that reordering under \(\alpha\). For a commutative monoid, the finite-fold laws justify it. |
| Scalar/reduction identity | A separately proved law applies, with every domain, sign, independence, or arithmetic-policy premise discharged. Distribution is included only when its specific law holds. |
| Composition | Reflexivity, symmetry, transitivity, and replacement in a pure well-defined context preserve defined equality. |

*Proof.* The read and congruence cases follow from function application. The
ordered reduction case follows by induction on its finite length. The
reindexing case follows from the specified reducer laws and domain bijection,
including the empty-domain convention. Each registered identity has its own
local proof. Induction over the derivation then establishes the claim.

Partial functions require explicit definedness reasoning: for example, a law
using division cannot discard a nonzero-denominator premise. No tensor-value
assumption may be inferred merely from an axis relationship. A metadata guard
cannot express an arbitrary condition on tensor contents.

**Example: elementwise functions.** For a full permutation \(P\) and a
pointwise function \(h\), both \(P(h(A))\) and \(h(P(A))\) read the same
element of \(A\) at each output coordinate and apply \(h\). Thus they are
equal wherever the source is defined. This covers ReLU and exact-real exp
without requiring either function to be injective. More generally,
\(L=R\) implies \(h(L)=h(R)\) wherever both applications are defined.
These are soundness arguments; arbitrary movement through sums or products
requires additional laws.

## 3. Constructor, guard, and verifier obligations

**Lemma 3 (Metadata-program correctness).** Give each metadata instruction its
precise mathematical meaning. Require finite syntax, bounded loops over finite
collections, and an invariant establishing every executed instruction's
preconditions. Then the generated program terminates and implements its
specified metadata computation.

*Proof.* Induct over instructions and use a loop invariant over each finite
iteration sequence. Reads return the specified source metadata; list, set,
integer, and conditional operations preserve the invariant by their
definitions. Bounds establish that lookups and selections are defined.
Every loop has a finite bound, so execution terminates. The final invariant
identifies the resulting descriptors with the verified symbolic constructor.

The corresponding compiler obligations are:

- **Guard preservation:** emitted `where` evaluation implements \(G\), with
  the intended short-circuit and dependency order. A check depending on derived
  metadata runs only after that metadata is safely available.
- **Defined construction:** each executed instruction has its preconditions
  established by source legality, earlier checked facts, or an executed guard.
  A later check cannot justify an earlier undefined lookup. Machine integer
  operations also need representability and overflow checks.
- **Legality:** check every RHS operation contract, descriptor type, output
  shape, and dtype after substituting the proposed attributes. Value equality
  alone does not establish these facts.
- **Sound discharge:** side-condition solvers establish implications from
  valid premises. An unresolved obligation is rejected or added as an explicit,
  safely evaluable guard; it is never treated as a proved premise.
- **Execution correspondence:** the runtime applies the same verified
  constructor and guards to the same match and attributes, and applies the
  replacement only after successful evaluation.

## 4. Operation coverage and relation to completeness

The soundness theorem is parameterized by operation contracts and proved laws;
it is not restricted to two operation names. The following are extension
obligations, not a claim that every listed rewrite is currently implemented.

| Operation family | Additional facts needed for accepted rewrites |
| --- | --- |
| Contraction, matmul, transpose | Paired coordinates, ordered interfaces, reduction domains, and the selected arithmetic laws. |
| Add, multiply, negate | Scalar laws and compatible coordinate access. Repeated captures are permitted when their identities are preserved. |
| ReLU, exp, other pointwise functions | Congruence suffices for identical calls with equal arguments; other transformations need proved identities and domain premises. |
| Sum, max, other reductions | Exact reducer and initialization semantics, empty-domain behavior, and the particular reordering/distribution laws used. |
| Broadcast, reshape, slice, reverse | Correct coordinate maps, bounds, and output shapes; equal axis names alone are insufficient. |
| Convolution | Correct spatial access maps, reduction domains, output sizes, and stride/dilation/padding legality for the declared variant. |

General soundness does not require the completeness proof's read-once captures,
independent attribute holes, or extents-at-least-two domain. Instead, each
accepted derivation must establish its own obligations; removing a restriction
does not automatically make the existing algorithm handle the newly admitted
case. General user-defined scalar programs are covered only through their
specified semantics and justified laws, not an assumed universal equality
decision procedure.

Within the narrower fragment, the existing completeness theorem additionally
proves \(\Phi_T(p)\iff\exists\beta\,\operatorname{Valid}(T,p,\beta)\) for its
generated feasibility predicate \(\Phi_T\) and stated domain. Extra user conditions may
restrict that coverage. The soundness theorem here requires only that a true
applicability result produces a valid replacement.

## 5. Implementation correspondence still to establish

The general verifier in
[`proof/verify.cc`](../../src/core/semantics/proof/verify.cc) checks constructor
coverage, metadata definedness, capture accesses, reduction domains, ordered
axes, shapes, dtypes, and legality in a fresh proof context. Candidate guard
expressions there are obligations to prove, not extra assumptions. Candidate
generation in
[`attr_synth/synthesize.cc`](../../src/core/attr_synth/synthesize.cc)
is separate from this verification stage. Explicit inferred-applicability
uses the specialized ordered path described in the completeness proof.

To turn the mathematical theorem into a claim about all successful compiler
results, establish preservation for frontend lowering, every scalar and index
normalization, metadata reasoning, operation legality, program emission, and
runtime execution. The scalar function registry and algebraic flags are not
substitutes for proofs of the laws applied under a particular arithmetic
policy. Tests provide implementation evidence but do not discharge those
universal obligations. Faithfulness to a concrete tensor backend is a further
operation-contract obligation.
