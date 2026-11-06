# Existence and ambiguity of automatic tensor attributes

> Historical research material. Implementation references and reproduction
> instructions describe the earlier approach. See the
> [current synthesis design](../../../design/attribute-synthesis.md) and
> [research index](../README.md) for present scope.

**Research draft, October 11, 2026.** The proposed paper subject is a complete
description of the attributes that realize a prescribed tensor rewrite:
when this set is empty, how to construct a member, and when that member is
unique. The useful specialization is ordered contraction, with an extension
to independently parameterized additive branches. It covers the structure of
all four rules in [algebraic.tepl](../../../tests/fixtures/semantic/rules/algebraic.tepl), under explicit
restrictions on metadata and semantics.

This builds on the existing [ordered-contraction theorem](../archive/ordered_contractions.md)
and the degree-decomposition note (`semantic_decomposition_proofs.md`, not retained in this checkout). The additions here
are a characterization and exact count of **all** attribute assignments, an
explicit connection to ambiguity policy, and a sufficient admission criterion
that combines those results for distribution and LoRA. These are hand arguments
with a finite enumeration experiment, not machine-checked proofs or established
novelty claims. The general SMT path is not proved complete by this note.

## 1. Define what auto promises

Let `p` contain input shapes, dtypes, known source attributes, and metadata-only
rule conditions. Let `X` range over independently variable input tensors. Let
`beta` be the entire tuple of automatic RHS attributes, including attributes
on different operations. Fix ideal real arithmetic.

Define the solution set at legal source metadata `p`:

\[
\mathcal A(p)=\{\beta:\operatorname{Legal}_R(p,\beta)
\land\operatorname{Shape}_L(p)=\operatorname{Shape}_R(p,\beta)
\land\forall X\;L(p,X)=R(p,\beta,X)\}.
\]

Output shape is an **ordered** dimension vector. Legality includes every
intermediate operation and all pins or shared-field constraints. Element
equality is considered only for well-defined, equally shaped tensors.

There are three separate questions:

1. **Existence:** is `A(p)` nonempty? Ordinary existential `auto` needs this.
2. **Identifiability:** does `A(p)` contain exactly one tuple, or one tuple
   modulo a precisely specified equivalence on attribute representations?
3. **Coverage:** for every legal source match satisfying a guard, is `A(p)`
   nonempty?

The coverage quantifiers are

\[
\forall p:\operatorname{Legal}_L(p)\land G(p)
\Longrightarrow\exists\beta\;\forall X:\operatorname{Correct}(p,\beta,X).
\]

The attributes may depend on metadata `p`; they may not depend on tensor values
`X`. Moving the existential quantifier after `forall X` changes the problem.
Finding a witness for one `p` does not establish coverage.

| Situation | Meaning and appropriate behavior |
| --- | --- |
| `A(p)` has one member | Infer it. |
| `A(p)` has several members | Existential `auto` may choose any verified member; strict uniqueness mode must report ambiguity. |
| `A(p)` is empty, proved completely | Decline this match; reject a claim that the rule covers every source match. |
| Analysis cannot decide | Return `Unknown`; this is neither impossibility nor ambiguity. |
| Operation semantics are outside an admitted fragment | Use a general sound fallback or diagnose unsupported inference. |

Multiple valid attributes cannot disagree about the observable output: every
member already equals the source for every input. They may still change
intermediate layouts, reduction traversal, or cost. If those properties matter,
put them in the specification or in a separate choice objective. Shape-only
alternatives that have different element semantics are not members of `A(p)`.

An `auto_unique` mode would be a proposed language extension, not current TEPL
syntax. A practical default is existential `auto` with deterministic selection.

## 2. A general exact foundation for polynomial semantics

Assume reads, rational constants, addition, subtraction, multiplication, and
finite summation; indexing and domains depend only on metadata. Assume a finite,
effectively enumerable attribute domain at the selected `p` and terminating
metadata evaluation. Repeated inputs are allowed in this section.

Introduce a formal variable `z_(t,i)` for every entry `i` of input tensor `t`.
For each output coordinate `o`, composition of the operation definitions yields
a polynomial `P_(E,p,beta,o)` over these variables. Then

\[
\forall X\;L(p,X)=R(p,\beta,X)
\iff
\forall o,m:\ [m]P_{L,p,o}=[m]P_{R,p,\beta,o}.
\]

Here `[m]P` is the coefficient of monomial `m`. The forward direction uses the
fact that a real polynomial vanishing on all real inputs has zero coefficients.
The reverse direction follows by evaluating equal polynomials.

This gives a complete, finite baseline algorithm: enumerate legal attribute
tuples, expand each output polynomial exactly, and compare coefficients. Record
all successful tuples or stop at one if only existence is requested. Equal
dimension sizes never justify identifying different entry variables. Empty
domains, singleton domains, repeated reads, and cancellation are handled by
actual expansion.

The baseline can be prohibitively expensive. Finite decidability is not an
efficient inference result. The research problem is to replace it with a compact,
necessary-and-sufficient characterization for a useful semantic fragment.

An exact SMT encoding would define a formula `Phi(p,beta)` equivalent to these
legality and coefficient obligations. Existence is satisfiability of `Phi`.
After finding `beta0`, uniqueness is unsatisfiability of

\[
\Phi(p,\beta)\land\beta\ne\beta_0.
\]

This test needs an **exact** encoding. Applying it to TEPL's sufficient access
matcher can establish existence with verification, but cannot in general prove
uniqueness or impossibility. If counting encoded models, project out helper
variables and inactive list padding; those are not attributes.

## 3. The tractable ordered-contraction fragment

Admit an operation by a verified semantic contract, not its name. A contraction
must multiply two reads, pair selected axes as retained batches or reductions,
and output batch axes followed by unpaired left axes followed by unpaired right
axes. A permutation must only reorder all axes. The operation's legality
contract must agree too; element semantics alone are insufficient.

For the first theorem, require:

- The same independent capture variables, each used once, on both sides.
- Trees of certified contractions and permutations.
- Every RHS descriptor unrestricted and independent of every other descriptor.
- Fixed legal source attributes, arbitrary finite ranks, and dimension sizes
  at least two; rank-zero tensors are allowed.
- Exact arithmetic, no tensor-value restrictions, and no broadcasting, diagonal
  extraction, reshape, separate unary reduction, or value-dependent indexing.

Pins, shared descriptors, extra semantic attributes, and extra legality
assertions require further reasoning. They are not automatically admitted.

### 3.1 Recover logical indices from the source

Treat each physical input axis as a port. Source pairings partition these ports
into logical indices. A logical index `a` has a support `S(a)`, the set of input
tensors carrying it, an extent, and either a final output position or a summed
status. Let `F` be the ordered final output list.

This network is an exact invariant in the admitted fragment. A monomial picks
one entry of each named input. It occurs exactly when the selected coordinates
satisfy the network's equalities and final output attachments. Its coefficient
is one because every summed index is observed in an input read. If two
partitions or output attachments differ, coordinates zero and one distinguish
them. This is why independent inputs and extents at least two matter.

### 3.2 Every intermediate interface is forced

For a subtree whose input set is `U`, define its live logical axes:

\[
\partial U=\{a:S(a)\cap U\ne\varnothing
\land(a\in F\ \lor\ S(a)\nsubseteq U)\}.
\]

An axis needed outside the subtree cannot be reduced early. An internal summed
axis must be eliminated at the lowest common ancestor of its supporting
inputs: above that node, no sibling carries it, so no legal contraction can
eliminate it without changing the network. This proves that the live set is
exactly `partial U`, not merely a sufficient choice.

At a dot node with child live sets `L,R` and parent live set `P`, infer

\[
B=L\cap R\cap P,\qquad K=(L\cap R)\setminus P,
\qquad L_f=L\setminus R,\qquad R_f=R\setminus L.
\]

These are batches, contractions, left-free axes, and right-free axes.
The logical pairings and roles are forced. Only certain orderings remain free.

### 3.3 Decide layout feasibility

Pass an ordered required subsequence `Q` down the RHS tree, starting with `F`.
At a dot, label each required axis by `B`, `L`, or `R`, according to its role.
The labels must form a word in

\[
B^*L^*R^*.
\]

Pass the left-free subsequence to the left child and the right-free subsequence
to the right child. Batch axes impose no child ordering: paired axis lists can
select them in any positional order. At an automatic transpose, pass an empty
requirement to its child. At a leaf, `Q` must be a subsequence of its physical
axis order.

If the checks pass, construct layouts bottom-up. Choose a batch order extending
its required subsequence; choose any ordering of contractions; append the
children's free axes in their existing order. Convert logical axes to child
positions to obtain the four concrete attribute lists. A transpose chooses an
output order extending `Q` and translates it into a positional permutation.

The existing ordered-contraction theorem proves

\[
\mathcal A(p)\ne\varnothing
\iff\text{all these order checks pass}.
\]

An inverted pair at a dot or leaf certifies failure of **every** RHS assignment.
One unsuccessfully tried assignment does not supply this certificate.

## 4. New characterization of the complete solution set

Existence does not explain ambiguity. The same interface analysis gives more:
it parameterizes **every** solution by independent local order choices.

For each dot node `v`, let `b_v` and `k_v` be its batch and contraction counts,
and let `q_v` be the number of batch axes in its incoming required subsequence.
For each transpose `v`, let `r_v` be its rank and `s_v` the length of its
incoming required subsequence.

**Solution-space theorem.** In the fragment above, if any order check fails,
`|A(p)| = 0`. Otherwise,

\[
\boxed{
|\mathcal A(p)|=
\prod_{v\text{ dot}}\left(k_v!\frac{b_v!}{q_v!}\right)
\prod_{v\text{ transpose}}\frac{r_v!}{s_v!}.
}
\]

This counts actual active attribute lists, including simultaneous reorderings
of the two contracting lists. It does not count scalar tensor values, bound
variable names, solver auxiliaries, or inactive vector padding. Use `0! = 1`.

**Proof.** Forced interfaces determine all paired logical axes and their roles.
At a dot, a batch order can be any permutation of its `b_v` axes that preserves
the `q_v` specified axes' relative order. Exactly `b_v!/q_v!` permutations do
so: each of the `q_v!` relative orders occurs equally often among the `b_v!`
permutations. The `k_v` contractions can be listed in any of `k_v!` orders;
their right list is then forced by the logical pairing.

For any allowed child layouts and these local choices, each positional field
is uniquely determined. Conversely, a valid descriptor recovers its ordered
batch and contraction labels from the child layouts. Thus there is a bijection
between these choices and valid local descriptors. Unpaired axes retain their
child order and introduce no additional choice.

At a transpose, every output permutation preserving the `s_v` required axes'
order is allowed, giving `r_v!/s_v!` choices for each child layout. Its child
has no incoming order requirement. The corresponding positional permutation
is unique because each logical axis occurs at most once in that layout.

The required subsequences depend only on source logical labels and tree roles,
not on any of the local choices. Fields are independent, so induction on the
tree multiplies these counts. Leaves have one choice if their check passes,
and none otherwise. This proves both exhaustiveness and absence of duplicate
counting. ∎

### Consequences for inference and ambiguity

- **Raw uniqueness:** the rule instance is unambiguous exactly when the product
  is one. In particular, a node with two contraction axes already gives at
  least two raw assignments.
- **Uniqueness modulo reduction ordering:** explicitly identify simultaneous
  permutations of each node's two contracting lists. The count of equivalence
  classes is the formula with the `k_v!` factors removed. This equivalence is
  justified by finite-sum reindexing under ideal arithmetic.
- **Intermediate layout freedom:** factors `b_v!/q_v!` and `r_v!/s_v!` can still
  produce multiple classes. These choices preserve the complete rewrite's
  output but can affect implementation cost. The theorem gives a factored
  family of choices; it does not prove an efficient optimizer for arbitrary
  global costs.
- **Deterministic witness:** fix a stable ordering of source logical labels.
  Order contractions by it. Extend each required subsequence by placing it
  first and then all remaining labels in stable order. Translate to positions
  bottom-up. This always constructs a solution when one exists. It is a
  canonical convention, not a claim of the globally smallest attribute tuple.

The structural checks and construction require polynomial work in explicit
tree size and axis count. The count's integers have polynomial bit length in
the sum of node ranks; use arbitrary-precision arithmetic if materializing the
number. This is not a polynomial bound for arbitrary semantic normalization.

For example, a full contraction of two matrices can use paired lists
`([0,1],[0,1])` or `([1,0],[1,0])`. Both sum the same products. Rejecting this
rewrite merely because two solutions exist would discard an ordinary correct
transformation. By contrast, swapping only one contracting list changes the
pairing and generally changes the value, even when both dimensions have equal
size.

Another example is `A => (transpose (transpose A))`: a rank-`r`
input admits `r!` inner permutations, with the inverse forced outside. Removing
reduction-order ambiguity does not make these attributes unique.

## 5. Extend the theorem to additive rules

The contraction theorem alone excludes `distribute` and `lora`. The extension
needs more than separately solving arbitrary normalized terms.

Scale each independent capture `X_i` by a scalar `z_i`. Every polynomial
expression decomposes into multidegree components:

\[
E(z_1X_1,\ldots,z_nX_n)=\sum_\alpha z^\alpha E_\alpha(X).
\]

Polynomial coefficient uniqueness gives `L = R` for all inputs exactly when
`L_alpha = R_alpha` for every degree vector `alpha`. This is the existing
degree-decomposition theorem; degree vectors describe capture multiplicities,
not axis numbers.

For a tractable extension, require the following **after certified algebraic
normalization**:

1. Each side has the same multidegrees and matching nonzero scalar coefficients.
   Each multidegree contains exactly one ordered contraction/permutation tree,
   using its captures once. Multiple terms of the same degree must not simply
   be paired arbitrarily.
2. Each RHS branch's attributes are private to that branch. Normalization must
   not silently copy one original unknown attribute into independent unknowns.
3. The RHS is a sum of those branches. Each branch is required to produce the
   source's ordered output shape. After imposing this requirement, all legality
   conditions factor by branch.
4. Each component meets the contraction theorem's remaining assumptions.

Let `A_alpha(p)` be the exact solution set of a branch. Degree equality forces
each branch to equal its corresponding source component, so

\[
\boxed{\mathcal A(p)=\prod_\alpha\mathcal A_\alpha(p).}
\]

Necessity follows by taking degree components of a valid full rewrite.
Sufficiency follows by combining branch witnesses: fields are disjoint and
their output shapes agree, so the sum is legal and componentwise equal.
This also proves that counts multiply, uniqueness reduces to branch uniqueness,
and one branch obstruction proves impossibility of the entire rewrite.

The result is polynomial in the **expanded** component trees and their axes.
Expansion can be exponential in the original expression size. A rewrite with
shared attributes across components remains a joint constraint problem and
is outside this product theorem.

### Application to the four example rules

| Rule | Components and exact conclusion within the admitted domain |
| --- | --- |
| `distribute` | Capture sets `{A,C}` and `{B,C}`. Copying the source descriptor to both RHS dots supplies a witness at every legal source match. If the source has `k` contracting axes, the raw solution count is `(k!)^2`. |
| `associate` | One component `{A,B,C}`. Boundary and order checks decide applicability exactly; some source matches fail. |
| `lora` | Components `{X,W}` and `{X,A,B}`. The first dot is always realizable by copying the source outer descriptor. The other two dots form one jointly solved contraction tree. The complete rule is realizable exactly when this second component passes its order checks. |
| `associate_reverse` | One component `{A,B,C}`. The same exact analysis applies in the opposite direction; it is not universally applicable either. |

The source dot/add contracts must establish the algebraic normalization and
branch shape facts. The fact that an operation happens to be named `add` or
`dot` provides no proof. Distinct capture variables can occur in several degree
components, as `X` does in LoRA; the requirement is once **within each component**.

For ordinary matrices `X[m,k]`, `W[k,n]`, `A[k,r]`, `B[r,n]`, the LoRA RHS uses
matrix multiplication on all three dots: left contraction `[1]`, right
contraction `[0]`, and empty batch lists. When these extents are at least two,
the product formula gives one raw assignment. This matrix instance does not
establish coverage for arbitrary generalized-dot descriptors.

## 6. Concrete impossibility in the supplied rules

Use length-two vectors throughout, so every ordering in these examples has the
same numerical output shape. Empty contracting and batching lists mean an
outer product; paired batching lists `[0]` mean elementwise multiplication of
two vectors.

For `associate`, let the inner source dot form `A[i] B[j]`, and let the outer
source dot batch its axis `1` with `C` axis `0`. Its ordered output is

\[
L[j,i]=A[i]B[j]C[j].
\]

In `dot(A, dot(B,C))`, the index `j` must remain live in the inner result. At
the root, `i` is left-free and `j` is right-free. The output must therefore be
`[i,j]`. The required `[j,i]` gives the forbidden role word `R,L`. No attribute
assignment repairs this while preserving the input-port partition.

For `associate_reverse`, use the reverse configuration: the source is
`A[i] * (B[j] C[j])`, where the inner dot batches `j` and the outer dot is an
outer product. Its output is `[i,j]`. The target `dot(dot(A,B),C)` must batch
`j` at its root, forcing output `[j,i]`. The required role word is `L,B`, also
forbidden.

The same obstruction occurs directly in `lora`. Let `X,W,A,B` be vectors, let
the source inner dot batch `A[j]` with `B[j]`, and let the source outer dot be
an outer product:

\[
L[i,j]=X[i]\bigl(W[j]+A[j]B[j]\bigr).
\]

The RHS `{X,A,B}` component must form an outer product of `X` and `A`, then
batch `j` with `B`. This forces output `[j,i]`, although the required output
is `[i,j]`. The `{X,W}` component cannot compensate: scaling `W` independently
separates it from the cubic component. In particular, setting `W=0` isolates
the failure.

A numerical illustration uses `X=(1,0)`, `A=B=(0,1)`, `W=(0,0)`. The source
matrix is `[[0,1],[0,0]]`; the RHS with the forced pairings produces
`[[0,0],[1,0]]`. This illustrates that pairing; the order obstruction, together
with network exactness, proves impossibility for all alternative assignments.
One numerical example alone would not prove that stronger claim.

These are reasons to generate an applicability guard, rather than treating
algebraic reassociation as universally legal for this ordered operation.

## 7. An explicit compiler algorithm and an exact guard

The proposed compiler output is a metadata procedure that computes a guard,
a witness, and an ambiguity description. Z3 is optional for the admitted
fragment.

```text
compile(rule, operation_definitions):
    certify contraction, permutation, and needed addition contracts
    normalize and compute capture multidegrees
    check component matching and independence of unknown descriptors
    if admission fails: retain general fallback / unsupported status
    otherwise emit the component trees and the procedure below

infer(legal_source_metadata):
    for each component:
        evaluate source pairings into a logical-index partition and output order
        compute each RHS subtree's forced boundary
        propagate required output subsequences from the component root
        if a block-order or leaf-order inversion occurs:
            return NoWitness(component, node, two logical axes)
        record permitted local orders and their exact counts
        construct one canonical choice bottom-up
        translate its logical axes to positional attribute lists
    combine the independently inferred descriptors
    independently check legality, source networks, and output order
    return Found(attributes, product_of_counts, local_choice_description)
```

Inside the admitted metadata domain, define `G_star(p)` as the conjunction of
all order checks in all components. The theorems give

\[
G_*(p)\iff\exists\beta\;\forall X:\operatorname{Correct}(p,\beta,X).
\]

Thus `G_star` is the **weakest sufficient applicability condition** within the
domain: any sound guard `H` must imply it. This eliminates the existential
attribute search into a directly executable metadata predicate. It is not
merely a heuristic necessary condition or a sufficient SMT constraint.

The same procedure constructs a metadata-dependent witness function `f(p)`:

\[
G_*(p)\Longrightarrow\forall X:\operatorname{Correct}(p,f(p),X).
\]

This is a useful paper formulation: derive the exact guard, a witness function,
and a factored description of all alternatives from certified operation
semantics and the rewrite skeleton.

An unconditional rule requires a further proof that every allowed source
configuration satisfies `G_star`. The existing two-axis cutoff is relevant for
the independent-input, unguarded contraction fragment under its stated source
restrictions. Do not automatically extend that cutoff to additive components,
shared metadata, or arbitrary user guards: deleting axes must preserve all
source legality conditions and the guard.

### Relation to the current implementation

The repository already has substantial pieces of this foundation:

- ordered.cc extracts conservative
  operation contracts and constructs/checks network witnesses and two-axis
  obstructions. It does not currently count solutions or admit additive rules
  through the product theorem.
- semantic_ir.h explicitly describes
  the compiled constraint plan as a sufficient equality condition.
- solver.cc falls back when sufficient
  constraints are exhausted and independently verifies candidates. This is
  essential: unsatisfiability of a sufficient encoding is not impossibility.

No production inference changes are introduced by this research note or its
experiment.

## 8. Restrictions and possible changes to semantics

Start with the existing mathematical dot semantics and make admission explicit.
A practical certification target is a normalized access relation, legality
predicate, scalar combining operation, and ordered output interface. Equivalent
names and field names should have no effect. An arbitrary equivalent semantic
program need not be recognized by the initial certifier.

The following cases need a broader method:

| Case | Why the complete structural theorem does not apply |
| --- | --- |
| Dimensions zero or one | Empty tensors and singleton coordinates can hide different index connections; use exact polynomial reasoning or a separately proved quotient. |
| Repeated captures within a component | Independent factor labels are lost; different read-occurrence matchings may describe the same polynomial. |
| Several terms with the same multidegree | Equality is between whole polynomial components; termwise matching need not be complete. |
| Pinned or shared attributes | Local ordering choices become coupled; the product count and independent constructor need not survive. |
| Unused reductions | Domain cardinalities affect coefficients and are not represented by a read-once network. |
| Reshape, broadcasting, coordinate arithmetic | Port partitions alone no longer characterize access maps. |
| Additional shape assertions | A structurally valid assignment may violate the custom legality contract. |
| Floating-point or execution-order requirements | Mathematical sum reindexing and distribution may not preserve the specified operational behavior. |

For finite polynomial instances, the baseline coefficient algorithm still
provides a mathematical specification of inference. A practical fallback may
time out; that remains `Unknown`.

If unconditional reassociation is important, adding a freely chosen output
permutation is a principled semantic extension. For a single contraction tree,
an automatic transpose at the root can repair every ordering obstruction in
this fragment. The live connections are constructible with no root order
obligation, and the transpose supplies the required final order.

For additive rules, allow a transpose on each independently inferred branch as
needed. One transpose outside the entire sum is not automatically sufficient:
the branches must already agree in order before addition. An output-permutation
attribute on a generalized contraction can expose equivalent expressive power
if lowering implements it correctly. It may incur a real layout cost; the
semantic theorem does not make that cost disappear.

## 9. Evidence and conference positioning

The experiment [check_auto_solution_space.py](../archive/check_auto_solution_space.py)
compares the closed-form count with exhaustive enumeration of every legal
ordered dot descriptor and transpose permutation. All physical dimensions are
two. It also compares the existing constructor against the oracle's existence
answer. It uses the same network evaluator as the earlier contraction
experiment, so network-to-tensor exactness remains a separate proof obligation.

```sh
python3 -B research/attribute-synthesis/archive/check_auto_solution_space.py
python3 -B research/attribute-synthesis/archive/check_auto_solution_space.py --max-total-axes 6
```

Results on October 11, 2026, with three inputs of rank at most two and 48 RHS
tree/permutation skeletons:

| Total input axes allowed | Decisions | No solution | Unique | Multiple solutions | Largest solution set |
| --- | --- | --- | --- | --- | --- |
| At most 4 | 26,352 | 11,928 | 11,880 | 2,544 | 4 |
| At most 6 | 305,040 | 192,516 | 82,476 | 30,048 | 24 |

Both runs had exact count agreement and correct reconstructed witnesses. The
smaller run is contained in the larger run; their counts should not be added.
These are finite checks of the counting extension, not a proof of arbitrary
ranks, performance measurements, or validation of the entire additive compiler
extension.

### Related work that limits a novelty claim

- [SPORES, VLDB 2020](https://www.vldb.org/pvldb/vol13/p1919-wang.pdf), especially
  Section 2.3, already gives a completeness argument for sum-product optimization
  through relational normal forms. Polynomial normalization and logical-index
  representations alone are not a defensible new contribution.
- [TASO, SOSP 2019](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf)
  generates graph substitutions from operator specifications and verifies
  candidates. Automatic optimization from supplied operator definitions is
  already an established goal.
- [TensorRight, POPL 2025](https://lsrcz.github.io/files/POPL25.pdf) verifies
  rank- and size-polymorphic tensor rewrites using a proved rank-bound analysis.
  A new arbitrary-rank or small-model claim must explain the different
  existential RHS-attribute obligation; a fixed-attribute counterexample
  projection does not automatically preserve the failure of all RHS choices.
- [Prism, 2026 preprint](https://arxiv.org/html/2604.15272v1), Sections 3.4 and 4,
  uses symbolic mappings, enumerates constrained mapping choices, and removes
  dimension-permutation symmetries before equivalence verification. Symbolic
  parameters and symmetry breaking alone are not sufficient differentiation.

These comparisons motivate a narrower proposed contribution:

> From certified user-defined operation semantics and a fixed rewrite
> skeleton, derive a complete, factored description of legal RHS attribute
> assignments. For ordered contractions and independent additive components,
> infer the exact applicability guard, construct a canonical witness, decide
> ambiguity, and produce small certificates when no assignment exists.

A possible title is **Complete Attribute Synthesis for Ordered Tensor
Rewrites**. The key distinction to investigate is complete realizability of a
prescribed ordered implementation, including absence of *any* assignment and
the entire space of successful assignments. This is a candidate research
position, not an assertion that related work lacks every part of it.

The counting formula is a useful theorem but probably too incremental for a
strong conference paper on its own. A credible package needs:

1. Independent review or mechanization of network exactness, forced interfaces,
   completeness, the counting bijection, and additive admission.
2. An implemented compiler pass that derives contracts and component obligations
   from user-defined semantics, with explicit failures outside its fragment.
3. Benchmarks comparing complete construction with generic SMT, sufficient
   constraint matching, and metadata-only search, including impossible cases,
   repeated extents, high ranks, many reductions, and renamed custom operations.
4. Coverage and cost measurements on useful rewrite libraries: admitted rules,
   inferred guards, ambiguity frequency, inference latency, and optimization
   effects. Fixed-rank success alone is not coverage evidence.
5. A closer related-work audit of tensor layout synthesis, relational query
   compilation, and synthesis of preconditions before claiming novelty.

The most defensible next implementation is component admission plus the exact
solution-family analysis, reusing the existing independent witness checker.
This would turn the mathematical specification into a reviewable compiler
algorithm for the user's actual algebraic rules.


