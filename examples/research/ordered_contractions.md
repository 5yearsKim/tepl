# Two-axis obstructions for attribute realizability

**Status:** research draft, October 10, 2026. This note develops a hand proof and
a standalone exhaustive small-model experiment. It is not a proof-assistant
development or a novelty claim.

**Implementation update, October 11, 2026:** TEPL now has a conservative
primitive-IR contract extractor and an instance constructor with independently
checked network witnesses and ordering obstructions in `src/core/semantics/ordered.*`.
Generated Rust applications invoke it through the existing native runtime. The
two-axis compile-time coverage check is not implemented. This engineering work
does not replace independent review of the mathematical proof.

## Proposed paper contribution

Investigate **complete attribute inference and small impossibility certificates
for ordered tensor contractions**. The result below addresses arbitrary tensor
ranks and arbitrary legal batching/contracting pairings, while restricting the
expression structure and attribute coupling.

The proposed result is stronger than a necessary-condition filter: in the
stated fragment, attribute realizability is decided exactly by boundary and
ordering constraints. Every negative instance has a two-logical-axis
obstruction. Keeping those axes yields a counterexample with each input rank
at most two, while preserving the failure of *every* RHS attribute assignment.

Three consequences form a coherent paper contribution:

1. A polynomial-time constructor or impossibility certificate for a fixed
   metadata configuration, without enumerating attribute lists.
2. A componentwise rank cutoff of two for universal coverage in the unguarded
   fragment, with an explicit restriction argument under the existential
   attribute quantifier.
3. A repair theorem: one unrestricted output transpose makes any RHS
   contraction tree realizable in this fragment.

The mathematical result is about ordered contraction semantics. A further
systems contribution would extract this semantic contract from user-defined
operations instead of dispatching on their names. That extraction is not
partially implemented by the normal-form certifier described in the
[README](../README.md). A general extraction procedure is not proved here.

## 1. Exact domain

Both sides are rooted expression trees with the following properties:

- The same independent input tensors occur exactly once on each side.
- Binary operations have ideal `dot_general` semantics: pair selected axes;
  either retain the paired coordinate as a batch axis or sum it out; preserve
  unpaired axes. Output order is ordered batch axes, then left free axes, then
  right free axes.
- Unary transpose operations may occur. They only permute axes.
- Each RHS dot has four independent, unrestricted automatic axis lists. Each
  RHS transpose has an independent, unrestricted automatic permutation.
  There are no shared descriptors, pinned nonempty lists, or coupled fields.
- LHS attributes are fixed for the instance being decided and are legal.
- Every physical dimension has size at least two. Rank-zero scalar inputs and
  intermediates are allowed. Tensor values are arbitrary exact real numbers.
- There is no addition of tensor expressions, repeated input occurrence,
  diagonal extraction, separate unary reduction, reshape, broadcasting,
  coordinate arithmetic, or value-dependent control flow.

For the universal cutoff theorem, metadata ranges over all legal source
configurations in this domain, with source descriptors independently ranging
over their legal attributes too. Extra rule guards are discussed in Section 7.

This includes arbitrary-rank, multi-batch, multi-contract `dot_general` trees.
It excludes distributivity and LoRA. Supporting those requires an extension;
the current theorem must not be attached to them.

## 2. Normalize the source to logical indices

Give every physical input axis a distinct port. Pairing axes identifies their
ports; transposes change only order. Each equivalence class is a logical index
`a`. It has:

- A support `supp(a)`: the set of input tensors containing it.
- A common dimension size.
- Either a position in the ordered final output `F`, or a summed status.

No class contains two axes of the same input. Inductively, a dot joins disjoint
sets of input tensors, and an intermediate has at most one live axis per class.
Every summed class has support in at least two inputs. There are no unused
summation variables.

The source is a finite sum of one product containing one entry of every input.
Its computation is described by the input-port partition and the ordered free
classes. Intermediate contraction order is absent from this representation.

### Lemma 1: exactness of the network representation

For fixed legal metadata in this domain, two expressions are equal for all
input values, including output order, exactly when these representations agree.

**Proof.** Fix an output coordinate and expand the expression as a polynomial
in independent input-entry variables. Each monomial selects one entry from
each named input. A tuple of selected entries is present precisely when its
coordinates satisfy the class equalities and output attachments. Its coefficient
is one: every summed coordinate appears in a selected entry, so the selected
entries determine its value uniquely.

Equal representations therefore give equal polynomials. Conversely, treat the
ordered output positions as additional fixed ports. Unequal partitions have two
ports identified on one side and separated on the other. Give the separated
classes coordinates zero and one, and extend consistently within that side's
classes. Both coordinates exist in every domain. This yields a tuple of entries
and an output coordinate admitted by one side and excluded by the other.
Independent input variables make the resulting polynomial functions distinct.
Different output rank or shape is already a structural failure. This establishes
necessity as well as sufficiency. ∎

Equal dimension sizes do not make axes interchangeable. Tensor entries at
different coordinates are independent variables.

## 3. Intermediate axes are forced

For an RHS subtree containing the input set `U`, define

\[
 B(U)=\{a:\operatorname{supp}(a)\cap U\ne\varnothing
       \land (a\in F\ \lor\ \operatorname{supp}(a)\nsubseteq U)\}.
\]

These are its boundary axes: indices needed outside the subtree or by the final
output.

### Lemma 2: forced boundary

Every realization has exactly `B(U)` as the live logical axes of that subtree.

**Proof.** An index needed outside cannot be summed early: that would destroy
the required coordinate equality with the outside inputs. A final output index
must also remain live. Lemma 1 rules out repairing either loss by inventing a
different equality later.

Conversely, consider a summed class whose support is entirely inside `U`. Its
last possible elimination is the lowest common ancestor of its supporting
leaves. Both children there contain the class. Above that ancestor, no sibling
contains it, so a dot cannot contract it with the same class again. Pairing it
with a different class changes the network and is forbidden by Lemma 1.
Transposes cannot eliminate it. Thus it must be eliminated at that ancestor
and cannot remain live at `U`. ∎

At a binary node with child boundaries `L` and `R` and parent boundary `P`, the
roles are consequently forced:

\[
\begin{aligned}
 \mathrm{batch} &= L\cap R\cap P,\\
 \mathrm{contract} &= (L\cap R)\setminus P,\\
 \mathrm{leftFree} &= L\setminus R,\\
 \mathrm{rightFree} &= R\setminus L.
\end{aligned}
\]

There is no search over pairings: occurrences of the same source logical index
must be paired. Contracting-list order is immaterial under exact arithmetic.
Batch-list order remains a genuine layout choice.

## 4. Exact layout algorithm

Propagate an ordered subsequence requirement `Q` from the root downward. At the
root, `Q = F`, the entire required final output order.

At a dot node:

1. Label each element of `Q` as batch (`B`), left-free (`L`), or right-free (`R`).
2. Require the resulting word to belong to `B* L* R*`.
3. Pass the left-free subsequence to the left child and the right-free
   subsequence to the right child.
4. Do not pass the batch subsequence to either child. Arbitrary positional
   attribute lists can pair those axes in any required order, independently
   of their positions in the child layouts.

At a transpose, pass an empty requirement to its child. Its automatic
permutation can subsequently realize any required subsequence.

At a leaf, require `Q` to be a subsequence of that input's physical axis order.

After the checks succeed, construct layouts bottom-up. At each dot choose a
batch order extending its required batch subsequence, then append the child
free axes in their existing order. Translate logical classes into positions in
those child layouts to obtain concrete attributes. At a transpose, choose a
permutation whose output extends `Q`.

### Theorem 1: complete realizability and construction

For every legal source instance `p` and RHS skeleton in Section 1,

\[
 \exists\beta\;\operatorname{Valid}(p,\beta)
 \quad\Longleftrightarrow\quad
 \text{all checks in the layout algorithm succeed}.
\]

**Necessity.** Lemmas 1 and 2 force the connections and roles. A dot's output
consists of the three ordered blocks, so any required subsequence must have
nondecreasing block labels. Child free axes retain their relative order, giving
the recursive requirements. A leaf has its prescribed physical order.

**Sufficiency.** Induct on the RHS tree. Leaves satisfy their required
subsequences. A transpose can order its child's axes as required. At a dot,
the two recursively constructed child layouts satisfy the required free-axis
subsequences. Order its batches to satisfy the required batch subsequence.
The block check ensures that their concatenation satisfies `Q`. The forced
pairings and reductions preserve the source network. At the root, `Q` contains
all output classes, so its order agrees exactly with `F`. Lemma 1 establishes
equality for every input value. ∎

Straightforward implementations are polynomial in the number of expression
nodes and input axes. This note does not claim a tight complexity bound. The
algorithm never enumerates permutations or ordered axis-list combinations.

### A concrete failure

Let `A`, `B`, and `C` be vectors. The source first forms `A[i] B[j]`, then
batches its `j` axis with `C[j]`. Its result is

\[
 L[j,i]=A[i]B[j]C[j].
\]

The proposed RHS skeleton is `dot(A, dot(B, C))`. Its inner result must retain
`j` as a batch axis. At its root, `i` is left-free and `j` is right-free, so its
output must have order `[i,j]`. The required order `[j,i]` has block labels
`R,L`, which fail the local check. Changing any attributes cannot repair this
without changing the index partition. All three dimensions can be two; equality
of their sizes does not remove the tensor-value distinction.

The certificate identifies just the root, the two classes, their required
order, and their forced roles. It rejects all attribute assignments together.

## 5. The two-axis obstruction theorem

An unsuccessful check has a witness consisting of two required output classes:

- At a dot, they occur in descending block order, such as `R` before `B`.
- At a leaf, their required order reverses their physical input order.

### Theorem 2: existential-preserving small obstruction

If an instance in Section 1 has no valid RHS attributes, there are two final
logical indices `a,b` such that restricting the source and inputs to those
indices still leaves no valid RHS attributes. Each restricted input has rank
at most two.

**Proof.** Apply Theorem 1 and choose an unsuccessful local check. Select an
inverted pair `a,b` witnessing that failure. Delete every other logical class
and its physical input axes. Filter the source's axis lists and transpose
permutations and renumber their positions. This produces a legal source
configuration; scalar inputs/intermediates are allowed.

The retained classes have the same input supports. At every node, their
boundary presence and batch/left-free/right-free roles are unchanged. Their
final relative order and each leaf's relative order are also unchanged.
Propagation of a required subsequence is just filtering; transpose nodes
still stop its propagation. Hence the selected inversion reaches the same
unsuccessful check in the restricted instance.

By Theorem 1's necessity direction, **every** RHS assignment fails in this
restricted instance. Each logical class has at most one axis per input, so
keeping two classes gives input ranks at most two. ∎

This proof does not assume that arbitrary axis deletion preserves failure.
It chooses the deletion using a complete characterization of realizability.
That is what preserves the existential attribute quantifier.

### Corollary: a computable cutoff

For a fixed source/RHS skeleton in the unguarded fragment,

\[
 \left[\forall p\text{ with every input rank}\le2,
       \exists\beta\operatorname{Valid}(p,\beta)\right]
 \Longrightarrow
 \left[\forall p,\exists\beta\operatorname{Valid}(p,\beta)\right].
\]

The two-axis obstruction depends on connectivity and order, not dimension
sizes once they are at least two. Its dimensions may all be set to two without
repairing the failure. Thus the unguarded fragment has both a rank cutoff of
two and binary-extent counterexamples. No optimality claim is made for this
cutoff across particular families of source/RHS skeletons.

For the two retained labels, each input has one of five layouts:
`[]`, `[a]`, `[b]`, `[a,b]`, `[b,a]`. This gives a finite structural search space
for universal coverage: consider these layouts and both final output orders,
check whether the source can realize that target, and whether the RHS can.
This shortcut assumes source descriptors are unrestricted and independent too.
Do not use it for pinned or shared source attributes without modeling them.
The search can be exponential in the number of inputs; the theorem does not
claim polynomial-time universal schema verification.

## 6. Transpose repair

### Corollary: one output transpose is sufficient

For every source instance in the fragment and every binary RHS contraction
tree over the same independent inputs, adding one unrestricted automatic
transpose at the root makes the RHS realizable.

**Proof.** The transpose stops all ordering obligations from propagating into
the contraction tree. Every internal required subsequence is then empty, so
all layout checks succeed. The forced boundaries construct the correct network;
the final transpose supplies the required output order. ∎

This is an existence result. It does not establish that the transpose is cheap
or that reassociation preserves floating-point rounding. A follow-up research
problem is choosing cheap internal transpose placements instead of an expensive
root transpose, with a proved optimality criterion for a specified cost model.

## 7. Compile-time meaning and limits

For known ranks, shapes, and source attributes, Theorem 1 decides exactly
whether any RHS assignment exists. It can run during compilation when that
metadata is available, or at match time otherwise.

For an unguarded rank-polymorphic schema, Theorem 2 permits compile-time
rejection of a claim of universal coverage using a small metadata witness.
It does not prove that a rejected schema has no useful matches.

If the rule has a supplied condition `C`, the cutoff additionally requires
that the chosen restriction preserves `C`. Fixed-rank, minimum-rank, cardinality,
and attribute-sharing conditions may not survive deletion. A statement such as
`rank(A) = 100` is not covered by the rank-two cutoff. The instance algorithm
still applies to a concrete legal configuration satisfying that condition.

Unqualified existential applicability is weak in this fragment: allowing all
inputs to be scalars makes every contraction tree compute the same product.
Thus `exists metadata, exists attributes` alone cannot reject these trees.
The useful questions are realizability at given metadata, coverage of a supplied
metadata family, and impossibility within that family.

Other boundaries of the proof:

- Singleton/empty dimensions can invalidate the necessity argument by hiding
  differences. They need separate reasoning.
- Repeated inputs remove the independent-factor labeling used in Lemma 1.
- Addition introduces multiple monomials and shared attribute choices across
  terms. Solving each term separately is insufficient.
- Unary reductions can permit delayed elimination, invalidating Lemma 2 as
  stated. A broader interface analysis must represent those alternatives.
- Shared/pinned RHS descriptors can couple choices otherwise made independently.
- Arbitrary user-defined operations are not covered merely because they use
  some of the same primitive names.

## 8. Reproducible evidence

Run the accompanying experiment from the repository root:

```sh
python3 examples/semantic/research/check_ordered_contractions.py --transpose-cases
python3 examples/semantic/research/check_ordered_contractions.py --inputs 4 --max-rank 1 --max-total-axes 4 --transpose-cases
```

The oracle independently enumerates all legal positional pairings, chooses
batch or contraction for each pair, and enumerates every batch order. It
identifies only semantically irrelevant contracting-list reorderings. Networks
are compared by exact input-port partitions and ordered output classes, using
Lemma 1. Dimensions are all two.

The structural algorithm must agree with that oracle and must reconstruct the
exact target network on success. Every failure is checked for a two-axis
restriction that remains unrealizable under the exhaustive oracle and remains
realizable by the source family. A root transpose must repair each failure.

The experiment is finite corroboration, not the proof of arbitrary-rank
correctness. In particular, it shares the network representation justified by
Lemma 1; it is not an independent proof of that lemma's tensor semantics.

Results from the commands above on October 10, 2026:

| Inputs | Maximum input rank | Maximum total input axes | RHS skeletons | Decisions | Unrealizable / pair witnesses | Constructed |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 2 | 6 | 48 | 305,040 | 192,516 / 192,516 | 112,524 |
| 4 | 1 | 4 | 480 | 95,040 | 34,320 / 34,320 | 60,720 |

There were no disagreements. All 226,836 unrealizable cases had a two-axis
restriction certified by exhaustive enumeration, and all were repaired by a
root transpose. The 400,080 decisions are small-model checks, not distinct
real-world rewrite rules. Source families include optional root transposes;
RHS variants include transposes at the root or either immediate child.

## 9. Turning the result into a conference proposal

A defensible working title is **"Attribute Realizability and Two-Axis
Obstructions for Ordered Tensor Rewrites."**

The proposed claim is narrow enough to audit: complete, constructive attribute
inference for independent-input ordered contractions, with a small obstruction
theorem preserving existential RHS attributes. Its practical benefit is replacing
factorial list enumeration with a structural algorithm and useful diagnostics.

For a strong paper, the next work should be:

1. Independently review and formalize the network exactness, forced-boundary,
   subsequence-construction, and restriction proofs.
2. Define a semantic contract for the admitted operations and prove a generic
   lowering from TEPL's element/index semantics into that contract.
3. Audit the implemented normal-form extraction, constructor, and separate
   certificate checks. Extend them only with sound admission proofs.
4. Evaluate against exhaustive search and the current generic engine on
   arbitrary-rank contractions, transposes, deliberate order failures, and
   renamed/custom-defined operators with the same semantic contract.
5. Extend one substantive dimension: addition/distributivity with coupled
   attributes, prescribed attribute restrictions, or minimum-cost transpose
   repair. Keep the original theorem's domain separate from each extension.

Measure compile-time rejection precision, witness/certificate sizes, runtime
construction cost versus rank and tree size, and rejection caused by exclusions.
Do not present the present exhaustive experiment as a performance benchmark.

### Novelty positioning

This note makes no claim that contraction boundaries, tensor-network equality,
or structural layout reasoning are new individually. Whether their combination
and this particular obstruction result are novel requires a focused literature
review and expert feedback.

- [TensorRight, POPL 2025](https://doi.org/10.1145/3704865) already derives rank
  bounds for verification in its aggregated-axis language. The distinction to
  investigate is a bound preserving *existential unknown RHS attributes* and
  ordered-axis realizability, with a constructive inference algorithm.
- [TASO, SOSP 2019](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf)
  already generates graph substitutions from operator specifications. "Automatic
  tensor rewrite synthesis" alone is not a novel framing.
- [Axon, 2026 preprint, Sections 4.4 and 5](https://arxiv.org/html/2606.26344v1)
  uses semantic specifications and sketch-driven ISA synthesis, with SMT
  equivalence checking for known ranks and symbolic sizes. The proposed
  distinction is a complete structural realizability criterion and bounded
  impossibility certificates in a stated rank-polymorphic fragment.

### Draft abstract

Tensor rewrite specifications often fix the target expression structure while
leaving operator attributes unknown. Verifying one candidate assignment does
not determine whether the target structure is realizable. We study this question
for ordered tensor contraction trees over independent inputs. We characterize
realizability using forced intermediate boundaries and subsequence constraints,
yielding a polynomial-time attribute constructor or a structural impossibility
certificate. We show that every unrealizable instance has an obstruction
involving two logical output axes. Restricting to these axes preserves failure
of every target attribute assignment and bounds each input rank by two. The
result supplies a finite verification procedure for universal coverage in the
unguarded fragment and explains when an output transpose repairs an otherwise
unrealizable contraction tree. A prototype compares the structural procedure
with exhaustive attribute enumeration on small instances. General semantic
lowering and extensions to additive expressions remain future work.

This abstract describes the draft mathematical development. The instance path
now has a restricted TEPL implementation; universal coverage and an independently
reviewed publication result remain future work.
