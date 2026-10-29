# A generic restriction that gives a computable rank cutoff

> Historical research material. Implementation references and reproduction
> instructions describe the earlier approach. See the
> [current synthesis design](../../../design/attribute-synthesis.md) and
> [research index](../README.md) for present scope.

This extends [the original problem](math_theorem.md). This research proposal is
not a coverage certificate for the current element-semantics implementation.

The proposed restriction is on the semantics of axis configurations and
attribute choices: they must admit an **exact finite-word, finite-state
representation**. Operation names play no role.

Under the explicit hypotheses below, there is an algorithm that returns a
finite cutoff and a soundness argument. It is a mathematical design for a new
verification fragment, not functionality implemented in TEPL today. It does not
claim that all existing IndexIR definitions belong to the fragment.

## 1. The restriction must cover the attribute quantifier

The failing condition is

\[
 \operatorname{Bad}(p)
 =\operatorname{Legal}(p)\land
   \neg\exists\beta\,\operatorname{Valid}(p,\beta).
\]

It is not enough to bound a counterexample to one *particular* assignment.
Deleting axes might invalidate that candidate while making another assignment
possible. A cutoff proof must preserve failure of **every** RHS assignment.

Nor is a finite set of axis roles sufficient. A finite alphabet can encode
nonregular constraints: the language `a^n b^n` requires comparing two unbounded
counts. Arbitrary matching or ordering attributes can introduce comparable
global constraints.

The restriction below addresses the whole existential relation before deriving
the bound.

## 2. Precise finite-state hypotheses

For a fixed rewrite `S`, require effective constructions of:

1. A finite alphabet `Sigma` describing one axis column. A configuration is a
   finite word `w` over this alphabet. Each input has at most one physical axis
   per column, so `rank_i(p) <= length(w)`.
2. A finite alphabet `Gamma` describing the RHS choices for one column.
   A candidate assignment is an annotation word `z` with the same length as `w`.
   The number of choices per column depends on `S`, not on the tensor ranks.
3. A deterministic finite automaton `L` for legal configuration words.
4. A finite automaton `V` over paired letters `(Sigma, Gamma)` such that its
   accepted words describe exactly the valid RHS assignments, including their
   legality, output shape, output order, and equality for all tensor values.

The encoding and decoding must have proved correspondence properties:

- Every legal concrete configuration is represented.
- Every accepted legal word is realizable by a legal concrete configuration.
- Every accepted annotation decodes to legal concrete attributes.
- The representation preserves attribute existence exactly. In particular,
  it cannot merge a solvable configuration with an unsolvable configuration.

If dimensions are kept symbolic, prove these properties for that symbolic
abstraction as well. Symbolic dimension variables do not automatically form a
finite alphabet. Their numerical values must be irrelevant to the existence
question, or handled by a proved finite classification.

These are stronger conditions than saying that a sound verifier exists.
A verifier that merely recognizes some valid assignments is not sufficient for
the counterexample claim: its rejection could mean incompleteness.

## 3. The cutoff theorem

Let `L` have `ell` states and let `V` have `v` states. First project away the RHS
annotations:

\[
 \operatorname{Good}
 =\{w\mid\exists z\in\Gamma^{|w|},\ (w,z)\in\mathcal L(V)\}.
\]

Projection nondeterministically guesses each annotation letter. It produces an
NFA with at most `v` states. Determinize it, including its rejecting sink if
needed; at most `2^v` states are required.

Complement that DFA and intersect with `L`:

\[
 \operatorname{Bad}
 =\mathcal L(L)\cap\overline{\operatorname{Good}}.
\]

The resulting DFA has at most

\[
 N=\ell\,2^v
\]

states. All RHS choices have been quantified out **before** this complement.

**Theorem.** Under the correspondence assumptions of Section 2, a sufficient
componentwise rank cutoff is

\[
 \boxed{K_i=\max(1,\ell\,2^v-1)\quad\text{for every input }i.}
\]

A smaller bound uses the actual number `N_B` of reachable states of the
constructed bad-configuration DFA:

\[
 K_i=\max(1,N_B-1).
\]

**Proof.** Suppose an unsolvable configuration exists. Exact correspondence
makes the bad-word language nonempty. Choose a shortest accepted word `w`.

Its accepting run cannot repeat a state. Otherwise, removing the segment
between two equal states gives a shorter accepted word. Consequently
`length(w) <= N_B - 1`.

Acceptance ensures that this smaller word still describes a legal configuration
and that **no** RHS annotation works for it. Decoding gives an unsolvable
configuration with every input rank at most `length(w)`. Thus a failure at any
rank implies a failure inside the stated bound. Taking the contrapositive gives
the requested finite-rank verification implication. ∎

If a column may contain at most `c_i` axes of input `i`, and there are `d_i`
fixed axes outside the word, use

\[
 K_i=\max(1,d_i+c_i(N_B-1)).
\]

The constants and automata come from the semantic translation. This is a
computable bound, not a rank guessed by experimentation. The construction can
be expensive; computability does not imply a small or practical cutoff.

### A useful difference from axis-deletion proofs

No assumption that arbitrary axis deletion preserves guards is required here.
The legal-language automaton already includes the guards. The repeated-state
argument removes only a segment that preserves both legality and unsolvability.

For example, a fixed minimum-rank guard can be represented by more automaton
states. It increases the bound rather than silently violating its proof.

## 4. A checkable language for producing the automata

A concrete way to ensure the hypotheses is to translate the supported axis
semantics into **monadic second-order logic over finite words**, MSO.

Its variables range over positions and sets of positions. Available relations
include position order, successor, equality, and finite letter predicates.
It does not quantify over arbitrary binary matching relations.

In this representation:

- One word position represents an axis column.
- Known attributes supply finite predicates describing each column.
- Unknown axis sets become quantified set variables.
- A finite choice of axis role becomes a partition into finitely many sets.
- Disjointness, coverage, relative order, and fixed finite layout phases become
  formulas over positions and sets.

Compile

\[
 \operatorname{Legal}(w)
 \land\neg\exists X_1\cdots\exists X_m\,
       \operatorname{Valid}(w,X_1,\ldots,X_m)
\]

into a finite automaton, then apply Section 3.

The effective connection between word logic and automata, including projection
for existential set quantification, is standard. See the
[MONA manual, Section 3.1](https://www.brics.dk/NS/01/1/BRICS-NS-01-1.pdf).
The tensor-specific work is proving that the semantic translation is exact.

This gives a syntactic admission rule: accept definitions and constraints that
can be lowered through the certified fragment. Do not attempt to decide whether
an arbitrary semantic program secretly has an equivalent MSO description.

## 5. A conservative tensor fragment for the first exact translation

The following restrictions are a concrete starting point for that translation.
They deliberately restrict the interfaces of operations, not their names.

### 5.1 Scalar computation

After expanding operation semantics, require a single finite sum of a product
of independent tensor reads. Each input tensor appears once. Every bound index
must occur in at least one read.

Use exact real arithmetic. Exclude scalar data-dependent branches, arbitrary
functions, repeated tensor factors, and multiple additive terms from this first
kernel. Those extensions need their own normalization and completeness proofs.

This restriction covers pure contraction trees and unary sum reductions.
It does not yet cover the full addition/distribution fragment needed for LoRA.

### 5.2 Index computation

Each axis is attached to an origin column. At a column there are only finitely
many ports, determined by the fixed rewrite tree. Attribute choices may retain,
pair, broadcast, or sum those ports.

Pairing is between ports at the same column. A general unknown permutation
mapping one column to an arbitrary other column is not allowed. Distinct
bound-variable classes within a column remain distinct; merely sharing a column
does not equate their coordinates.

Semantics may pass a coordinate through unchanged or omit it for broadcasting.
They may not compute coordinates such as `i+j`, use data-dependent indexing, or
flatten/unflatten dimensions.

The resulting connectivity at one column is a partition of a fixed finite set
of ports. The set of possible partitions is finite. This is what makes the
annotation alphabet finite even when the rank is unbounded.

### 5.3 Layout computation

Input and intermediate layouts must be expressible by a fixed finite recipe
from the origin word, using operations such as:

- Selecting positions with a finite role predicate.
- Concatenating a fixed number of disjoint selected subsequences.
- Reversing a selected subsequence.
- Assigning positions to a fixed number of layout buckets and ordering buckets.

The recipe size may depend on the rewrite, but cannot grow with rank. The recipe
must neither duplicate nor drop an axis unless the semantic operation explicitly
permits that behavior.

These layouts have MSO-definable membership and relative-order relations. For
example, a stable partition into `batch`, `left-free`, and `right-free` makes
`x` precede `y` when its bucket is earlier, or their buckets agree and `x < y`.

Consequently, output order can be compared exactly: require the same output
origins and require the two order relations to agree on every pair. This checks
all axes, not a fixed sample of their positions.

### 5.4 Shapes

Start with nondegenerate dimensions, at least 2, and dimensions attached to the
index classes they measure. Shape reasoning may enforce equal extents for paired
ports and preserve the ordered list of surviving dimension origins.

Do not permit branches on arbitrary dimension arithmetic, unrelated cross-column
dimension comparisons, or products of dimension sizes. The finite abstraction
must preserve the particular equality classes used for legal pairing.

Broadcasting can omit coordinates of axes introduced from another input's
existing dimension origins. Every logical index should occur in an input read;
reducing an index absent from all reads creates a dimension factor and is outside
this initial kernel. Standalone introduction of arbitrarily many shape-only axes
needs an explicit encoding and rank-accounting extension.

Singleton expansion and empty dimensions require separate cases and an exact
abstraction proof. They must not be silently classified as equivalent to general
dimensions.

### 5.5 Guards and unknown attributes

Guards must also lower to the same word logic. Examples include disjoint axis
sets, a fixed required number of distinguished axes, and ordering constraints.
Arbitrary host predicates do not qualify.

Unknown attributes are axis sets, local role choices, and the permitted layout
bucket assignments. Concrete lists are recovered by enumerating selected axes
in the chosen layout order.

Arbitrary bijections, comparisons of two unrelated unbounded axis counts, and
unbounded integer arithmetic do not qualify merely because their individual
values are finite at match time.

## 6. Why tensor-value equality can disappear from the formula

For the single-product, independent-read fragment, assign independent formal
variables to tensor entries. For a fixed output index, the finite contraction
is a polynomial.

Because every summed index appears in a read, distinct assignments to the
summed index classes produce distinct tuples of input entries. Monomial
coefficients are therefore 1. Equality for all tensor values is equivalent to
equality of the allowed tuples of input coordinates at every output coordinate.

Those allowed tuples specify exactly the index-equality partition and the
attachments of free indices to output ports. With extents at least 2, distinct
classes can be separated using coordinates 0 and 1. This proves that a mismatch
in these relationships is detectable by tensor values.

For the restricted index language, the partitions are finite local data at each
column. Output membership and order are given by the layout formulas. Comparing
these objects is therefore expressible in word logic. No quantifier over actual
tensor elements is left in the automaton.

The admission checker must also establish the scalar normal form, valid reduction
domains, and shape correspondence. An automaton for axis annotations alone is
not a proof of tensor equivalence.

## 7. What this includes and excludes

| Operation description | Admission in the first fragment |
| --- | --- |
| `dot_general` finite sums/products | Yes when pairing and layout attributes obey the restrictions above. |
| Unary sum reduction | Yes when summed indices occur in reads and the resulting axis layout is admitted. |
| Broadcast along existing dimension origins | Yes under the index and shape restrictions above. |
| Transpose by a fixed finite layout recipe | Yes. |
| Completely arbitrary permutation attributes | Not automatically; their matching relations are not unary axis annotations. |
| Reshape, affine indexing, data-dependent access | Outside this fragment. |
| Addition, repeated tensor factors, `reduce_max` | Require further exact semantic normalization theorems. |

Arbitrary transpose composition can still be proved separately by the generic
algebra of bijections, as in the earlier note. That does not establish that every
rewrite combining unrestricted permutations has this finite-state encoding.

Similarly, unrestricted `dot_general` axis lists are not automatically covered
by calling their operation a contraction. The restrictions on correspondence,
layout, and shape must be enforced or proved for the actual definitions and
attribute domain.

## 8. Automatically deriving a guard for a particular rewrite

Restricting the semantic language makes the verification problem decidable;
it does not make every rewrite true. These are two different restrictions.

The same construction can derive a configuration guard:

\[
 G_S(w)\quad\Longleftrightarrow\quad
 w\in\mathcal L(L)\cap\operatorname{Good}.
\]

This is an effectively constructed finite automaton. Within the admitted
domain, it accepts exactly the configurations for which attributes exist.
Therefore the guarded rewrite is valid, and the original unprojected automaton
can recover attributes for each accepted match.

This guard is maximal within the domain: it excludes no solvable configuration.
It need not simplify to a short source-level predicate. If its language is empty,
report that the rewrite has no applicable configuration; an always-false guard
is not a useful successful rewrite.

Alternatively, a user can supply a guard `H` in the supported logic. Check

\[
 \mathcal L(L)\cap\mathcal L(H)\subseteq\operatorname{Good}.
\]

This is another automaton emptiness problem, with a cutoff derived from its
states. Thus both admission restrictions and rewrite-specific guards can be
handled generically, without a handwritten guard for each operation name.

## 9. Algorithm and proof obligations

```text
Input: rewrite, operation semantics, supported attribute domains, guards

1. Expand semantics and prove the admitted scalar/index normal form.
2. Build a finite-word encoding and prove its correspondence with configurations.
3. Translate legality and candidate validity into word logic.
4. Compile legality and validity to automata.
5. Existentially project RHS choices, complement, and intersect with legality.
6. Count reachable states; return the derived rank bound.
7. Check whether any bad configuration is accepted.
```

The proof artifact should contain the normal-form derivation, encoding
correspondence, logic-to-automata construction, rank-accounting constants, and
the shortest-counterexample theorem instantiated with the resulting state count.
Checking these certificates can use a small trusted proof kernel. A state count
alone is not a semantic certificate.

The automaton can decide emptiness directly; enumerating ranks through the
possibly huge cutoff is unnecessary. If SMT is used for bounded checking,
it must discharge the exact quantified problem, including all annotations and
legal symbolic configurations. A timeout remains `unknown`.

At match time, an accepting path through the unprojected validity automaton can
recover an annotation word, which is then decoded into RHS axis lists. This
constructs attributes for the finite-state representation. Its cost is linear
in word length for a fixed automaton, with a constant depending on its states
and transitions; building that automaton may be prohibitively expensive.

## 10. The answer to the generic-restriction question

A sufficient generic restriction is:

> Require an exact translation of configurations and candidate validity into
> finite-word MSO, with a proved correspondence to tensor semantics and a rank
> bound per word position.

This gives a computable cutoff and a proof by automata projection,
complementation, and shortest accepted words. It handles the existential RHS
attributes, rather than proving only a particular candidate correct.

The restricted scalar, index, shape, and layout language above is a concrete
starting point for obtaining that translation from semantic definitions. It
requires generic rules for mathematical primitives, not handwritten rules for
each named tensor operation or rewrite tree.

Completeness is claimed only for definitions whose exact translation is proved.
Extending that domain is a separate mathematical task; a broad list of familiar
operation names is not a substitute for the translation proof.
