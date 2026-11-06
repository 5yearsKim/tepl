# TEPL: Synthesizing Tensor Attribute Programs from Rewrite Semantics

**Research draft. Revised November 5, 2026.**

## Abstract

Tensor rewrite rules are often short algebraic identities, but their replacements
carry attributes that depend on source shapes and attributes: permutations,
contraction lists, reduction axes, broadcasts, and similar metadata. Compiling
such rules therefore requires both a semantic correspondence and a reusable
metadata program. TEPL is a tensor rewrite language in which dialects declare
operation axes and element semantics, rules declare patterns and applicability
conditions, and RHS attributes can be written explicitly or marked for inference.
We describe a synthesis method that constructs symbolic coordinate correspondences,
normalizes component semantics, enumerates admissible axis descriptors, and recovers
a witness that is executable for all compatible source metadata.

We distinguish three guarantees. First, guarded semantic verification establishes
that accepted, completed rewrites are legal and preserve their declared semantics.
Second, a bounded axis-attribute construction has an exact feasibility and witness
property for a precisely stated signed-axis fragment. Third, a broader structural
extension admits supplied metadata, positive fresh axes, and weighted signed
wiring, again under explicit domain restrictions. These are mathematical
procedures and finite checks, not claims of completeness for every TEPL rule or
formal verification of the C++ implementation.

## 1. Introduction

Algebraic rewrite engines operate on graph structure. Tensor operations additionally
require attributes that describe coordinate interfaces. For example, a dot product
can reassociate only if the batch and contraction axes are reordered consistently
in both operands and in the replacement result. Equal dimensions do not establish
that two physical axes represent the same logical coordinate: TEPL therefore
retains axis identities, positions, and ordered mappings.

This creates an authoring problem. A compiler can match
\(\operatorname{dot}(A,B)\), but it must still construct the replacement operation's
permutation and contraction lists. Writing those transformations by hand duplicates
reasoning already present in operation definitions and is brittle across ranks and
axis layouts.

TEPL separates four concerns:

1. **Operation semantics** defines axis bindings, output order, legality, and
   coordinate-indexed element values.
2. **Rewrite patterns** define graph structure and captured tensors.
3. **Conditions** restrict source metadata and may exclude invalid matches.
4. **Attribute construction** supplies replacement descriptors explicitly through
   expressions or constructs omitted descriptors automatically.

Our central technical observation is that coordinate correspondence used to prove
equivalence can also be constructive. If the source and replacement read the same
input coordinates in a semantic component, the correspondence identifies how logical
axes move through each operation. TEPL converts this correspondence into an
executable metadata function rather than a concrete attribute value.

## 2. Problem

Let \(s\) contain source shapes and operation attributes, \(X\) a tensor value
tuple compatible with those shapes, and \(a\) replacement attributes. Let
\(\operatorname{LHS}(s,X)\) and \(\operatorname{RHS}(s,a,X)\) denote the two
computations and \(C(s)\) a metadata guard.

A synthesized constructor is a program
\[
f:s\longrightarrow a
\]
that depends only on source metadata. An accepted rewrite must satisfy, for every
legal source instance satisfying the guard,
\[
s\in\operatorname{dom}(f)
\quad\land\quad
\operatorname{LegalRHS}(s,f(s))
\quad\land\quad
\operatorname{ShapeRHS}(s,f(s))
=\operatorname{ShapeLHS}(s),
\]
and
\[
\forall X,\quad
\operatorname{RHS}(s,f(s),X)
=\operatorname{LHS}(s,X).
\]

The equality obligation is relative to the selected arithmetic policy. Exact-real
equivalence does not establish strict floating-point reassociation. For incomplete,
unknown, or resource-bounded reasoning, TEPL rejects the rule as unproved; failure
does not imply that the rewrite is semantically false.

## 3. Language and Axes IR

### 3.1 Operations

A TEPL dialect defines operations once and reuses them in rewrites. Operation
definitions contain attributes, axis relationships, output shape and dtype rules,
and element semantics. Axes are declared as ordered symbolic sequences so that one
definition can describe operations at arbitrary rank.

For example, a contraction operation groups batch, contraction, and free axes:

```tepl
op dot_general(lhs: tensor, rhs: tensor) -> tensor {
  attrs {
    lhs_batching_dimensions: axis[];
    rhs_batching_dimensions: axis[];
    lhs_contracting_dimensions: axis[];
    rhs_contracting_dimensions: axis[];
  }

  axes {
    let lb = attrs.lhs_batching_dimensions;
    let rb = attrs.rhs_batching_dimensions;
    let lc = attrs.lhs_contracting_dimensions;
    let rc = attrs.rhs_contracting_dimensions;
    let lf = ordered_complement(all_axes(lhs), lb ++ lc);
    let rf = ordered_complement(all_axes(rhs), rb ++ rc);

    bind B { lhs: lb, rhs: rb };
    bind K { lhs: lc, rhs: rc };
    bind I { lhs: lf };
    bind J { rhs: rf };
    yield B ++ I ++ J;
  }

  semantics {
    yield Reduce(K, Product(lhs, rhs));
  }
}
```

Axis positions are zero-based. Binding two operands to one group pairs corresponding
axes in list order. Thus contraction attributes may pair different physical
positions while preserving logical coordinate order.

### 3.2 Rewrite rules

A rewrite pattern identifies operation structure and captures tensor operands:

```tepl
rule transpose_transpose {
  lhs: transpose(transpose(X, inner), outer)
  rhs: transpose(X, derived)
  where: no_other_users(X)
}
```

The actual language supports guarded rules, explicit axis expressions, repeated
captures, capture reuse, and omitted attributes marked by inference. Attributes
may also be computed in `derive` blocks. Automatic synthesis and explicit
construction use the same acceptance contract: the completed RHS must be legal,
the output interface and dtype must agree, and semantic verification must succeed.

### 3.3 Symbolic coordinate states

Axes IR models four related concepts:

* an ordered tensor interface \(P=\langle a_0,\ldots,a_{n-1}\rangle\);
* a symbolic ordered group \(G=\langle g_0,\ldots,g_{m-1}\rangle\);
* a position sequence \(p=\langle p_0,\ldots,p_{m-1}\rangle\); and
* a mapping \(\operatorname{gather}(P,p)=G\).

Composed operations establish mappings such as
\(\operatorname{transpose}(P,p)=Q\), and TEPL composes them through common logical
axis groups. These mappings remain symbolic until a concrete match supplies rank
and metadata. Immutable expression nodes and structural interning reduce repeated
comparison cost.

### 3.4 Value semantics

The `axes` block describes coordinate correspondence; `semantics` describes the
value at each output coordinate. Value expressions include constants, input reads,
addition, multiplication, and reduction. Calculated index expressions support
coordinate selection, scaling, offset, and concatenation.

Semantic lowering turns operation trees into symbolic scalar terms over operand
reads. Normalization groups coefficients, reads, and reduction domains, enabling
syntactic and algebraic comparison without enumerating every tensor coordinate.

## 4. Synthesis Method

### 4.1 From correspondence to attributes

Attribute synthesis uses one semantic graph for values and a separate application
graph for rule matching. Operation axes define mappings between their operands and
outputs; producers and consumers connect captures through those mappings.

The compiler proceeds as follows:

1. Lower the source and candidate replacement to symbolic value expressions.
2. Normalize supported scalar terms, preserving ordered reduction domains.
3. Establish coordinate correspondences between captured input reads.
4. Generate legal local descriptor choices for axis-bearing RHS nodes.
5. Replay child witness states through the operation tree.
6. Match a source network or semantic state at the replacement root.
7. Recover one witness and emit a reusable metadata program.
8. Verify legality, ordered shape, dtype, guard evaluation, and value preservation
   for the completed rule.

A witness is not a tensor result. It is a descriptor assignment together with
symbolic state and arithmetic coefficients sufficient to replay the construction.

### 4.2 Ordered mappings

For arbitrary-length axis groups, position lists can be composed:
\[
p_{AB}=p_B[p_A].
\]
Concatenation partitions groups into components, complements produce groups in
operand order, and inverse mappings recover positions in the original interface.
These operations must retain duplicate-free and in-range constraints.

The implementation stores mappings by tensor interface and axis group, with reverse
lookups by interface and position sequence. Partition overlap records selected
subgroups and their positions in each parent. Cache invalidation occurs when a
mapping or established reasoning fact changes.

### 4.3 Partition refinement

Composed operations may introduce alternative relationships among logical axes.
TEPL refines these relationships through partitions rather than concrete ranks.
A partition records equivalence classes of axis origins; edges record signed or
ordered correspondences where required. Registering a producer-consumer mapping
may merge groups, split them into free and contracted subsets, or expose overlaps.

All choices remain finite for an admitted finite operation tree and axis contract.
The current method is syntax-directed and bounded: it may be exponential in the
number of axis choices and states, and unresolved or budgeted searches are rejected
rather than applied heuristically.

## 5. Exact Axis-Attribute Fragment

The following fragment has constructive feasibility and completeness results.

Operations are generalized contraction, full permutation, whole-axis reversal,
unary summation, fixed positive-arity aligned weighted scalar products, and
separated elementwise addition. Tensors have exact real entries and arbitrary finite
ranks; extents are metadata. The scalar product fragment is
\[
h(x_1,\ldots,x_k)=q_h\prod_i x_i,
\qquad q_h\in\mathbb Q\setminus\{0\}.
\]
Broadcast, slice, reshape, pad, concatenate, singleton and zero extent assumptions,
extent-bearing descriptors, coordinate offsets, and general tensor coefficients are
outside this specialization.

The source may be expanded by linearity and bilinearity. The RHS must contain the
same distinct input sets in corresponding separated additive components. Each input
occurs once within a component, and every RHS descriptor is an independent axis hole.

Let \(\Sigma\) specify source and fixed RHS structure, \(p\) supply source metadata,
and \(\beta\) be a candidate RHS descriptor assignment. Define
\[
\operatorname{Valid}_\Sigma(p,\beta)
\]
to require RHS legality, source ordered shape, and equality for every admitted input
tensor value.

### Theorem 1: Complete constructive synthesis

For every admitted specification and legal in-domain metadata, a terminating
generator produces \((\Phi_\Sigma,f_\Sigma)\) such that
\[
\Phi_\Sigma(p)
\iff
\exists\beta\,\operatorname{Valid}_\Sigma(p,\beta),
\]
and, on success,
\[
\operatorname{Valid}_\Sigma(p,f_\Sigma(p)).
\]
The same metadata program handles every finite rank in the admitted domain.

The constructor enumerates all legal canonical axis choices at each node, stores one
realizing witness per network state, and combines witnesses only after component
equivalence is established. For contraction, permutation, reversal, and summation,
choice counts are finite and descriptor legality is checked locally. For aligned
weighted scalar operations, the procedure uses finite semantic states and exact
rational coefficients. Its cost can be combinatorial, but it does not enumerate
tensor coordinates.

The complete argument is given in
[the unified completeness proof](proofs/completeness.md). Its finite checks include
small-instance oracles for oriented, pointwise, and supplied-axis synthesis:

```sh
python3 research/attribute-synthesis/scripts/check_oriented_axis_synthesis.py
python3 research/attribute-synthesis/scripts/check_pointwise_axis_synthesis.py
python3 research/attribute-synthesis/scripts/check_supplied_axis_synthesis.py
```

These scripts support the mathematical claims but are not production synthesis or
formal proof checkers.

### General structural extension

A separate theorem admits metadata-weighted signed wiring in which identity or
reflection identifies whole axes, entire domains may be summed, surviving axes may
be reordered, and fresh positive axes may be introduced. A nonzero rational factor
may be determined by semantics, supplied metadata, child shapes, and selected axes.
Only axis slots are unknown; extent, coordinate, offset, stride, and general integer
or unsigned fields are supplied, defaulted, or computed but not inferred.

This extension includes positive-extent broadcast, unit-dimension insertion and
removal, keepdims summation, and full-axis mean. Its completeness theorem is about
the specified structural procedure, not about arbitrary convolution, slice, reshape,
padding, dynamic indexing, or strict floating-point arithmetic.

Convolution, general reshape, pad, concatenate, mixed descriptors, repeated captures
within additive components, and value-dependent rearrangement require separate
adapters or semantic representations. They are not admitted merely by supplying
their attributes.

## 6. Soundness and Verification

Verification and synthesis are separate obligations. Soundness requires a faithful
operation lowering, sound local algebraic lemmas, metadata-program termination and
definedness, RHS legality, ordered output shape, dtype agreement, and value
preservation.

Let \(G(p)\) be the accepted applicability predicate, including the user guard and
every additional metadata obligation. Let \(f(p)\) construct an attribute tuple.
Under successful acceptance, the replay argument establishes:
\[
G(p)\Rightarrow
\operatorname{LegalRHS}(p,f(p))
\land
\operatorname{ShapeRHS}(p,f(p))
=\operatorname{ShapeLHS}(p)
\land
\forall X,\operatorname{RHS}(p,f(p),X)
=\operatorname{LHS}(p,X).
\]

Local rules cover tensor reads at equal coordinates, scalar congruence, ordered and
reindexed reductions, reduction identities, distribution under stated side
conditions, and replacement in a pure context. Partial functions require explicit
definedness premises: nonzero denominators, in-bounds coordinates, and matching
reduction domains cannot be inferred from equal shapes alone.

This is soundness of guarded acceptance, not a proof that every compiler routine
matches it exactly. The implementation-correspondence obligation remains explicit:
each lowering, normalization, legality check, constructor, and evaluator must
implement the mathematical rule attributed to it.

## 7. Implementation

TEPL is implemented in C++20 with ANTLR parsing. Compilation resolves names,
checks types, lowers declarations, synthesizes or checks attributes, verifies the
completed rewrite, and exports an executable representation. Compilation pipelines
separate:

```text
TEPL source
  -> parsing
  -> name resolution and type checking
  -> synthesis or explicit construction
  -> semantic verification
  -> executable rewrite program
  -> JSON export
```

The runtime or generated engine loads the compiled program, finds LHS matches,
checks guards and metadata programs, constructs the candidate RHS, verifies
operation contracts, and applies the replacement. Rewriting can continue to a
fixed point or stop at a configured limit.

Pattern captures record graph nodes, shapes, and source attributes. A repeated
capture must identify the same node. Attribute expressions and synthesized
programs are evaluated from these bindings. Failed guard evaluation, invalid
construction, or operation-contract violation skips the match.

Internal expression nodes distinguish axes, positions, extents, integers, and
Booleans. Value semantics use constants, operand reads, addition, multiplication,
and reductions. Mapping and reasoning caches accelerate repeated queries while
preserving structural keys for collision-safe lookup.

## 8. Evaluation Status

The prospective XLA experiment adapts 209 concrete rewrites from the XLA algebraic
simplifier and separates semantic coverage from explicit verification and
automatic inference.

| Stage | Target result | Interpretation |
|---|---:|---|
| Semantic coverage | 118/209 | Every required operation has a usable declaration |
| Explicit verification | 86/118 | Candidate attributes supplied or derived explicitly |
| Axis/extent inference | 20/20 | `auto` completion independently reverified |

The remaining cases do not establish rewrite correctness or incorrectness. Missing
operation semantics, unsupported coordinate reasoning, additional applicability
conditions, and resource limits all affect the result. The counts are conditional
on the recorded compiler revision and arithmetic policy and should not be read as
new support claims for XLA as a whole.

Historical and current measurements belong in `labs/`; the relevant entry point is
[the XLA rewrite lab](../../labs/attribute_inferences/xla_rewrite/README.md). Compiler and source hashes
and reproduction commands must accompany any reported number.

The finite synthesis scripts validate representative ranks and metadata layouts,
while the reasoning scalability lab measures rank-polymorphic chain and additive-term
compilation and metadata execution. These experiments do not establish broad
runtime guarantees, and correctness must continue to be measured separately from
authoring coverage.

## 9. Limitations and Future Work

Automatic coordinate reasoning does not currently cover general reshape, dynamic
slice, coordinate-dependent pad and concatenate, padding-aware convolution layouts,
mixed value-dependent metadata, or broad strict floating-point reassociation.
Structural rewrites can still be written and manually guarded in TEPL, but they
do not receive the semantic-equivalence guarantee of verified rules.

The synthesis theorem is intentionally narrower than the language. Candidate
generation can be complete in a defined fragment while remaining heuristic for
operations outside it. Conversely, sound semantic verification can accept rules
that do not originate from the complete generator. A resource failure is not a
completeness result.

Future work includes:

* efficient rank-polymorphic representations for larger axis partitions;
* stronger implication checking for user guards;
* adapters for slice, reshape, convolution, padding, and concatenation;
* arithmetic-policy-specific floating-point validation and diagnostics;
* independent literature comparison with tensor rewrite and equality-saturation
  systems; and
* benchmark protocols that report source hashes, coverage denominators, resource
  limits, and reproduction commands.

## 10. Related Work

TensorRight uses aggregated axes to verify tensor graph rewrites over arbitrary
ranks. TEPL's ordered symbolic-axis representation builds on the same broad need
for rank-polymorphic reasoning while retaining axis order, position mappings, and
constructive witnesses for replacement attributes. TensorRight is prior work for
aggregated-axis verification and does not by itself establish TEPL synthesis
novelty. [TensorRight, POPL 2025](https://doi.org/10.1145/3704865)

TASO and other tensor rewrite systems generate or apply optimized rewrite artifacts,
and equality-saturation systems use semantic or structural equivalence classes.
A complete comparison must distinguish semantic assumptions, attribute handling,
synthesis targets, and verification contracts. Index notation and affine compiler
IRs are also related because both represent coordinate access, but TEPL derives
operation descriptors through declared tensor interfaces rather than treating
attribute updates as host-language callbacks.

The documented differentiation is compositional synthesis of reusable attribute
programs from user-defined tensor semantics. It does not claim to be first until
systematic literature review and implementation-independent evidence support such
a claim.

## 11. Conclusion

TEPL treats tensor rewrites as semantic objects rather than graph substitutions
alone. Operation definitions expose coordinate correspondence and element semantics;
rules expose patterns and applicability conditions; and verification connects the
source and replacement with an executable metadata contract.

For a bounded axis fragment, coordinate states give constructive, rank-polymorphic
attribute witnesses and an exact feasibility result. Outside that fragment, TEPL
can still support explicit attribute construction and structural rewrites, but it
only accepts semantically verified rewrites after the corresponding legality and
equivalence obligations are discharged. The key result is therefore not a universal
automatic compiler: it is a precise synthesis and verification discipline that
distinguishes proved attribute programs from heuristic candidate generation.

## Reproduction and Artifacts

* Implementation design:
  [Attribute synthesis](../../design/attribute-synthesis.md)
* Unified completeness argument:
  [proofs/completeness.md](proofs/completeness.md)
* Verification discipline:
  [proofs/soundness.md](proofs/soundness.md)
* Evaluation indexes:
  [research collection](../README.md) and
  [lab index](../../labs/README.md)
