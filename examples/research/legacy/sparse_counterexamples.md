# Sparse counterexamples for compile-time attribute rejection

**Status:** theorem and research proposal, October 11, 2026. A bounded
[explicit-attribute early checker](early_verification.md) now implements degree
analysis, sparse Z3 queries, confirmed witnesses, and Rust host-function
validation. The `auto` synthesis/rejection loop proposed below is not implemented.
The proofs assume ideal mathematical arithmetic and do not claim research novelty.

## Goal

Reject a rewrite at compile time when some legal input configuration admits no
correct RHS attribute assignment. Derive the analysis from operation semantics,
without recognizing operation names or specific rewrite patterns.

Let `p` contain input ranks, shapes, and LHS attributes; `beta` contain RHS
attributes; and `X` contain all input tensor values. A universally valid rule
requires:

\[
\forall p\text{ satisfying the rule's conditions},\quad
\exists\beta\text{ legal}:\quad
\forall X,\ L(p,X)=R(p,\beta,X).
\]

To reject this claim, find one legal configuration `p0` for which every legal
`beta` fails on some input. Different attributes may need different inputs.
All examples in one rejection certificate must share `p0`: attributes may
change when metadata changes.

## Supported semantic fragment

Start with tensor reads, metadata-dependent index maps, rational constants,
addition, subtraction, multiplication, and finite summation. Shapes, iteration
domains, and branches must be independent of tensor values. Inputs have
arbitrary rational or real values, with no value-dependent rule guards.

For fixed metadata and attributes, each output entry is a polynomial in the
input entries. This includes suitable semantic definitions of addition,
contraction, transpose, convolution, and their compositions. Repeated tensor
inputs are allowed; repeated reads refer to the same polynomial variables.

Value-dependent division, comparisons, `max`, and transcendental functions are
outside this initial fragment. IEEE floating-point behavior is also outside
the proof. A general reduction requires analysis of its combining expression;
the degree rule for summation does not apply to an arbitrary reduction.

## Derive a degree bound from IR

Analyze primitive expressions after composing intermediate tensor semantics:

| Expression | Polynomial-degree upper bound |
| --- | --- |
| Constant independent of tensor values | 0 |
| Original input tensor read | 1 |
| `a + b`, `a - b` | `max(degree(a), degree(b))` |
| `a * b` | `degree(a) + degree(b)` |
| Finite `sum(...) { a }` | `degree(a)` |

Use a bound `D` covering both sides and all legal candidate RHS attributes for
the selected metadata. Metadata-dependent branches require a bound covering
every admitted branch. Contraction of two inputs has degree at most two;
contraction of three inputs and the usual LoRA expression have degree at most
three. The compiler obtains these facts from the primitive IR.

## Sparse witness theorem

Fix legal metadata and RHS attributes, and suppose output shapes agree. If the
two sides have degree at most `D` and differ, there is a distinguishing input
with at most `D` nonzero scalar entries across all input tensors. Each entry can
be chosen from `{1, ..., D}`.

**Proof.** Some output coordinate has a nonzero difference polynomial `P`.
Choose a monomial with nonzero coefficient and let `S` be its variable support.
Then `|S| <= D`. Setting all variables outside `S` to zero preserves that
monomial, so the restricted polynomial remains nonzero. A polynomial with
degree at most `D` in each variable cannot vanish on the entire Cartesian grid
`{0, ..., D}^|S|`, by repeated univariate interpolation. A nonvanishing grid
point provides the desired input. For `D = 0`, the all-zero input suffices.

The support bound is independent of tensor rank and element count. It does
**not** bound the ranks or shapes needed to expose a bad rewrite.

### Consequence for attribute existence

For fixed metadata with `N` input entries, let `H_D` contain all inputs with at
most `D` nonzero entries, with values in `{1, ..., D}`. Among legal attributes
whose RHS output shape matches the LHS:

\[
\exists\beta\ \forall X:\ L(p,X)=R(p,\beta,X)
\quad\Longleftrightarrow\quad
\exists\beta\ \forall X\in H_D:\ L(p,X)=R(p,\beta,X).
\]

The forward implication is immediate. The reverse follows because any
incorrect candidate would have a witness in `H_D`.

This is an exact finite test family for the selected metadata, not random
testing. Its size is at most
`sum(s = 0..min(D,N), binomial(N,s) * D^s)` and can be large. Generate useful
members lazily instead of enumerating the family.

## Construct the inputs

Two complementary methods fit TEPL's semantic pipeline.

### Symbolic sparse slots

Allocate `D` slots. Each holds an enabled flag, a tensor identifier, coordinates,
and a value. Require in-bounds coordinates, distinct entries among enabled
slots, and values in `{1, ..., D}`. Unspecified entries are zero.

Lower reads to conditionals selecting a matching slot. For a candidate `beta`,
ask the solver for slots and an output coordinate where the sides differ.
Coordinates and values are solver variables; no dense tensor is allocated.

This representation can make index routing constraints small. Arbitrary
reductions and nonlinear arithmetic can still be expensive. Sparse storage
alone does not guarantee efficient evaluation or solving.

### Extract a monomial from a polynomial difference

If the existing polynomial verifier produces a nonzero difference, select one
monomial, zero all entries outside its support, and search the remaining
values on `{0, ..., D}`. The theorem guarantees success.

For example:

```text
P = A[0,0] * B[1,0] - A[0,1] * B[0,0]

A[0,0] = 1
B[1,0] = 1
all other entries = 0

P = 1
```

The main cost is obtaining the polynomial difference, which may require a
large expansion. Failure to normalize an expression is not proof that it
differs; unsupported analysis returns `unknown`.

## Compile-time algorithm

### 1. Search metadata

Choose bounds `K` on input ranks and `M` on dimension sizes. Generate rank
assignments and use SMT to find shapes and LHS attributes satisfying all source
legality conditions and rule guards. Fix one resulting configuration `p0`.

After analyzing a configuration, block that exact configuration before asking
for another. Progressively increase `K` and `M`. Include zero-size dimensions
where the dialect permits them. Unknown guards cannot be ignored.

These bounds control search effort. No general rank or shape cutoff is proved
here. Exhausting a bounded search does not establish universal correctness.

### 2. Alternate attribute synthesis and counterexample construction

```text
examples = []
repeat within the resource budget:
    solve for beta satisfying:
        RHS legality and matching output shape
        LHS(X) == RHS(beta, X) for every X in examples

    if UNSAT:
        return rejection certificate for p0
    if solver result is unknown:
        return unknown for p0

    verify this beta using the semantic IR
    if equivalence is proved:
        return valid for p0 only
    if a sparse counterexample is found:
        add it to examples
    otherwise:
        return unknown for p0
```

One example may eliminate many attribute assignments. The attribute solver
represents those assignments symbolically; explicit enumeration is not
required. A loop implementation must distinguish verified equivalence from
failure to find a counterexample within a budget.

### 3. Check the rejection certificate

A certificate contains `p0`, sparse inputs `X1, ..., Xm`, and an UNSAT proof of:

\[
\operatorname{LegalRHS}(p_0,\beta)
\land \operatorname{SameOutputShape}(p_0,\beta)
\land \bigwedge_i L(p_0,X_i)=R(p_0,\beta,X_i).
\]

Checking also establishes source legality, rule conditions, and correct
lowering of each example constraint. UNSAT is conclusive only if the encoding
covers every legal attribute assignment, including shared fields and pins.
Arbitrarily truncating an attribute domain cannot justify rejection.

If the attribute domain is finite and both solving and candidate verification
are complete, the loop terminates: each counterexample removes at least the
current incorrect assignment. Validity at `p0` or impossibility at `p0` is then
decided. This does not establish a practical runtime bound or a global metadata
cutoff. Unbounded scalar attribute domains need a separate treatment.

## Example: an incorrect LoRA rewrite

Consider a rule that drops the `XW` contribution:

```tepl
(outer: dot X (add W (inner: dot A B)))
=>
(dot (dot X A) B)
```

Choose ordinary matrix multiplication on the LHS and compatible nonempty
matrix shapes. Set `X[0,0] = W[0,0] = 1` and all other entries, including `A` and
`B`, to zero. The LHS has output entry one. Every legal RHS contraction
assignment produces zero. Shape-incompatible assignments are already excluded.

This sparse input rejects the rule without operation-name-specific logic.
The conclusion follows from the primitive addition, multiplication, read, and
sum semantics.

## Research contribution and limits

The proposed contribution is a semantics-derived checker for attribute
unrealizability, combining a finite sparse test theorem, compact symbolic
attribute constraints, and checkable rejection certificates. Evaluate whether
it reduces time and memory relative to dense verification and explicit
attribute enumeration, including examples that expose its worst cases.

The polynomial lemma and counterexample-guided synthesis are established
mathematical and algorithmic ingredients. Their combination alone does not
establish novelty. A stronger result would show efficient constraint generation
and substantial elimination of attribute families across user-defined dialects.

This proposal does not prove all valid rewrites, establish a finite rank bound,
or cover every scalar operation. Unsupported cases and exhausted searches
remain `unknown`.

## Related foundations

- [Quantitative Combinatorial Nullstellensatz](https://arxiv.org/abs/1212.5694):
  polynomial nonvanishing on finite grids; the lemma above only needs elementary
  interpolation.
- [Program Synthesis using Abstraction Refinement](https://www.cs.utexas.edu/~isil/syngar.pdf):
  related synthesis techniques for excluding families of candidates.
- [TensorRight: Automated Verification of Tensor Graph Rewrites](https://arxiv.org/abs/2511.17838):
  arbitrary-rank verification and a rank-bound theorem for its DSL. The present
  sparse-support bound addresses a different parameter and does not replace
  that kind of rank argument.
