# Metadata and coordinate reasoning

Semantic verification uses source legality and the rule's declared conditions to
prove the replacement's attributes, accesses and output layout. The checker does
not assume a replacement condition or add a guard to make a proof succeed.
Unsupported obligations remain unproved.

## Arithmetic and bounds

The checker propagates integer intervals through addition, subtraction,
multiplication and metadata indexing. Shape dimensions are nonnegative `u64`
values. Source attributes retain their declared integer ranges. Intermediate
metadata arithmetic uses checked signed 128-bit integers; constructed attributes
must also fit their destination `index` or `i64` type. The pointwise vector
checker currently uses the union of these attribute ranges as a sufficient
arithmetic bound; safe vector intermediates outside it may remain unproved.

Pointwise extent expressions normalize as polynomials. Established inequalities
can justify new bounds through positive multiples of source inequalities. For
example, if `a`, `b` and `d` are nonnegative sequences of equal length and the
source establishes `a + b <= d`, the checker can derive `a <= d`. It does not
infer axis identities from equal dimension sizes.

The original derived expressions are checked before simplification. An invalid
index or overflowing intermediate cannot become valid merely because its value
later cancels. Known metadata conditionals check the selected branch; empty
comprehensions do not evaluate their bodies. Symbolic branches require enough
information to establish safe evaluation.

## Ordered sequences and comprehensions

The semantic checker lowers metadata indexing and list comprehensions into
internal sequence expressions. These use the existing rule syntax:

```tepl
[x for x in a.attrs.dims]
[a.attrs.starts[i] + b.attrs.starts[i] for i in range(len(a.attrs.starts))]
```

An identity comprehension preserves its sequence. Pointwise indexing and
arithmetic become symbolic selection and extent expressions, so their proofs do
not require a fixed tensor rank. Small concrete lists can also be evaluated
within a bounded amount of work. Uniform sequences and supported `all`
predicates supply general element facts. Other comprehension bodies may remain
unproved.

Sequence order and repeated positions are preserved. For example, selecting
positions `[0, 0, 1]` from `[7, 9]` gives `[7, 7, 9]`. Disjointness is required
when combining distinct axis selectors, rather than assumed for all mappings.

## Calculated coordinates

Coordinate normalization composes affine offsets, selections and reversals.
Two reversals in the same extent domain combine by symmetric difference: an
axis selected twice returns to its original coordinate. Only reversal selectors
are treated as unordered sets; tensor interfaces and ordinary position sequences
keep their order. Reversals with different extent domains do not cancel.

Proof search is bounded and uses no catalog rule names or host callbacks.
Checked division, metadata reductions, unresolved symbolic branches and more
general comprehensions still need additional domain reasoning. This extension
does not introduce tensor values into coordinate expressions.
