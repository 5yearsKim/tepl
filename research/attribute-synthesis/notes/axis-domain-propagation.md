# Axis-domain propagation before attribute synthesis

> Historical research material. Implementation references and reproduction
> instructions describe the earlier approach. See the
> [current synthesis design](../../../design/attribute-synthesis.md) and
> [research index](../README.md) for present scope.

## Scope

This implementation simplifies the **sufficient access-matching constraints** at
one concrete match. It does not prove that those constraints characterize every
semantically valid attribute assignment. It also does not derive a rank cutoff.

The input is the existing compiled primitive DAG, source metadata, and uniquely
paired polynomial terms with distinct capture occurrences. No operation names
are needed. Repeated captures, ambiguous term pairing, optional unused RHS
reductions, and unsupported metadata defer to the existing SMT matcher.

## Representation

A vector has a finite length domain and a domain for each potentially active
entry. An entry domain describes its value **when the entry is active**. Its
inactive padded value has no semantic significance. Shared attributes share
these domains. Reduction-coordinate domains refer to logical LHS labels; their
extents and distinctness remain constrained.

Each primitive instruction installs a relation between its operands and result.
The runtime runs these relations from a work list, revisiting an instruction
when one of its domains shrinks. Concrete primitives fold immediately.

## Local deductions

- Equality intersects domains in both directions. Vector equality also equates
  lengths and compares mandatory entries.
- For `value = source[index]`, discard positions whose source-entry domain has
  no overlap with the required value. A forced position restricts that source
  entry. Length-derived indices retain inactive alternatives; optional entry
  domains can be narrowed only as domains of active entries.
- `gather` uses this lookup relation for each active output slot. Complete
  `scatter` uses the reverse relation and requires a valid placement permutation.
- `concat` equates the output length to the sum of input lengths and routes each
  active input entry to its offset. Forward output support conservatively unions
  the possible source positions; correlations can remain unresolved.
- `take` routes through its start offset and requires the requested segment to
  fit. A concrete segment reuses its source entries.
- `exclude` represents membership, keep flags, and prefix counts. Kept source
  entries route to their prefix positions. It removes selected values while
  preserving source order, matching the existing metadata builtin.
- Distinctness removes forced values from other mandatory slots. For each
  prefix of length `j`, let `U_j` be the union of possible active entry values.
  If `|U_j| < j`, that prefix cannot be active. Restrict the list length below
  `j`. This is a necessary cardinality condition, not a complete matching test.

All these deductions inspect local supports. They do not enumerate assignments
to full attribute vectors. Losing correlation makes propagation weaker, not an
impossibility proof about the rewrite.

## Correctness obligations

For an admitted sufficient matcher relation `C(p, beta, auxiliary)`, each
deduction must retain every satisfying assignment's active attribute entries.
Inactive padding is existential and is reconstructed independently. If the
remaining attribute domains are `D`, the intended invariant is:

```text
C(p, beta, auxiliary) implies beta is in D.
```

Therefore adding `D` to the original sufficient constraints preserves their
attribute solutions. Fixed entries may be substituted by constants. Residual
domains lower to ordinary SMT variables, and reconstructed witnesses still pass
the independent legality and polynomial verifier.

Termination follows from finite domains: each update removes values, and only
an actual update reschedules dependent instructions. This gives no useful
worst-case efficiency guarantee. Materialized domains, relations, and expression
construction still consume time and memory; budgets can stop the pass.

The local invariant does **not** justify rejecting a rewrite when the sufficient
matcher fails. A non-bijective algebraic identity can still hold. The implementation
discards all propagated restrictions and rebuilds unrestricted metadata search
after reduced matching is exhausted. Time/work budgets are shared across attempts.
Resource exhaustion yields `Unknown`.

## Practical checks

`tests/axis_domains_test.cc` compares preprocessing against unreduced inference
on small shapes, exhaustively checks small permutations and reduction-list
witnesses, preserves both valid reduction orderings, tests shared fields and
failed sufficient proofs, and solves rank-100/200 cases without the ordered
constructor. Generated Rust also applies rank-100 distribution through the
native runtime. These tests support the implementation; they are not a
mechanized proof of the entire interpreter.

The expensive search can shrink to a few residual choices even when vectors are
long. Processing the vector and reconstructing the result still depend on rank.
No claim of complexity depending only on the number of residual variables is
made.
