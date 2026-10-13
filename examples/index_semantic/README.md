# Symbolic tensor semantics and attribute synthesis

`tepl check` compiles a rewrite into attribute constructors, a sufficient
applicability guard, and a checked semantic graph. No concrete shapes,
attributes, or tensor values are supplied. Generated Rust/C++ execution remains
outside this implementation.

## Run the checker

```sh
bazel run //:tepl -- check examples/index_semantic/rules/algebraic.tepl \
  --semantic-policy=algebraic --index-rule distribute
```

Distribution returns `status=proved`. Both RHS descriptors copy the four source
fields through expressions such as:

```text
lhs_contracting_dimensions = @d.lhs_contracting_dimensions
additional-guard: true
```

Inspect guarded reassociation or transpose remapping:

```sh
bazel-bin/tepl check examples/index_semantic/rules/algebraic.tepl \
  --semantic-policy=algebraic --index-rule associate
bazel-bin/tepl check examples/index_semantic/rules/basic.tepl \
  --semantic-policy=algebraic --index-rule transpose_dot_cycle
```

Conditional results exit 1 so they cannot be mistaken for unconditionally
accepted rules. Their constructors and guards remain available for inspection.
Without a selector, all rules in the file are checked.

The old `--input-shape` and `--known-attribute` flags have been removed. Concrete
inference is retained only in test support; it is not a compiler fallback.

## Define operation semantics

```tepl
axes {
    bind B { lhs: lb, rhs: rb };
    bind K { lhs: lc, rhs: rc };
    bind I { lhs: lf };
    bind J { rhs: rf };
    yield B ++ I ++ J;
}
semantics {
    yield Reduce(K, Product(lhs, rhs));
}
```

Each group is an arbitrary-length ordered sequence of logical coordinates.
Its members have individual extents and explicit operand-position mappings.
`bind K { lhs: [1,3], rhs: [2,0] };` connects `K[0]` to lhs position 1 and rhs
position 2, and `K[1]` to lhs position 3 and rhs position 0.

Equal extents never identify coordinates. Sharing a group explicitly connects
them. Batch, contraction, and free are local roles derived from bindings and
reductions, rather than permanent coordinate types.

Legality includes distinct in-range position lists, paired extents, complete
input partitions, dtype, and an ordered output containing every free coordinate
once. Empty groups and zero extents are supported.

`bind` names are visible in `semantics`; ordinary `let` names remain private to
`axes`. `axes` yields output order and thereby determines shape. `semantics`
yields the scalar value expression at those coordinates.

See [TensorLang](dialects/tensor.tepl) for addition, elementwise multiplication,
sum reduction, transpose, and generalized dot.

## Composed coordinate graph

Operation boundaries disappear during semantic lowering:

```text
TensorMeaning {
    axes:  ordered logical coordinate sequence
    shape: symbolic extent sequence
    dtype: symbolic dtype
    value: Read | Add | Product | Reduce
}
```

`Read` records a capture and its coordinates in physical dimension order.
Composition connects producer output positions to consumer bindings,
transports free coordinates, and preserves independent reduction scopes.
Transpose changes the ordered interface while retaining the value expression.

Axis identities, physical positions, and extents have separate expression sorts.
Selection and transport can split a group without enumerating its members at
compilation. `positions(interface, group)` derives attribute lists; a permuted
interface produces inverse-permutation remapping.

The compiler records both directions of each established binding:

```text
positions(interface, group) = selector
select(interface, selector) = group
```

Complete partitions also record how to select a group in the interface's
physical order. When a read views a producer through a validated permutation,
composition transports the consumer partition into that read's table. Each
operand use has its own coordinate map; sharing a tensor does not merge its
different accesses.

Map simplification consults indexed bindings and recorded legality evidence.
It does not launch general entailment recursively. Position, support, and
evidence results are cached until bindings or premises change. General
entailment has a bounded attempt; unresolved expressions remain conditions or
produce `Unknown`. Full permutation cancellation requires legality evidence;
a partial axis list cannot justify cancellation.

The dump includes a coordinate frame, access maps, normalized computations,
constructors, and guards. `m0`, `m1`, etc. are shared expression aliases. They
describe a metadata DAG, not executable TEPL statements. Fresh coordinate
identities are instantiated from metadata alone; they are not tensor element
values or flattened dimensions.

## Normalization and reconstruction

1. Lower the LHS and retain legality and supported declared rule conditions.
2. Normalize both arithmetic graphs into sums of products with reduction domains.
3. Match monomials by capture identity. The initial fragment requires unique
   term correspondence and no repeated capture factors within a monomial.
4. Derive each RHS intermediate's boundary: coordinates used inside that
   intermediate that another factor or the final output also needs.
5. Fit binding tables to that boundary. Operand incidence and reduction
   membership determine candidate groups. Construct position lists and respect
   fixed attributes and pins.
6. Re-lower the original RHS in a separate verifier. Check accesses, reduction
   domains, ordered axes, shape, dtype, legality, and partial-helper preconditions.
7. Discharge obligations or retain them in a sufficient guard. Close additional
   guard preconditions before independently checking the fixed result.

Laws include distributivity, reduction linearity, reduction extraction,
reduction composition, associativity, commutative matching, and coordinate
renaming. Extraction requires bound coordinates to be absent from the other
factor. Domains must be finite and independent of tensor values. Structured
rewrite steps record before/after expressions and side conditions.

Arithmetic normalization requires `--semantic-policy=algebraic`. These are ideal
arithmetic semantics; the implementation does not establish floating-point
reassociation safety. Transpose composition also works under strict policy.
The rule being checked is never used as an algebraic axiom.

No operation names, rule names, attribute names, or supplied numeric examples
participate in inference. Binding tables and primitives determine the analysis.
Renamed and structurally different operation definitions are tested.

## Current coverage

| Rule | Result under algebraic policy |
| --- | --- |
| `distribute`, `distribute_explicit`, `sum_distribute` | Proved |
| `transpose_compose` | Proved, including strict policy |
| `merge_reductions`, `transpose_reduce` | Proved |
| The four rules in [multiply.tepl](rules/multiply.tepl) | Proved |
| `transpose_dot_lhs` | Proved; chooses an identity transpose witness |
| `transpose_dot_cycle` | Conditional; rank and coordinate-order obligations |
| `absorb_transpose`, `transpose_dot_swap` | Conditional |
| `associate`, `associate_reverse`, `lora` | Conditional; sufficient routing guards |

General coordinate construction is bounded. If it cannot establish a result,
the routing strategy selects a conservative fragment: shared retained
multiplicative groups are empty, and consumer selectors route whole producer
groups according to the capture sets needed by RHS intermediates. This supports
unbatched contractions with arbitrary-length free and contracting groups.
It does not establish unrestricted batched reassociation or the weakest
condition for arbitrary partial-group routing.

The initial general construction attempt uses at most 100,000 work units.
Conservative routing and independent verification retain the caller's work
budget. Guard closure is bounded; optional guard minimization does not rerun
verification on the compilation path.

For internal profiling, set `TEPL_INDEX_PROFILE=1` when running the checker.
It prints elapsed time and counters for each inference/verification context to
stderr. Outer context times include any nested verification, so they should not
be added together. Normal compiler output is unchanged.

The host-copy fixture is in [unsupported/host_call.tepl](unsupported/host_call.tepl).
It remains unknown because `$copy_permutation` has no semantic contract. The
compiler never assumes a host function's behavior from its name.

## Proof contract and limits

For source metadata `p`, tensor values `X`, and constructor `f`:

```text
LegalLHS(p) AND DeclaredConditions(p) AND Guard(p)
    => LegalRHS(p, f(p)) AND LHS(p, X) = RHS(p, f(p), X)
```

Equality includes ordered output coordinates, shape, and dtype. Constructors
and guards depend only on source metadata. Inverse permutation and position
lookup have explicit well-definedness obligations. A guard fails when a required
partial metadata computation is undefined.

- **Proved:** no additional guard remains under the selected arithmetic policy.
- **Conditional:** constructors are verified under the sufficient printed guard.
- **Unknown:** the fragment, inversion, correspondence, or work budget did not
  establish a result. This does not prove impossibility.

Exit codes are 0 for proved selected rules, 1 for conditional/unknown results,
and 2 for an unknown selector. Conditional does not establish satisfiability,
uniqueness, completeness, or a weakest applicability condition.

Supported tensor declarations and pure metadata conditions are imported.
Unsupported conditions are omitted conservatively, making the proof obligation
stronger. Intermediate captured-computation declarations are currently omitted.

Required `index[]` fields, fixed operand arity, and the four scalar primitives
form the initial operation fragment. Broadcasting, reshape, affine indexing,
value-dependent domains, and general host semantics are deferred.

## Implementation and verification

The implementation lives in [`src/core/index_semantics`](../../src/core/index_semantics):

| File | Responsibility |
| --- | --- |
| `ir.h`, `check.cc`, `plan.cc` | Definitions, templates, declared assumptions |
| `expr.cc` | Immutable typed expressions and unconditional sequence identities |
| `context.h`, `context.cc` | Private rule state, source scope, attribute assignment, guards |
| `axis_bindings.cc` | Coordinate tables and their lookup indexes |
| `axis_maps.cc`, `axis_transport.cc` | Position lookup, selection, producer-consumer map composition |
| `axis_normalize.cc` | Evidence-based normalization and axis-set operations |
| `axis_theory.cc` | Fact registration, shared structural laws, bounded entailment |
| `proof_policy.h` | Named work limits and expected inference failures |
| `graph.cc` | Generic composition into coordinate primitives |
| `normalize.cc` | Arithmetic normalization and intermediate boundaries |
| `routing.cc` | Sufficient whole-group routing restrictions |
| `synthesize.cc` | Constructors from required coordinates |
| `verify.cc` | Independent re-lowering and fixed-guard verification |
| `print.cc` | Coordinate tables, shared expressions, diagnostics |

Public APIs are `synthesize(plan)` and `verify(plan, proof)` in the corresponding
headers. `Plan::proof` stores the compiler result. Neither API takes concrete
metadata. Verification shares semantic lowering and sequence theory with
synthesis, but does not trust inferred coordinate annotations or claimed status.
This is a checked implementation, not a mechanically proved theorem.

Normalization consults established evidence; bounded entailment is a separate
query. A failed query means the property remains unestablished. Mapping and
attribute updates invalidate their associated caches through `Context` methods.
Transport uses indexed snapshots of the relevant interfaces, so it can add
bindings safely while iterating. Position lookup retains a fallback for records
whose expressions have acquired equivalent canonical forms.

Expected unsupported cases, exhausted budgets, and invalid proof candidates
produce `Unknown` with diagnostics. Unexpected implementation exceptions
propagate instead of being classified as unsupported inference.

Tests independently interpret the metadata program and compare dense tensor
results for matrices, multiple contracting axes, zero extents, transpose order,
and reassociation. They also exercise corrupted constructors and unsupported
cases. Concrete reference inference is confined to `tests/index_reference_*`.

```sh
bazel test //tests:index_semantics_test //tests:index_cli_test
```
