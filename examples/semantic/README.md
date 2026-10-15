# Symbolic tensor semantics and attribute synthesis

`tepl check` compiles a rewrite into attribute constructors and a checked
semantic graph under the rule’s explicit conditions. No concrete shapes,
attributes, or tensor values are supplied. Generated Rust/C++ execution remains
outside this implementation. The [greedy engine](../../docs/advanced/greedy_rewriting.md)
can execute compiled constructors on concrete graphs; see
[`examples/rewrite/auto.tepl`](../rewrite/auto.tepl).

## Run the checker

```sh
bazel run //:tepl -- check examples/semantic/rules/algebraic.tepl \
  --semantic-policy=algebraic --rule distribute
```

Distribution returns `status=proved`. Both RHS descriptors copy the four source
fields through expressions such as:

```text
lhs_contracting_dimensions = @d.lhs_contracting_dimensions
additional-guard: true
```

Inspect guarded reassociation or transpose remapping:

```sh
bazel-bin/tepl check examples/semantic/rules/algebraic.tepl \
  --semantic-policy=algebraic --rule associate
bazel-bin/tepl check examples/semantic/rules/basic.tepl \
  --semantic-policy=algebraic --rule transpose_dot_cycle
```

A proved rule exits 0. An unresolved proof fails analysis with a diagnostic and
exits 1. Without a selector, all rules in the file are checked. `--rule`
selects which rule receives semantic verification; syntax and name errors in
the file are still diagnosed.

The old `--input-shape` and `--known-attribute` flags have been removed. Concrete
inference is retained only in test support; it is not a compiler fallback.

## Declare conditions using output groups

`@operation->Group` returns the ordered output-axis positions of a named
logical group. The name is resolved from that operation's dialect definition;
it is not a built-in batch or contraction keyword.

For a dot whose output is `B ++ I ++ J`:

```tepl
rule associate {
    (dot[@d] (dot[@inner] A B) C)
    => (dot[auto] A (dot[auto] B C))
    where {
        @d.lhs_batching_dimensions == @inner->B
    }
}
```

The condition preserves the inner batch group. The outer contraction may
select any subset of either free group, including unsorted or interleaved
selections. Groups may contain multiple batch and contraction axes, and may be
empty. This is a supported sufficient restriction, not a weakest-condition
characterization of every valid reassociation.

The compiler lowers group references to metadata expressions. For example,
`@inner->J` gives positions after the batch and LHS-free groups. Such a reference
can still enforce whole-group routing, as in reverse reassociation and LoRA.
RHS attributes are inferred and checked independently.

References are allowed in `where` conditions and must identify a unique LHS
operation occurrence. Unknown groups, groups absent from the output (such as a
contracted `K`), and RHS-only references are errors. A group retains its declared
coordinate order: for a transpose with group `S`, `@p->S` gives the inverse
permutation, locating each original coordinate in the output. An empty output
group yields an empty position list.

Shape declarations constrain ranks and extents; they do not silently constrain
dot attributes. Required `index[]` descriptor fields can be compared with group
references and other index-list expressions in `where` conditions. Unsupported condition expressions are rejected by this checker rather than
being treated as symbolic facts.

Only declared conditions and LHS legality are proof assumptions. Missing facts
produce errors; the compiler never adds applicability restrictions to a rule.
The greedy engine checks the declared conditions and compiled proof conditions
before applying the rule. Axis-semantic Rust/C++ generation is still unsupported.

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
    value: Read | Const | Add | Product | Reduce
    interface: ordered axes and established source-to-output position maps
    occurrence_axes: input occurrence interfaces, including unused values
}
```

`Read` records a capture interface in physical dimension order and an optional
calculated access expression. Capture interfaces establish correspondence;
calculated accesses determine which elements are read.
Composition connects producer output positions to consumer bindings,
transports free coordinates, and preserves independent reduction scopes.
Transpose changes the ordered interface while retaining the value expression.

Axis identities, physical positions, and extents have separate expression sorts.
Selection and transport can split a group without enumerating its members at
compilation. `positions(interface, group)` derives attribute lists; a permuted
interface produces inverse-permutation remapping.

Selections use a generic ordered partition, independent of operation names:

```text
partition(parent, selected_positions) {
    selected  = parent[selected_positions]
    remaining = parent[complement(selected_positions)]
    selected_positions, remaining_positions
}
```

Selected axes retain selector order; remaining axes retain parent order. Splits
are interned and materialized only when needed. A subgroup keeps its maps into
both the selected interface and the original frame, including contraction
pairings. Composing maps handles cascading reductions and contractions through
the same mechanism.

A constructed tensor publishes the maps of surviving source groups. To move a
group into a selected view, the compiler first proves that the group survives,
then restricts its parent map. Composing the restricted map back recovers the
original positions. Lookup follows indexed origins with a breadth-first search
and caches results; it does not invoke the theorem prover.

Registered coordinate views retain their origin maps during normalization.
Output-order proofs compare ordered decompositions and maps into a common
frame. Equal axis sets alone cannot establish equal output order.

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
entailment has a bounded attempt; unresolved obligations produce `Unknown` and
prevent acceptance. Full permutation cancellation requires legality evidence;
a partial axis list cannot justify cancellation.

The dump includes a coordinate frame, access maps, normalized computations,
constructors, and guards. `m0`, `m1`, etc. are shared expression aliases. They
describe a metadata DAG, not executable TEPL statements. Fresh coordinate
identities are instantiated from metadata alone; they are not tensor element
values or flattened dimensions.

## Normalization and reconstruction

1. Lower the LHS and retain legality and supported declared rule conditions.
2. Normalize both arithmetic graphs into sums of products with reduction domains.
3. Match monomials by capture identity and establish access coordinates before
   combining coefficients. Repeated captures are supported when their access
   correspondence is unambiguous; general ambiguous matching remains deferred.
4. Derive each RHS intermediate's boundary: coordinates used inside that
   intermediate that another factor or the final output also needs.
5. Fit binding tables to that boundary. Operand incidence and reduction
   membership determine candidate groups. Construct position lists and respect
   fixed attributes and pins.
6. Re-lower the original RHS in a separate verifier. Check accesses, reduction
   domains, ordered axes, shape, dtype, legality, and partial-helper preconditions.
7. Prove every obligation from LHS legality and explicit conditions. Reject an
   unproved rule; candidate obligations cannot serve as their own assumptions.

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
| `transpose_dot_cycle` | Proved with declared rank and contraction restrictions |
| `absorb_transpose` | Proved with an explicit identity-permutation restriction |
| `transpose_dot_swap` | Proved with the declared output-block permutation |
| `associate`, `associate_reverse`, `lora` | Proved with explicit group-routing conditions |

Coordinate construction and independent verification are bounded. The former
fallback that invented routing guards has been removed. Compatible batched
forward reassociation supports partial selection across both free groups with
an explicit batch-preservation condition. Reverse reassociation and LoRA retain
their whole-group restrictions. Weakest-condition inference and completeness
for arbitrary tensor compositions are outside the supported scope.
Removing the conditions from the restricted examples causes verification to
fail. A failure to prove a rule is not a counterexample to every possible RHS
attribute assignment.

For internal profiling, set `TEPL_INDEX_PROFILE=1` when running the checker.
It prints elapsed time and counters for each inference/verification context to
stderr. Outer context times include any nested verification, so they should not
be added together. Normal compiler output is unchanged.

Explicit descriptor constructors are shown in
[rules/constructors.tepl](rules/constructors.tepl). Spreads and field expressions
are lowered to LHS metadata before independent semantic verification. Unsupported
symbolic expressions produce diagnostics. Application callbacks are no longer
part of the language.

## Proof contract and limits

For source metadata `p`, tensor values `X`, and constructor `f`:

```text
LegalLHS(p) AND DeclaredConditions(p)
    => LegalRHS(p, f(p)) AND LHS(p, X) = RHS(p, f(p), X)
```

Equality includes ordered output coordinates, shape, and dtype. Constructors
and declared conditions depend only on source metadata. Inverse permutation and position
lookup have explicit well-definedness obligations. A guard fails when a required
partial metadata computation is undefined.

- **Proved:** correctness follows from legality and declared conditions under
  the selected arithmetic policy; no additional guard is introduced.
- **Unknown:** the fragment, inversion, correspondence, or work budget did not
  establish a result. Compilation rejects the rule without claiming impossibility.

Exit codes are 0 for proved selected rules, 1 for rejected rules, and 2 for an
unknown selector. Normal analysis rejects unproved semantic rules. Internal test
inspection can retain unproved plans, but cannot mark them accepted.

Supported tensor declarations and pure metadata conditions are imported.
Unsupported conditions are diagnosed rather than silently ignored.
Intermediate captured-computation shape declarations are currently omitted,
so they cannot be relied upon to discharge a proof obligation.

Required `index[]` fields, fixed operand arity, and the scalar primitives
`Operand`, `Read`, `Const`, `Add`, `Product`, and `Reduce`
form the operation fragment. Calculated reads support addition of coordinate
tuples. Size-one expansion, reshape, general affine access inversion,
value-dependent domains, and arbitrary external computation are outside this language.

## Constants and cancellation

`Const(n)` accepts a signed 64-bit integer literal. Constants are scalar values
at the coordinates declared by the operation's axis interface. They do not
supply a tensor shape. The dialect defines negation and subtraction using these
primitives:

```tepl
semantics { yield Product(Const(-1), input); }
semantics { yield Add(lhs, Product(Const(-1), rhs)); }
```

`zero_like(input)` uses its input's axes and dtype and yields `Const(0)`.
Consequently, `subtract(A,A) => zero_like(A)` retains A's shape, including scalar
and empty tensors. Unused operands remain part of legality checking.

```sh
bazel run //:tepl -- check examples/semantic/rules/scalars.tepl \
  --semantic-policy=algebraic
```

Normalization keeps reads while finding coordinate correspondence. After RHS
lowering, it combines coefficients only for proven identical access mappings
and reduction domains. Equal tensor names alone do not justify cancellation.
Repeated factors retain their multiplicity. Arithmetic overflow reports an
unsupported proof instead of wrapping coefficients.

Constant reductions retain their domains: `Reduce(K, Const(1))` counts the
coordinates in K, and is not simplified to one. General ambiguous occurrence
matching, symbolic cardinality simplification, and rational constants are not
implemented. Arithmetic cancellation requires the algebraic policy.

## Explicit broadcasting

`broadcast_in_dim` (alias `broadcast`) inserts axes while preserving every
existing dimension's extent. Its attributes are:

- `dimensions`: input axis i maps to output position `dimensions[i]`.
- `new_extents`: extents for unoccupied output positions, in increasing position order.

For input shape `[2,3]`, `dimensions=[2,0]` and `new_extents=[5,7]` produce
shape `[3,5,2,7]`. The mapping is a distinct, in-range injection and may reorder
input axes. It does not expand an existing extent of one. This is TEPL's
restricted definition, not a complete StableHLO broadcast implementation.

The dialect uses generic axis primitives:

```tepl
axes {
    bind S { input: all_axes(input) };
    bind N = group(attrs.new_extents);
    yield place(S, attrs.dimensions, N);
}
semantics { yield input; }
```

`group(extents)` declares fresh axes with explicit nonnegative extents, including
zero. `place(S,p,N)` places S at positions p and fills the remaining positions
with N in order. Output coordinates may be independent of the scalar value;
that property propagates through consumers. Tensor operand alignment remains
explicit, so ordinary `add` does not acquire implicit broadcasting.

```sh
bazel run //:tepl -- check examples/semantic/rules/broadcast.tepl --rule broadcast_compose
```

The three structural examples prove under strict policy. Composition constructs
`dimensions = gather(@outer.dimensions, @inner.dimensions)`. New extents are
collected and reordered using their positions in the final output. Transpose
before or after broadcast also lowers to one inferred broadcast.

RHS output-only coordinates are matched to the required interface by position.
Equal extents alone never identify axes. A separate verifier reconstructs the
RHS and checks mappings, domains, output order, shape, dtype, and legality.
General synthesis with multiple output-only groups and arbitrary interface
changes remains limited. Reducing an inserted axis retains a reduction in the
semantic graph; the symbolic cardinality scaling law is deferred.

## Convolution and calculated reads

The dialect defines `convolution`, with alias `conv`, for arbitrary spatial
rank with at least one spatial dimension. It uses this layout:

```text
input:  [N, spatial..., C]
kernel: [kernel_spatial..., C, F]
output: [N, output_spatial..., F]
```

This is valid convolution in the deep-learning convention (cross-correlation):
no padding, no channel grouping, and no kernel reversal. The required
`strides` and `dilations` attributes contain one positive integer per spatial
axis. For each spatial dimension:

```text
effective_kernel = dilation * (kernel_size - 1) + 1
output_size = floor((input_size - effective_kernel) / stride) + 1
```

Each kernel extent is positive, and the effective kernel must fit inside the
input. Empty batches, input channels, and output channels are supported.
Use lists of ones for the previous stride-1, undilated behavior.

The spatial groups `O` and `Window` are arbitrary-length sequences. They are
not flattened. The semantic read is:

```tepl
Read(input,
    Concat(Coords(N),
        IndexAdd(IndexScale(attrs.strides, Coords(O)),
                 IndexScale(attrs.dilations, Coords(Window))),
        Coords(C))
)
```

`Coords(G)` is the current integer coordinate tuple of group G. `IndexAdd`
adds paired coordinates elementwise. `IndexScale(attrs.field, tuple)` multiplies
each coordinate by the corresponding attribute member. `Concat` joins tuples
in physical input-dimension order. For 2D, the read is
`input[n, sh*h+dh*r, sw*w+dw*s, c]`.

These are typed index expressions, separate from scalar `Add` and `Product`.
An operand with explicit `Read` may have partial or no direct axis bindings;
its reads must supply its full physical rank. Mixing implicit operand reads
with explicit reads of the same operand is rejected. Index sums, scales, and offsets require equal
tuple lengths. Access bounds are operation-legality obligations.

`extent_add(a,b)` and `extent_sub(a,b)` perform paired extent arithmetic;
`extent_mul(a,b)` multiplies paired members, and `extent_floor_div(a,b)`
performs elementwise mathematical floor division with positive divisors.
`offset_extents(a,n)` adds a scalar integer offset to each extent;
`scale_extents(a,n)` multiplies each member by a scalar integer.
`extent_le(a,b)` compares all paired extents, and `is_positive_extents(a)`
requires every member to be positive. Their tuple lengths must agree.

Composition substitutes calculated coordinates into producer values while
preserving reduction scopes. Addition of indices is flattened, sorted by scoped
identities, and combines repeated coordinates into integer coefficients.
Scaling distributes over coordinate sums and composes by multiplying per-axis
coefficients. Bounds use `s * floor(q/s) <= q` for nonnegative `q` and positive
`s`; this is a generic arithmetic law, independent of operation names.
The verifier compares calculated accesses as well as reduction domains and
ordered output interfaces. Equal extents alone never identify coordinates.
Nested computed-read producers can retain fresh intermediate domains until
substitution eliminates their local output coordinates.

Check input distribution, kernel distribution, transpose composition, and
nested convolution distribution:

```sh
bazel run //:tepl -- check examples/semantic/rules/convolution.tepl \
  --semantic-policy=algebraic
```

These four examples require no concrete metadata and introduce no additional
applicability guards. Input distribution, kernel distribution, and transpose
composition use `conv[auto]`. The compiler matches scoped coordinate bases in
calculated reads and constructs both RHS coefficients from the LHS attributes:

```text
strides   = @c.strides
dilations = @c.dilations
```

This mechanism uses the semantic IR, not convolution names or attribute names.
It solves whole per-axis coefficient holes by direct matching. Constructed
values are reusable metadata expressions; no concrete shapes or attribute
values are required. Matching supplies sufficient constructors and does not
claim to characterize every possible attribute assignment.

Output-domain checks run after coefficient construction. The independent
verifier lowers the completed RHS afresh and checks actual accesses, domains,
ordered interfaces, dtype, and legality. Trial matches add no proof premises.
Conflicting equations, ambiguous coordinate matches, and inverse or coupled
coefficient equations are rejected as unproved.

Nested convolution distribution still reuses `@inner` and `@outer`: general
inversion of composed coefficients and intermediate-domain correspondence
remain outside direct matching. Padding and configurable layouts remain
deferred. General calculated-access matching is bounded and incomplete;
an unestablished proof is rejected. Code generation remains unsupported.

## Slice and coordinate offsets

`slice` selects a window at arbitrary rank. Its three attributes have one
member per input axis:

```text
starts:  first selected input positions
sizes:   output element counts, not exclusive limits
strides: distances between selected input positions
```

Starts are nonnegative; sizes and strides are positive. The final selected
position `starts + strides * (sizes - 1)` must fit inside the input. No
clamping is performed. For example, `starts=[2], sizes=[3], strides=[2]`
selects positions `[2,4,6]`. Rank-zero tensors are supported; empty results,
negative strides, and dimension removal are outside this initial scope.

The semantic read uses a generic index primitive:

```tepl
Read(input,
    IndexOffset(attrs.starts, IndexScale(attrs.strides, Coords(O))))
```

`IndexOffset(attrs.field, tuple)` adds paired metadata offsets. Offsets have
no logical-coordinate dependencies. Composition, selection, and renaming keep
offsets separate from coordinate identities. The access normalizer uses:

```text
a + (b + i) = (a + b) + i
s * (a + i) = s*a + s*i
```

The same calculated-read matcher constructs unknown offsets and coefficients.
Slice sizes come from the ordered output interface. Two composed slices yield:

```text
starts  = inner.starts + inner.strides * outer.starts
sizes   = outer.sizes
strides = inner.strides * outer.strides
```

The verifier proves bounds using established LHS inequalities and the generic
law `L<=U => r+m*L <= r+m*U` for nonnegative `m`. Polynomial factoring recognizes
these relationships without enumerating tensor dimensions. This proof attempt
is bounded and incomplete; unresolved obligations are rejected rather than
added as assumptions.

Check distribution, composition, and transpose composition with inferred RHS
attributes:

```sh
bazel run //:tepl -- check examples/semantic/rules/slice.tepl \
  --semantic-policy=algebraic
```

The three examples require no concrete metadata and introduce no extra guards.
Pure Slice composition also checks under the default strict policy using
`--rule compose_slice`. The distribution examples use the algebraic policy
required by the current scalar normalizer. Code generation is not supported.

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
| `index_expr.h`, `index_expr.cc`, `extent_arithmetic.cc` | Coordinate tuples, access substitution, index normalization, extent arithmetic |
| `index_inference.h`, `index_inference.cc` | Scoped coordinate matching and coefficient/offset constructors |
| `normalize.cc` | Arithmetic normalization and intermediate boundaries |
| `synthesize.cc` | Constructors from required coordinates |
| `verify.cc` | Independent re-lowering under declared conditions |
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
bazel test //tests:semantic_test //tests:semantic_cli_test
```

## Compiler migration

This directory is the canonical semantic example suite. Operations define an
`axes` interface and a value `semantics` expression. The compiler has one proof
pipeline: lowering, coordinate composition, algebraic normalization, attribute
construction, and independent verification. It does not search with Z3.

Legacy `semantics(out_idx)`, tensor reads, and explicit sum comprehensions are
retired. `auto` requires supported axis semantics. Unproved rewrites must supply
explicit conditions or change their equation; there is no solver fallback.
`--rule` selects a rule. `--verify-small` and `--solver-timeout-ms` are retired.
Shape/dtype programs and ordinary rules without semantic definitions remain
supported.

The existing generated-code runtime is isolated in
[`src/legacy_semantics`](../../src/legacy_semantics/README.md). Its plan types and
serialization are compatibility interfaces for the existing generators. The
compiler does not populate those plans. New semantic constructors are not yet
emitted into Rust or C++.
