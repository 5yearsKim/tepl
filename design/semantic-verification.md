# Semantic verification

The compiler checks a rewrite for all legal source matches satisfying its
written conditions, under the selected arithmetic policy. Attribute constructors
and conditions depend on metadata, not tensor values. The executable result
contains completed computations; runtimes do not repeat proof search.

Use the [semantic example project](../examples/semantic/README.md) for runnable
rules, and the [public metadata reference](../docs/references/metadata-reasoning.md)
for language syntax. [Attribute synthesis](attribute-synthesis.md) describes
candidate construction and the ordered specialization.

## Operation contracts

An operation's `axes` interface declares logical groups and output order. Its
`semantics` block describes scalar values through operands, explicit reads,
constants, sums, products, and reductions. Explicit shape programs take priority
for executable shape inference; semantic legality still comes from the declared
contracts.

| Semantic primitive | Meaning |
| --- | --- |
| `Read(input, coordinates)` | Read an operand in its physical dimension order |
| `Coords(G)` | Integer coordinate tuple for a declared logical group |
| `Concat` | Concatenate coordinate tuples |
| `IndexSelect` | Select distinct in-range coordinate positions in order |
| `IndexAdd` | Add coordinate tuples elementwise |
| `IndexScale` | Apply supplied positive per-axis scales |
| `IndexOffset` | Add supplied nonnegative per-axis offsets |

Explicit reads must supply full operand rank and satisfy access bounds. Tuple
operations require matching lengths where appropriate. An operand cannot mix
implicit reads and explicit reads in the same definition. Equal extents do not
establish logical-axis identity.

Scale/offset metadata can use pure expressions. Production `auto` infers axis
selectors only: it does not construct convolution strides, slice starts/sizes,
or other coordinate/extent fields. The
[convolution](../examples/semantic/rules/convolution.tepl) and
[slice](../examples/semantic/rules/slice.tepl) rules supply those fields explicitly.
Broadcast examples similarly supply new extents while requesting axis mappings.
Mathematical extensions with broader operation contracts are described separately
in [research](../research/attribute-synthesis/README.md).

## Composition and normalization

`src/core/semantics/graph.cc` assembles operations and interfaces.
`metadata_lowering.cc` lowers metadata/dtype contracts; `value_lowering.cc`
substitutes tensor values and scoped coordinates. Shared tree walks live in
`symbolic/tree.h`; callers retain reduction-scope checks.

Composition substitutes calculated coordinates into producer values while
preserving bound-coordinate scope. The normalizer flattens/sorts coordinate
sums, combines repeated coordinates, and composes scales and offsets. For
example, nested supplied slices require the explicit formulas
`inner.starts + inner.strides * outer.starts` and
`inner.strides * outer.strides`; their availability as formulas does not grant
automatic coordinate-field synthesis or prove every resulting bound.

`symbolic/correspondence.cc` separates value dependencies from interfaces and
propagates whole-operand shape bindings. `index_maps/` owns lookup, selection,
map composition, evidence-based normalization, and axis-set reasoning. Context
updates invalidate dependent caches; immutable snapshots permit safe transport
while new bindings are added.

Symbolic keys are immutable DAGs with deterministic lookup digests and structural
collision checks. Each proof context owns its key arena. Repeated DAGs do not
expand into trees, and no global registry determines identity or proof order.

`symbolic/reduction_reflection.cc` can reflect bound coordinates in established
finite domains while preserving free coordinates. Such reduction reordering
requires the appropriate arithmetic policy; ideal-algebraic equality does not
establish strict floating-point equality.

## Acceptance contract

A successful attempt establishes legal, defined RHS construction, matching shape
and dtype, and equivalent values in ordered output coordinates for every legal
source match satisfying the declared conditions and generated feasibility q. Partial metadata computations,
including inverse permutations and position lookup, need definedness evidence.

| Outcome | Meaning |
| --- | --- |
| Proved | Required obligations follow under the selected policy and C && q; q is retained for matching |
| Unknown | The fragment, correspondence, implication query, or budget did not establish the result |

Failure to construct or prove one candidate does not prove that every RHS
attribute assignment is impossible. Ordinary compilation rejects unproved
semantic rules. Inspection can retain failed attempts for diagnostics, but cannot
manufacture a verified program. Structural rules without value semantics have
no semantic-equivalence proof.

Synthesis proposes candidates; `proof/verify.cc` re-lowers the completed RHS in
a fresh context without trusting inferred annotations or claimed status.
Semantic orchestration requires successful proof before materialization.
All accepted semantic rules have zero unresolved obligations; those obligations
cannot be converted into hidden runtime guards. Unsupported cases and exhausted
budgets produce diagnostics; unexpected implementation exceptions propagate.

Written conditions on groups and metadata remain effective. Generated
feasibility restricts applications without replacing semantic or backend
conditions. The exact admission
restrictions of a research theorem are separate from practical proof success.

## Implementation and validation

Entry points and stage types are described in the
[compiler pipeline](compiler-pipeline.md). Definition checking lives in
`src/core/checking/operations/semantics.cc`; immutable expressions and coordinate
normalization live in `src/core/semantics/symbolic/`; proof limits and legality
checks live in `src/core/semantics/proof/`.

```sh
bazel test //tests:semantic_test //tests:semantic_cli_test \
  //tests:convolution_test //tests:slice_test //tests:reverse_test
```

Independent dense references check tensor results and constructed metadata,
including empty domains, ordering, repeated captures, and corrupted constructors.
Concrete reference inference is confined to `tests/index_reference_*`.
For diagnostic profiling, `TEPL_INDEX_PROFILE=1` reports elapsed time and counters
per context. Outer context times include nested verification and must not be
summed as disjoint phases.
