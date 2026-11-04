# Ordered synthesis implementation

This directory implements direct axis-attribute construction for algebraic
contraction, full permutation, whole-axis unary summation, and separated addition.
Recognition uses declared axis/value semantics, not operation or field names.

| Files/API | Responsibility |
| --- | --- |
| `contract.*`, `recognize(Definition)` | Conservative semantic recognition |
| `admit(Plan, Mode)` | Explicit structural admission and a reason on rejection |
| `compile(Plan, Mode)` | Reusable schedules, component scopes, field adapters, and domain requirements |
| `inference::emit(...)` | Shared symbolic metadata IR: canonical memberships, order predicates, layouts, and positional fields |
| `construct(Context&, EmissionBlock, ...)` | Optional symbolic specialization of that same IR, with independent verification |
| `//src/inference` | Concrete analysis, feasibility, construction, and independent replay |

Admission requires ideal algebraic arithmetic, read-once components, matching
input sets, and independent unrestricted RHS attributes. The concrete domain
requires input extents at least two; scalar inputs and empty selections are legal.
Domain requirements are separate from structural admission and feasibility.

The source is distributed into components by linearity and bilinearity. Component
input sets must be distinct. The RHS must already be an explicit sum of read-once
components with matching input sets; normalization never duplicates RHS holes.
Captured source descriptors retain their identities in every distributed copy.
Each component owns its port partition, even when components share inputs.
Original addition nodes are also checked for equal ordered operand shapes.

The single-component solver first computes M and P bottom-up, then propagates
required orders. Feasibility includes distinctness, membership, subsequences,
contraction block order, and exact root retained-set equality. Summation removes
unneeded retained axes without reordering survivors; a later sum may have an
empty reduction list after a contraction eliminates its axes early.

Symbolic generation produces q independently of the written guard. The verifier
checks q's source scope and definedness, then validates builders, operation
legality and semantic equivalence under C && q. The runtime evaluates q before
builders; false q is ordinary non-application. Bounded conditional verification
may remain unresolved after successful generation.

The [synthesis design](../../../../design/attribute-synthesis.md) describes
integration, evaluation, and validation. The
[research index](../../../../research/attribute-synthesis/README.md) identifies
current proofs and their relationship to implementation.

```sh
bazel test //tests:ordered_synthesis_test //tests:attribute_reasoning_test \
  //tests:named_nodes_test
```

## One constructor, two realizations

`inference::emit` is the single canonical constructor for admitted synthesis
plans. It builds the symbolic metadata IR before creating any proof context or
running normalization. `ordered::compile(...).emission` and the ordered symbolic
strategy consume the same IR. The former separate `deriveRequirements` and
symbolic attribute-building recurrences have been removed.

`Result::metadata_program` retains this IR even when bounded source lowering,
correspondence, or expression specialization fails. Set-valued nodes use only
membership; ordered layouts and selectors retain their sequence order. The
symbolic adapter specializes IR operations through existing source mappings and
partitions. It may jointly reorder paired contraction selectors into source
reduction order, preserving the finite sum and output layout. Its candidate is
still checked by a fresh independent verifier, including original definedness.
Generated feasibility becomes an explicit runtime predicate; semantic obligations
remain compilation requirements under that predicate.

The native interpreter executes the same residual IR, using source port-class
replay and finite metadata operations. Checks are evaluated before any partial
position builder. Symbolic simplification is not a precondition of emission or
native execution. Resources, missing/illegal metadata, and out-of-domain extents
remain distinct from infeasibility. The completeness domain is unchanged.

`core-emit` exposes residual IR for inspection. `--core` compiles supported
constructors and schema extensions with match-time feasibility. The symbolic
exporter retains feasibility expressions in generated Rust/C++. Successful
emission alone does not prove arbitrary tensor equivalence; admission and
conditional semantic validation remain required.
