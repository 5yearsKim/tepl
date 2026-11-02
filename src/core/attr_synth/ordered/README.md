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

Symbolic generation does not use the written guard to simplify its predicate,
prove feasibility, or assume it. The independent
verifier must establish the generated obligations from source legality and the
written guard before checking the conditional builders. A guard proof may remain
Unknown after successful generation. No runtime guard is automatically added.

The [synthesis design](../../../../design/attribute-synthesis.md) describes
integration, evaluation, and validation. The
[research index](../../../../research/attribute-synthesis/README.md) identifies
current proofs and their relationship to implementation.

```sh
bazel test //tests:ordered_synthesis_test //tests:attribute_reasoning_test \
  //tests:where_suggestions_cli_test //tests:named_nodes_test
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
No candidate obligation becomes an assumption or an extra runtime guard.

The native interpreter executes the same residual IR, using source port-class
replay and finite metadata operations. Checks are evaluated before any partial
position builder. Symbolic simplification is not a precondition of emission or
native execution. Resources, missing/illegal metadata, and out-of-domain extents
remain distinct from infeasibility. The completeness domain is unchanged.

`core-emit` exposes the residual IR without accepting a rewrite. `--core` and
`--certified` retain their certification/admission policies and native portable
format; the ordinary symbolic exporter uses checked specialization. Rust/C++
exports still require successful expression lowering. These interfaces do not
select different canonical constructors, and successful IR emission does not
prove the written guard. Broader general construction and sound schema
extensions remain outside the complete-constructor theorem.
