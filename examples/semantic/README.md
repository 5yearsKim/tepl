# Element semantics and automatic attributes

TensorLang uses the readable `semantics(out_idx)` DSL. The compiler lowers
operation definitions into a shared expression IR and derives constraints from
language primitives. It does not recognize operation names or particular
rewrite trees.

The earlier aligned-axis and ordered-network engines have been removed, together
with the legacy operation-template classifier and handwritten rewrite handlers.
The [implementation plan](IMPLEMENTATION_PLAN.md) records the implemented passes
and remaining work. Semantic access constraints are compiled into a symbolic
plan and instantiated at match time. The ordered-contraction constructor remains
an optimization for a checked fragment. The research rank cutoff does not
provide a compiler guarantee yet.

## Define an operation once

Shape contracts specify legal metadata. Element semantics specify computation.
Both blocks are lowered into the same expression bytecode.

```tepl
op add(lhs: tensor, rhs: tensor) -> tensor {
    dtype(a, b) { assert a == b; assert is_numeric(a); yield a; }
    shape(l, r) { assert l == r; yield l; }
    semantics(out_idx) {
        yield read(lhs, out_idx) + read(rhs, out_idx);
    }
}
```

See [TensorLang](dialects/tensor.tepl) for `add`, `dot_general` (alias `dot`), and
`transpose`. Dot attributes contain four dimension lists; precision and algorithm
attributes are omitted.

Semantics are optional for ordinary TEPL rules. A rule using `auto` requires
semantics and supported metadata contracts for every operation it expands.
Missing definitions produce a compile-time diagnostic. Well-typed expressions
outside the inference fragment produce an explicit unsupported/unknown result.

```tepl
rule transpose_dot_lhs {
    (dot[@d] A B)
    =>
    (dot[auto] (transpose[auto] A) B)
}
```

Pins remain supported, including `transpose[auto, permutation = [1, 2, 0]]`.
Fields outside the inferred `index[]` domain must be pinned or have an empty
schema default, and must not participate in shape or element semantics.

## Lowering and solving

```text
resolved shape + semantics expressions
    → operation-independent expression bytecode
    → compose reads, distribute products, compile symbolic constraint plan
    → embed plan in generated Rust
at match time:
    → certified ordered-contraction fragment?
        yes: construct attributes and check network/certificate
        no: instantiate metadata + semantic constraints in Z3
            → candidate → exact equivalence check
            → if sufficient constraints fail: unrestricted metadata search
    → construct RHS only after verification
```

The compile-time pass recursively substitutes intermediate reads and normalizes
scalar sums and products. It combines structurally equal terms after renaming
reduction binders, sorts factors, and removes zero coefficients. This comparison
preserves reduction-domain order; it does not decide every equivalent program.
Each additive term records its coefficient, reduction
domains, and reads of captured tensors. Its access expressions reference symbolic
output coordinates, reduction coordinates, shapes, and attribute fields. Vector
lengths remain symbolic; compilation does not choose a rank or unroll axes.

For example, `dot(add(A, B), C)` lowers to two terms, each with a reduction:

```text
sum(k) read(A, [i,k]) * read(C, [k,j])
sum(k) read(B, [i,k]) * read(C, [k,j])
```

The distributed RHS lowers to the same two-term structure, with access maps
depending on its unknown descriptors. At match time, Z3 chooses a bijection
between terms and between their reduction coordinates. Corresponding reduction
extents, coefficients, captures, and read-coordinate vectors must agree. Repeated
reads of the same capture get a separate occurrence bijection. Their multiplicities
are preserved, and every occurrence uses the same reduction renaming. All
terms constrain the same attribute variables jointly. Coordinate labels denote
identities, not sampled tensor entries. Operation and attribute names do not
participate in this procedure.

These are **sufficient constraints**, not a complete characterization of equality.
An unused reduction over two points, for example, can equal twice its body without
having a reduction bijection to that scalar multiplication. If this stage is unsatisfiable,
its constraints and candidate exclusions are removed, and the existing generic
solver searches the full metadata-legal domain. Resource exhaustion can still
return unknown. A sufficient-plan failure never proves a rule invalid.

Unused one-point reductions are supported. The compiler eliminates literal
singleton domains, including the empty dimension vector whose index space has
one point. For matched LHS metadata, it checks whether every extent is one.
For symbolic RHS metadata, Z3 can choose to retain or eliminate an unused binder,
but elimination requires every active extent to be one. A zero extent prevents
elimination. Usage is checked conservatively from access-expression references;
used singleton coordinates are not eliminated by this optimization.

At match time, ranks, shapes, and captured descriptors are known. Each unknown
`index[]` field needs an explicit shape assertion of the form
`is_valid_axis_list(attrs.field, bound)`. That assertion supplies a finite domain:
a length variable and bounded integer entries, with range and distinctness
constraints. Other assertions constrain lengths, extents, and axis relationships.
Intermediate output shapes and final output order are constrained as well.

The generic solver uses Z3 models rather than enumerating permutations in
application code. When no compiled semantic plan is available, a source-field
copy or identity ordering is tried first, checking legality and equivalence
before acceptance. Otherwise, the compiled constraints guide the first search.
If a candidate fails verification, the solver excludes that exact joint
assignment and requests another model. The unrestricted fallback can still
inspect many assignments.

The verifier recursively substitutes intermediate tensor reads, expands scalar
addition and multiplication, and compares polynomials over independent input
entries. A faster sufficient check compares symbolic output and reduction
coordinates, renaming bound coordinates consistently and preserving reduction
cardinalities. If that check does not prove equality, finite entry-polynomial
expansion checks the actual extents. Tensor values are never sampled.

The implementation is shared between the compiler and both generated backends:

- `src/core/inference/check.cc`: rule composition, shape lowering, domain discovery.
- `src/core/inference/semantic_ir.h`: portable expressions, normalized terms, and constraint plan.
- `src/core/inference/constraints.*`, `constraint_normalize.*`: compile-time composition and term normalization.
- `src/core/semantics/solver.cc`: primitive constraints, Z3, and equivalence checks.
- `src/core/semantics/constraint_match.h`: term, factor-occurrence, and reduction matching constraints.
- `src/core/semantics/ordered.*`: contract extraction, construction, certificate checks.
- `src/core/semantics/runtime.cc`: versioned native ABI.
- `templates/{rust,cpp}/src/builtins/semantic.*`: generated runtime adapters.

Rust code generation embeds a per-rule `AUTO_CONSTRAINT_PLAN` constant. The
native request format is versioned as `tepl-constraints-2`; regenerate applications
and use the corresponding runtime library after updating the compiler.

## Structural inference for ordered contractions

The compiler extracts a contract from normalized primitive IR. It checks both
coordinate routing and the exact shape assertions. A contraction contract has
one read of each operand, their scalar product, a common reduction coordinate,
and output coordinates ordered as batch, left free, right free. A permutation
contract reads one operand through a complete `scatter` permutation.

Admission does not use operation names, attribute names, or rewrite templates.
Renamed fields and operations, swapped operand roles, commuted products, and
consistently reordered scatter groups retain the same contract. The first
certifier recognizes these normal forms; arbitrary equivalent programs can
remain unrecognized and use the generic solver. Extra shape assertions, missing
legality checks, or different scalar computations prevent admission.
Lowering also preserves legality checks from shape `let` evaluations, including
unused expressions that can fail through division, bounds, or overflow.

At match time, the structural path requires:

- Algebraic arithmetic and every physical dimension at least two; scalars are allowed.
- The same distinct capture variables, used once on each side, in expression trees.
- Certified contractions/permutations throughout the trees.
- Independent, unrestricted RHS fields: four axis lists per contraction or one
  permutation list. Pins, shared descriptors, and known RHS fields use the generic path.

The LHS supplies a partition of input-axis ports into logical indices. For each
RHS subtree, indices used outside that subtree or in the final output must stay
live. Shared live indices become batches; shared indices whose use ends there
are contracted. Required output subsequences determine the remaining layouts.
The constructor translates these logical connections into concrete axis lists.
It performs no SMT candidate search and never enumerates permutations.

Before returning `Found`, a separate concrete evaluator checks shape legality,
the exact port partition, and output order. Before returning `Invalid`, a separate
checker verifies a two-axis ordering obstruction at a dot or input leaf. This
rejects every RHS assignment in the admitted fragment. Adding an unrestricted
root transpose repairs ordering failures in that fragment.

Resource exhaustion remains `Unknown`. The generic list-allocation limit of 32
does not apply to this path; it has a total input-axis limit of 100,000, tree depth
limit of 256, and the configured time/work budget. These are implementation
limits, not a mathematical rank bound. Construction cost is polynomial in tree
size and axis count; no tight complexity bound is claimed.

The Rust `semantic::solve` result exposes `method` (`Ordered`, `ConstraintsZ3`, `Z3`, or
`Unavailable`) and `reason`. Invalid ordered results identify the RHS node and
the conflicting logical axes. Normal generated rewrite application still
declines failures before graph mutation. `ConstraintsZ3` identifies a witness
found using the compiled sufficient constraints and checked by the verifier.

Compile-time checking records how many operation definitions acquire contracts.
It does **not** enable the two-axis universal-coverage theorem: exhaustive schema
checking, source-attribute restrictions, and guard-preservation proofs remain
separate work. See [the theorem draft](research/ordered_contractions.md).

## Supported fragment

| Layer | Supported |
| --- | --- |
| Tensor structure | Fixed operand arity, nested operations, repeated captures, named intermediate bindings. |
| Attributes | Nonoptional `index[]`, explicit finite axis-list bounds, literals and captured-field pins. |
| Metadata and indexing | `shape`, `rank`, `len`, `range`, `gather`, `exclude`, `concat`, `take`/`slice`, `scatter`, lists, indexing, comparisons, integer arithmetic, axis validity and disjointness. |
| Element computation | `read`, finite `sum(k in index_space(dims))`, `+`, `-`, `*`, unary signs, integer constant expressions as coefficients or elements. |
| Conditions | Metadata/index conditions that become concrete during evaluation. Symbolic conditional or short-circuit metadata needs future guarded lowering. |
| Dtypes | Numeric captures with a common dtype; operation dtype contracts forward an operand, optionally checking equality and numeric type. |

`exclude(v, selected)` follows the existing metadata builtin: it removes selected
**values** while preserving order. For axis lists, use
`exclude(range(rank(t)), selected_axes)`.

`scatter(rank, axes, values)` requires complete coverage: axes are a permutation
of all positions, and both vectors have the required length. Equal extents do not
identify distinct coordinates.

This supports the [example rules](rules/algebraic.tepl), general contractions,
transposes, distribution, reassociation, LoRA, and user-defined reductions within
the fragment. A rule can be expressible while some matches have no witness.
The backend may also return unknown before finding one.

The compiled semantic plan currently admits algebraic polynomials with rectangular
reductions and pure coordinate routing through `gather`, `take`, indexing,
`concat`, lists, and complete `scatter`. Products may use captures repeatedly;
the constraint matcher supports at most 128 read occurrences per term.
Coordinate arithmetic, constant input indexing, and strict arithmetic retain the
original verifier/search path. Compilation limits include 64 additive terms,
10,000 constraint expressions, and expansion depth 256. Exceeding these limits
disables this optimization; it does not reject the rule.

Shape comprehensions, symbolic conditional shape programs, inferred non-list
attributes, tensor-value-dependent branches, scalar division, and arbitrary
nonpolynomial operators are outside this implementation. The parser/typechecker
can accept some expressions that the proof engine cannot handle.

## Result meanings

| Result | Meaning |
| --- | --- |
| `Found` | The returned attributes satisfy operation legality, and equivalence has been checked for all tensor values at these metadata. |
| `Invalid` | In algebraic mode, a checked ordering certificate or complete fixed-metadata exclusion proves no witness exists. |
| `Unknown` | Unsupported reasoning, unavailable runtime, time/budget limits, or an inconclusive candidate prevents a conclusion. |

A failed symbolic index comparison alone never proves invalid. A compiled
semantic plan adds sufficient access-map conditions to metadata legality;
every returned candidate still passes the independent equivalence verifier. Candidate
selection does not claim a unique solution. An impossible match and an unknown
match both leave the graph unchanged.

The compile-time checker also rejects an unconditional, unconstrained,
attribute-free rule when its scalar instance has an exact counterexample. For
example, `(add X Y) => X` is rejected. It does not ignore user guards or declared
shape constraints to produce such a diagnostic.

**There is no generic compile-time arbitrary-rank coverage proof.** A compiled
auto rule reports `coverage=match-guarded`; some or all matches can fail. Checking
small ranks does not establish a theorem for higher ranks.

The generic solver's rank representation has practical resource limits:
symbolic unknown axis-list allocations currently have capacity at most 32, index/shape vector capacity
at most 256, and polynomials at most 4096 terms. The solver timeout covers
constraint construction and verification as well as Z3. Exceeding a limit means
unknown, never a mathematical rank counterexample.

## Measuring constraint generation

The native benchmark runs the solver used by generated Rust. It compares
metadata-only search, compiled constraints with fallback, and normal production
dispatch. The first two modes disable the ordered optimization and initial
candidate preferences. Production retains those optimizations. All modes share
the time and work budgets; controlled modes also have a configurable candidate
limit. Exhausting a limit returns unknown.

```sh
bazel build -c opt //:semantic_benchmark
bazel-bin/semantic_benchmark --repeats 3 --warmup 1 --timeout-ms 500 > /tmp/semantic.csv
python3 tools/summarize_semantic_benchmark.py before.csv /tmp/semantic.csv > /tmp/report.md
```

The CSV records shapes, known attributes, pins, compilation time, matching time,
metadata/semantic construction time, Z3 time, verification time, solver calls,
candidate counts, method, fallback reasons, limits, seed, and compiler/Z3 versions.
Metadata construction time includes source-legality solver calls, so phase
times overlap. These measurements exclude Rust graph matching and ABI serialization.
Compilation is measured once per distinct rule and excluded from matching time.

Fixtures cover transpose, distribution, association, LoRA, multiple batch/reduction
axes, repeated extents and captures, pins, empty/singleton domains, and impossible
rewrites. Use `--filter`, `--mode`, `--budget`, `--max-candidates`, and `--seed`
to control a run. Run without concurrent builds for useful timing comparisons.
The [recorded report](benchmarks/REPORT.md) includes before/after CSVs and source
fingerprints. Timing is informational; correctness tests do not assert speedups.

## Numerical policy

Use `--semantic-policy algebraic` for ideal exact arithmetic. This permits
polynomial distribution and reassociation; it does not prove floating-point
error bounds.

The default `strict` policy requires identical expanded evaluation traces.
Different traces remain unknown when no acceptable candidate is found. Trace
comparison is a conservative semantic criterion, not an operational IEEE-754
proof: dialect semantics do not specify rounding, reduction implementation,
NaNs, or overflow behavior.

## Build and run

```sh
bazel build //:tepl //src/core:libtepl_semantic_runtime.so
bazel-bin/tepl check examples/semantic --semantic-policy algebraic
bazel-bin/tepl generate examples/semantic --semantic-policy algebraic \
  --target rust --out /tmp/tepl-semantic
export TEPL_SEMANTIC_RUNTIME="$PWD/bazel-bin/src/core/libtepl_semantic_runtime.so"
```

Generated Rust and C++ applications load the same native library at match time.
Set `TEPL_SEMANTIC_RUNTIME` before the first inference call, or install
`libtepl_semantic_runtime.so` on the dynamic loader path. The library needs its
matching Z3 shared dependency; preserve the Bazel runfiles/solib layout or package
that dependency alongside a suitable loader configuration. No Rust Z3 crate or
C++ Z3 headers are required by generated applications.

The adapters currently support Unix dynamic loading. Other targets return
unknown. Missing libraries also return unknown and decline the rewrite.
Ordinary rules without semantic inference continue to work without the library.

Validation commands:

```sh
bazel test //tests/...
tools/test_codegen.sh
tools/test_codegen_cpp.sh
```

The integration scripts build and configure the native runtime automatically.
