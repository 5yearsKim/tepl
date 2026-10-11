# Element semantics and automatic attributes

TensorLang uses the readable `semantics(out_idx)` DSL. The compiler lowers
operation definitions into a shared expression IR and derives constraints from
language primitives. It does not recognize operation names or particular
rewrite trees.

The earlier aligned-axis and ordered-network engines have been removed, together
with the legacy operation-template classifier and handwritten rewrite handlers.
The [implementation plan](IMPLEMENTATION_PLAN.md) records the migration and the
remaining work. Research notes in this directory describe proposals, not enabled
compiler guarantees.

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
    → primitive metadata constraints at matched ranks
    → Z3 candidate attributes
    → exact symbolic equivalence check
    → construct RHS only after verification
```

At match time, ranks, shapes, and captured descriptors are known. Each unknown
`index[]` field needs an explicit shape assertion of the form
`is_valid_axis_list(attrs.field, bound)`. That assertion supplies a finite domain:
a length variable and bounded integer entries, with range and distinctness
constraints. Other assertions constrain lengths, extents, and axis relationships.
Intermediate output shapes and final output order are constrained as well.

The solver uses Z3 models rather than enumerating permutations in application
code. A source-field copy or identity ordering is tried as an initial candidate,
checking every primitive constraint and tensor equivalence before acceptance.
This can avoid symbolic vector allocation entirely. Preferences affect
candidate order only. If a candidate
fails verification, the solver excludes that exact joint assignment and requests
another model. In the worst case this can still inspect many assignments.

The verifier recursively substitutes intermediate tensor reads, expands scalar
addition and multiplication, and compares polynomials over independent input
entries. A faster sufficient check compares symbolic output and reduction
coordinates, renaming bound coordinates consistently and preserving reduction
cardinalities. If that check does not prove equality, finite entry-polynomial
expansion checks the actual extents. Tensor values are never sampled.

The implementation is shared between the compiler and both generated backends:

- `src/core/inference/check.cc`: rule composition, shape lowering, domain discovery.
- `src/core/inference/semantic_ir.h`: portable expression and constraint plan.
- `src/core/semantics/solver.cc`: primitive constraints, Z3, and equivalence checks.
- `src/core/semantics/runtime.cc`: versioned native ABI.
- `templates/{rust,cpp}/src/builtins/semantic.*`: generated runtime adapters.

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

Shape comprehensions, symbolic conditional shape programs, inferred non-list
attributes, tensor-value-dependent branches, scalar division, and arbitrary
nonpolynomial operators are outside this implementation. The parser/typechecker
can accept some expressions that the proof engine cannot handle.

## Result meanings

| Result | Meaning |
| --- | --- |
| `Found` | The returned attributes satisfy generated legality constraints, and equivalence has been checked for all tensor values at these metadata. |
| `Invalid` | In algebraic mode, all legal attribute models have been excluded by complete fixed-metadata checks. |
| `Unknown` | Unsupported reasoning, unavailable runtime, time/budget limits, or an inconclusive candidate prevents a conclusion. |

A failed symbolic index comparison alone never proves invalid. Solver SAT proves
metadata legality only; tensor equivalence still needs verification. Candidate
selection does not claim a unique solution. An impossible match and an unknown
match both leave the graph unchanged.

The compile-time checker also rejects an unconditional, unconstrained,
attribute-free rule when its scalar instance has an exact counterexample. For
example, `(add X Y) => X` is rejected. It does not ignore user guards or declared
shape constraints to produce such a diagnostic.

**There is no generic compile-time arbitrary-rank coverage proof.** A compiled
auto rule reports `coverage=match-guarded`; some or all matches can fail. Checking
small ranks does not establish a theorem for higher ranks.

The rank representation is generic, but resource limits are practical limits:
symbolic unknown axis-list allocations currently have capacity at most 32, index/shape vector capacity
at most 256, and polynomials at most 4096 terms. The solver timeout covers
constraint construction and verification as well as Z3. Exceeding a limit means
unknown, never a mathematical rank counterexample.

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
