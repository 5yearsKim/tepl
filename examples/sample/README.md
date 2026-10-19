# Sample project

The examples demonstrate [shape](../../docs/references/shape.md) and
[dtype programs](../../docs/references/dtype.md), dtype-valued
attributes, rule dtype bindings, and descriptor-field access. Operation shape/dtype programs remain supported by both backends; the extended
rule expression frontend is described below.

[`tepl.toml`](tepl.toml) marks this directory as the project root. Rules and
graphs import dialects with paths such as `"dialects/tensor.tepl"`; reusable
rules use paths such as `"rules/abstract.tepl"`.

- `dialects/tensor.tepl`: a StableHLO subset with generated shape definitions and
  attribute schemas.
- `dialects/scalar.tepl`: a minimal second dialect with `add` and `negate`.
- `dialects/dtype.tepl`: numeric conversion, comparison, selection, and a fixed
  result dtype expressed with dtype programs.
- `rules/`: all concrete rules and reusable abstract templates.
- `rules/dtype.tepl`: dtype binding and identity-conversion removal.
- `rules/lowering.tepl`: checked rank-zero TensorLang → Scalar lowering.
- [Shape reference](../../docs/references/shape.md): dimension lists, inference,
  attributes, variadic operands, and rule constraints.
- [Dtype reference](../../docs/references/dtype.md): element types, inference,
  explicit conversion, and rule bindings.

## 🧩 Concrete graphs

The [graph examples](graphs/README.md) define concrete inputs, attributes, shared
values, and outputs. Both backends generate constructors from `graphs/`.

## Rank-zero shapes

Use `[]` for a scalar (rank-zero tensor), and add a dtype prefix when needed:

```tepl
rule commute_scalars {
    X: []
    Y: []
    (t.add X Y) => (t.add Y X)
}
```

Here `t` is the imported TensorLang dialect. Both captures allow any dtype;
`X: f32[]` additionally restricts the element type to `f32`. Scalars use the
same shape declarations and rank checks as tensors with dimensions.

## Pure rule expressions

Calls such as `len(Dims)` and `gather(values, axes)` resolve to builtins.
Rules no longer support application callbacks or `fn` template parameters.
Abstract rules still parameterize operations.

`where` contains Boolean guards and local `let` bindings. `derive` calculates
replacement descriptors with constructors and immutable copy/update expressions:

```tepl
derive {
    let permutation = gather(@inner.permutation, @outer.permutation);
    @combined = { ..@outer, permutation: permutation };
}
```

The RHS operation supplies the constructor's schema. A spread copies all fields;
explicit fields override the copied values. Without a spread, optional fields
become `none`, declared defaults apply, and required fields must be supplied.

Rules support homogeneous nested lists, indexing, `if ... then ... else ...`,
and comprehensions. `all` and `any` accept Boolean lists in rules as well as
metadata programs. Locals are scoped to their block; earlier derived descriptors
remain available to later derivations. See
[attribute construction](../../docs/advanced/attributes_and_conditions.md) and
[builtins](../../docs/references/built-ins.md).

`rules/attributes.tepl` demonstrates transpose composition and identity guards.
`rules/lora.tepl` expresses equal-batch matrix contractions through explicit
axis guards and descriptor copies. Distribution and reassociation still require
an appropriate numerical policy; metadata checks do not authorize them.

The frontend checks these examples today. Code generation for constructors,
local bindings, lists, indexing, conditionals, and comprehensions in rules is
pending. Generating this entire project reports a diagnostic and writes no
files. Existing simple rule and operation metadata generation remains available.

The operation set follows the [StableHLO specification](https://openxla.org/stablehlo/spec).
`relu`, `scale`, `square`, `alias`, `rmsnorm`, `vocab_cross_entropy`,
`online_attention`, and `symbol` were removed because they are not StableHLO
operations. `exp` is now an alias for `exponential`; explicit broadcasting uses
`broadcast_in_dim`. Elementwise arithmetic requires equal shapes.

Rule-level shape declarations are shorthand for match-time `where` conditions.
An absent operation shape definition means no generated inference: the host may
supply metadata, otherwise the shape is unknown rather than invalid.

## Checking and generation

Run from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples/sample
bazel-bin/tepl generate examples/sample/rules/simple.tepl --out my_app/src/generated
```

Generation emits only module contents. Add `pub mod generated;` to your
application and egg 0.11 to its Cargo dependencies. The output directory can
have any name and live at any depth in the application's module tree.
Rust output is formatted by default; use `--no-format` to skip rustfmt.

Generation recursively discovers `.tepl` files under `dialects/`, `rules/`, and
`graphs/`. Imports resolve from the project root marked by `tepl.toml`, and each
dialect is emitted once. Nested rule and graph files keep their relative module
path. Rules with the same name in different files
remain distinct. Use unique dialect names across the project; aliases select
names in TEPL source but do not create extra Rust dialect modules.

Single-file generation is also supported:
`bazel-bin/tepl generate examples/sample/rules/simple.tepl --out generated/simple`.

Validate changes to these dialects and rules with:

```sh
bazel test //tests:all --test_timeout=240
```

The existing generation integration scripts and lab artifacts use the previous
host-function API. Migrating those applications is deferred with extended rule
code generation. Use `bazel test //tests:all` for current compiler coverage.
