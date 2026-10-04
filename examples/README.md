# Example project

- `dialects/tensor.tepl`: a StableHLO subset with generated shape definitions and
  attribute schemas; see the guide for adapter metadata and coverage limits.
- `dialects/scalar.tepl`: a minimal second dialect with `add` and `negate`.
- `rules/`: all concrete rules and reusable abstract templates.
- `rules/lowering.tepl`: checked rank-zero TensorLang → Scalar lowering.
- `shape_guide.md`: implemented shape definitions, builtins, and host-call syntax.

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

## Function names

Builtin calls are unprefixed, such as `len(s)` or `gather(s, axes)`. Host-function
calls use `$`, such as `$is_same_dtype(X, Y)` or `$infer_dot(X, W, @outer)`.
Descriptors retain their `@` prefix. Abstract-rule examples parameterize only
operations; host calls appear directly in concrete rules. See
[the shape guide](shape_guide.md) for shape syntax and resolution rules.

The compiler requires `$` for direct host calls and accepts the extended attribute
types used here. Unknown unprefixed calls in rules are errors.
Operation shape blocks now have structured parser/AST support, including builtin
calls, lists, indexing, conditionals, and comprehensions. Core checks parameter
signatures, names, builtin types, attribute fields, assertions, and yield types.
Rust codegen emits checked shape evaluators and declared dtype policies used
by e-class analysis and RHS validation. Missing definitions return `Unknown`
and may require application inference hooks. Rule `where` and `derive` support
builtins on `index`, `i64`, and `index_list`; `all` and `any` remain shape-only.
See [the builtin reference](../src/core/builtins/README.md) and
[`rules/builtins.tepl`](rules/builtins.tepl) for an executable rule example.
The lab IR is regenerated from these declarations and shared runtime templates.

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
bazel-bin/tepl check examples
bazel-bin/tepl generate examples --out my_app/src/generated
```

Generation emits only module contents. Add `pub mod generated;` to your
application and egg 0.11 to its Cargo dependencies. The output directory can
have any name and live at any depth in the application's module tree.
Rust output is formatted by default; use `--no-format` to skip rustfmt.

Generation recursively discovers `.tepl` files under both directories, resolves
imports relative to each file, and emits each dialect once. Nested rule files
keep their relative module path. Rules with the same name in different files
remain distinct. Use unique dialect names across the project; aliases select
names in TEPL source but do not create extra Rust dialect modules.

Single-file generation is also supported:
`bazel-bin/tepl generate examples/rules/simple.tepl --out generated/simple`.

The checked-in lab IR is generated from this project:

```sh
./tools/regenerate_lab.sh
./tools/regenerate_lab.sh --check
```

Edit dialects and rules here, regenerate, then run
`cargo test --manifest-path labs/rust-egg/Cargo.toml` to exercise their generated
code with the handwritten hosts under `labs/rust-egg/src/host/`.
