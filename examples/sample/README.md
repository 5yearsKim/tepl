# Sample project

The examples demonstrate [shape](../../docs/references/shape.md) and
[dtype programs](../../docs/references/dtype.md), dtype-valued
attributes, rule dtype bindings, and descriptor-field access. Both Rust and C++
backends support these definitions.

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

## Function names

Builtin calls are unprefixed, such as `len(s)` or `gather(s, axes)`. Host-function
calls use `$`, such as `$infer_dot(X, W, @outer)`.
Descriptors retain their `@` prefix. Abstract-rule examples parameterize only
operations; host calls appear directly in concrete rules. See
[the builtin reference](../../docs/references/built-ins.md) for function syntax
and resolution rules.

The compiler requires `$` for direct host calls. Unknown unprefixed calls in
rules are errors. Shape and dtype blocks support builtin calls, lists, indexing,
conditionals, and comprehensions. Core checks parameters, names, builtin types,
attribute fields, assertions, and yield types. Both backends emit checked
evaluators for e-class analysis and RHS validation. Missing definitions return
`Unknown` and may require application inference hooks. Rule `where` and `derive`
builtins operate on `index`, `i64`, `index_list`, and `dtype`; `all` and `any`
accept Boolean lists inside shape and dtype programs.
See [the builtin reference](../../docs/references/built-ins.md) for signatures,
context availability, and rule examples.
The generated lab IR includes the matching dtype programs and runtime bindings.

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
bazel-bin/tepl generate examples/sample --out my_app/src/generated
```

Generation emits only module contents. Add `pub mod generated;` to your
application and egg 0.11 to its Cargo dependencies. The output directory can
have any name and live at any depth in the application's module tree.
Rust output is formatted by default; use `--no-format` to skip rustfmt.

Generation recursively discovers `.tepl` files under `dialects/`, `rules/`, and `graphs/`,
resolves imports relative to each file, and emits each dialect once. Nested rule and graph files
keep their relative module path. Rules with the same name in different files
remain distinct. Use unique dialect names across the project; aliases select
names in TEPL source but do not create extra Rust dialect modules.

Single-file generation is also supported:
`bazel-bin/tepl generate examples/sample/rules/simple.tepl --out generated/simple`.

Validate changes to these dialects and rules with:

```sh
./tools/test_codegen.sh
```

The script regenerates `labs/rust-egg/src/ir/` from this project, runs the lab's
tests, and checks additional compiler fixtures in temporary crates. Application
helpers remain in `labs/rust-egg/src/host/`.
