# TEPL documentation

**Write tensor rewrites that read like math.**

TEPL (Tensor Equality Pattern Language) describes tensor operations, rewrite
rules, and computation graphs. It generates Rust code for
[egg](https://egraphs-good.github.io/) or C++ headers for
[egg-c](https://github.com/5yearsKim/egg-c), including shape and dtype inference.
Your application runs the optimizer and chooses the cost model.

## Start here

| Your goal | Guide |
| --- | --- |
| Understand equality saturation | [What is an e-graph?](0_what_is_e_graph.md) |
| See why TEPL exists | [What is TEPL?](1_what_is_tepl.md) |
| Learn the language's building blocks | [Dialects, rules, and graphs](2_core_concept_tepl.md) |
| Run a rewrite in Rust | [Rust tutorial](tutorial/rust/01_build.md) |
| Run a rewrite in C++ | [C++ tutorial](tutorial/cpp/01_build.md) |

## A first rule

This rule moves negation through a transpose while keeping the permutation:

```javascript
from "../dialects/tensor.tepl" import TensorLang as t;

rule transpose_negate {
    X: f32[Dims...]

    (t.transpose[@perm] (t.negate X))
    =>
    (t.negate (t.transpose[@perm] X))
}
```

`X` captures an `f32` tensor of any rank. `@perm` captures the transpose's
attributes and reuses them in the replacement. The generated runtime checks
shape and dtype compatibility; rule authors and host applications supply the
mathematical and numerical legality policy.

The repository's [sample project](../examples/sample)
contains this rule and its tensor dialect. From the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples/sample
bazel-bin/tepl generate examples/sample --out my_app/src/generated
```

Rust is the default output target. Add `--target cpp` to generate C++ headers.
Follow the [Rust](tutorial/rust/02_run.md) or [C++](tutorial/cpp/02_run.md)
tutorial to connect generated code to an application and run the optimizer.

## Go further

- [Attributes and conditions](advanced/attributes_and_conditions.md): capture
  operation attributes, check legality, and derive replacement attributes.
- [Binding and early pruning](advanced/binding_and_early_pruning.md): understand
  how constraints filter matches.
- [Custom analysis](advanced/custom_analysis.md): connect facts from your host
  application to generated rules.
- [Language reference](references/cli.md): look up CLI commands,
  [shapes](references/shape.md), [dtypes](references/dtype.md), and
  [built-in functions](references/built-ins.md).
- [Developer guide](developer_guide.md): contribute to the compiler and runtimes.
