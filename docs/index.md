# TEPL documentation

<p align="center">
  <img src="assets/images/tepl_thumb.png" alt="TEPL tensor rewrite illustration" width="720">
</p>

<div align="center" markdown="1">


<p align="center"><strong>TEPL: <em>Write tensor rewrites that read like math.</em></strong></p>

<div class="tepl-badges" align="center" markdown="1">

[![Target: Rust](https://img.shields.io/badge/target-Rust-CE422B?style=flat&logo=rust)](tutorial/rust/01_build.md) [![Target: C++](https://img.shields.io/badge/target-C%2B%2B-00599C?style=flat&logo=cplusplus)](tutorial/cpp/01_build.md) [![Runtime: egg](https://img.shields.io/badge/runtime-egg-F2C94C?style=flat)](https://egraphs-good.github.io/) [![Build: Bazel](https://img.shields.io/badge/build-Bazel-43A047?style=flat&logo=bazel)](#try-the-sample-project) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

</div>

</div>

TEPL (Tensor Equality Pattern Language) describes tensor operations, rewrite
rules, and computation graphs. It generates Rust code for
[egg](https://egraphs-good.github.io/) or C++ headers for
[egg-c](https://github.com/5yearsKim/egg-c), including shape and dtype inference.
Your application runs the optimizer and chooses the cost model.

## Start here

<div class="tepl-start" markdown="1">

1. **Understand the idea**

    Learn about equality saturation and see why TEPL exists.

    [What is an e-graph? →](0_what_is_e_graph.md)
    [What is TEPL? →](1_what_is_tepl.md)

2. **Learn the language**

    Get to know the language's building blocks.

    [Dialects, rules, and graphs →](2_core_concept_tepl.md)

3. **Run your first rewrite**

    Build TEPL and run a rewrite in your preferred language.

    [Rust tutorial →](tutorial/rust/01_build.md)
    [C++ tutorial →](tutorial/cpp/01_build.md)

</div>

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
attributes and reuses them in the replacement.

!!! note "Shape, dtype, and legality"

    The generated runtime checks shape and dtype compatibility; rule authors
    and host applications supply the mathematical and numerical legality policy.

### Try the sample project

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

## 💡 Write rules in VS Code

The TEPL extension for VS Code brings syntax highlighting and code formatting
to your editor, so you can write rules with less friction.

<p align="center">
  <img src="assets/images/tepl_vsc.png" alt="TEPL extension in VS Code with syntax highlighting for a LoRA rewrite rule" width="640" loading="lazy">
</p>

## Go further

<div class="tepl-resources" markdown="1">

- **Write more expressive rules**

    [Attributes and conditions](advanced/attributes_and_conditions.md): capture
    operation attributes, check legality, and derive replacement attributes.

    [Binding and early pruning](advanced/binding_and_early_pruning.md): understand
    how constraints filter matches.

- **Connect your application**

    [Custom analysis](advanced/custom_analysis.md): connect facts from your host
    application to generated rules.

- **Look up the details**

    [Language reference](references/cli.md): look up CLI commands,
    [shapes](references/shape.md), [dtypes](references/dtype.md), and
    [built-in functions](references/built-ins.md).

- **Contribute to TEPL**

    [Developer guide](developer_guide.md): contribute to the compiler and runtimes.

</div>
