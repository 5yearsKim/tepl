<h1><img src="misc/images/logo/tepl_192.png" alt="tepl logo" width="48" height="48" align="absmiddle"> TEPL - Tensor Equality Pattern Language</h1>

<p align="right">
  <a href="labs/rust-egg/README.md"><img src="https://img.shields.io/badge/target-Rust-CE422B?style=flat&amp;logo=rust" alt="Target: Rust"></a>
  <a href="https://egraphs-good.github.io/"><img src="https://img.shields.io/badge/runtime-egg-F2C94C?style=flat" alt="Runtime: egg"></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/build-Bazel-43A047?style=flat&amp;logo=bazel" alt="Build: Bazel"></a>
</p>

**TEPL: *Write tensor rewrites that read like math.***

[Quick start](#-quick-start) · [Example rules](examples/rules)

The same tensor computation can be expressed in different ways—with very different costs. In machine learning, choosing the right form can make a big difference: depending on tensor dimensions, `(XA)B` can require far less computation than `X(AB)`.

[E-graphs](https://en.wikipedia.org/wiki/E-graph) help optimizers explore those alternatives. They compactly represent many equivalent expressions, letting rewrite rules uncover more ways to perform a computation.

[egg](https://egraphs-good.github.io/) brings this approach to Rust with a fast, flexible e-graph library for building optimizers. But applying it to tensor computations introduces a challenge: the rules need to capture more than the operations alone.

## 😢 Tensor rewrites need more than symbols

With egg, a symbolic rewrite is easy to express. For example, moving negation through a transpose:

```rust
use egg::{define_language, rewrite, Rewrite};

// Define symbolic operators and their arity without writing the language implementation.
define_language! {
    enum Math {
        "transpose" = Transpose(egg::Id),
        "negate" = Negate(egg::Id),
        Symbol(egg::Symbol),
    }
}

// Express both sides as compact symbolic expressions; ?x captures any subexpression.
let rule: Rewrite<Math, ()> = rewrite!(
    "transpose-negate";
    "(transpose (negate ?x))" => "(negate (transpose ?x))"
);
```

But which axes does `transpose` permute? Which dimensions does a matrix multiplication contract? Tensor IRs such as [StableHLO](https://openxla.org/stablehlo/spec) carry these details as attributes. A useful rewrite must preserve or transform them, while respecting shapes and dtypes.

## 😭 Pattern rewriting in Rust is painful

Custom matchers and appliers can handle those details, but a complete rewrite becomes verbose. Here is the same rule using this repository's Rust tensor runtime:

<details>
<summary><strong>👉 See the equivalent Rust rewrite</strong></summary>

```rust
// 👀 No need to follow every line! This example shows how verbose
// 👀 a simple tensor rewrite becomes when defining patterns by hand in Rust.

use egg::Var;
use rust_egg::ir::{DType, OpAttrs, dialects::tensor_lang::Op};
use rust_egg::ir::analysis::{infer_tensor_output, tensor_info};
use rust_egg::ir::pattern::{
    AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern, tensor_rewrite_checked,
};

let x: Var = "?x".parse().unwrap();
let perm = AttrVar::from("perm");

// Match transpose(negate(x)), capturing the transpose's permutation.
// Each operator needs explicit attributes and a nested vector of child patterns.
let lhs = TensorPattern::op(
    Op::Transpose,
    AttrPattern::Bind(perm),
    vec![TensorPattern::op(
        Op::Negate,
        AttrPattern::Exact(OpAttrs::None),
        vec![TensorPattern::Var(x)],
    )],
);

// Build negate(transpose(x)), reusing x and the captured permutation.
// The replacement requires a separate expression tree with the nesting reversed.
let rhs = TensorExpr::op(
    Op::Negate,
    AttrExpr::Exact(OpAttrs::None),
    vec![TensorExpr::op(
        Op::Transpose,
        AttrExpr::Captured(perm),
        vec![TensorExpr::Var(x)],
    )],
);

// Connect the two trees and check that the replacement preserves shape and dtype.
let rule = tensor_rewrite_checked(
    "transpose-negate",
    lhs,
    rhs,
    tensor_info,
    infer_tensor_output,
    move |graph, matched| {
        // Apply only to f32 inputs, matching the TEPL declaration below.
        let input = tensor_info(graph, matched.tensors[x])?;
        (input.dtype == DType::F32).then_some(Default::default())
    },
).unwrap();
```

</details>

Even this simple rewrite needs nested constructors, explicit attribute handling, and metadata checks. All that Rust code makes the underlying rule harder to see, review, and maintain—and larger patterns only add more nesting and repetition.

## ✨ Write the rule. Let TEPL generate the Rust.

TEPL turns declarative tensor rewrite rules into Rust code for egg, with attribute capture, shape/dtype analysis, and support for multiple dialects.

Here is the complete TEPL rule, using the example tensor dialect. Save it as `examples/rules/transpose_negate.tepl`:

```javascript
from "../dialects/tensor.tepl" import TensorLang as t;

rule transpose_negate {
    X: f32[Dims...]

    (t.transpose[@perm] (t.negate X))
    =>
    (t.negate (t.transpose[@perm] X))
}
```

TEPL keeps tensor rewrites concise while supporting the features needed for real tensor optimizations:

- **Readable rules:** Write the match and replacement directly, without manually building nested Rust trees.
- **Attribute capture and reuse:** `@perm` carries the transpose attributes from the match into the replacement.
- **Shape and dtype analysis:** `f32[Dims...]` matches an `f32` tensor of any rank; generated checks require the replacement to preserve the output shape and dtype.
- **Custom rewrite conditions:** Use `where` to express legality checks and `derive` to compute new attributes.
- **Multiple dialects:** Define and import your own operation dialects, and write rules that rewrite between them.
- **Host function bindings:** Implement complex logic in Rust and call it from TEPL with `$function(...)` in `where` and `derive` blocks.
- **egg integration:** Generate Rust rewrites that run directly in egg, with generated tensor analysis or hooks for your own analysis and inference.

## 📉 A smaller computation with LoRA

The [LoRA example](labs/rust-egg/README.md#lora-saturation-example) uses TEPL-generated rules with egg to explore equivalent expressions and select one with a lower estimated arithmetic cost. Here, `@` denotes matrix multiplication.

| | Expression | Estimated arithmetic operations |
| --- | --- | ---: |
| Before | `X @ (W + A @ B)` | 69,632 |
| After | `X @ W + (X @ A) @ B` | 39,168 |

That's **about 44% fewer estimated arithmetic operations** for input shapes `X=[2,4,64]`, `W=[2,64,32]`, `A=[2,64,4]`, and `B=[2,4,32]`. The demo checks equal results using deterministic integer inputs; floating-point reassociation is controlled by the host's legality policy.

Read the [TEPL rule](examples/rules/lora.tepl) or reproduce the result from the repository root:

```sh
cargo run --manifest-path labs/rust-egg/Cargo.toml --example lora_saturation
```

## 🚀 Quick start

Try it from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples
bazel-bin/tepl generate examples --out my_app/src/generated
```

Explore the [example rules](examples/rules), [shape guide](examples/shape_guide.md), and [runnable egg integration](labs/rust-egg/README.md).
