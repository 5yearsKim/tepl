# 🧩 What is TEPL?

TEPL (Tensor Equality Pattern Language) is a declarative language for tensor
graph rewrites. Rules specify transformations, operation attributes, tensor shapes,
dtypes, and legality conditions. TEPL generates Rust code for egg or C++ code for
egg-c.

## Background

Equivalent tensor computations can have very different costs. For compatible
matrices under exact arithmetic, `(XA)B = X(AB)`. Their intermediate tensor sizes
and computation costs depend on the matrix dimensions. An optimizer can choose a
cheaper form when the application's numerical rules allow it.

E-graphs compactly represent these alternatives. Each **e-class** groups equivalent
expressions. An **e-node** represents an operation whose children refer to
e-classes. Rewrite rules add alternatives; extraction selects an expression using
a cost model.

The Rust library egg supports this workflow with compact symbolic patterns.
Tensor rewrites also need attributes, shape/dtype inference, and legality checks.
TEPL brings these requirements into a dedicated language and generated runtime.

## Problem statement

**1. Operation symbols alone do not fully describe tensor semantics.**

<details markdown="1">
<summary><strong>👉 Check in detail</strong></summary>

egg expresses rewrites as symbolic patterns, such as moving negation through
a transpose:

```rust
use egg::{define_language, rewrite, Rewrite};

define_language! {
    enum Math {
        "transpose" = Transpose(egg::Id),
        "negate" = Negate(egg::Id),
        Symbol(egg::Symbol),
    }
}

let rule: Rewrite<Math, ()> = rewrite!(
    "transpose-negate";
    "(transpose (negate ?x))" => "(negate (transpose ?x))"
);
```

Which axes does `transpose` permute? For a rank-three tensor, swapping axes
`(1, 0)` gives `[1, 0, 2]`, while swapping `(2, 1)` gives `[0, 2, 1]`. The
replacement must preserve the matched permutation, which the symbol alone
does not encode.

Reassociating matrix multiplication adds another challenge:

```text
(matmul (matmul ?x ?a) ?b)
=>
(matmul ?x (matmul ?a ?b))
```

When represented as general tensor contractions, these operations carry attributes
for contracting and batching dimensions. Reassociation may require new attribute
values, but this pattern does not say how to derive them.

The rewrite must also check operand compatibility, preserve output shape and
dtype, and respect the host's floating-point reassociation policy.

egg supports these details, but the author must encode and check them.

</details>

<br>

**2. Custom tensor patterns in Rust become verbose and complex.**

<details markdown="1">
<summary><strong>👉 Check in detail</strong></summary>

Custom matchers and appliers can handle these checks, but the code grows quickly.
Here is the transpose-negate rule using this repository's Rust tensor runtime:

```rust
use egg::Var;
use rust_egg::ir::{DType, OpAttrs, dialects::tensor_lang::Op};
use rust_egg::ir::analysis::{infer_tensor_output, tensor_info};
use rust_egg::ir::rewriting::{
    AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern,
    tensor_rewrite_checked,
};

let x: Var = "?x".parse().unwrap();
let perm = AttrVar::from("perm");

// Match transpose(negate(x)), capturing the transpose's attributes.
let lhs = TensorPattern::op(
    Op::Transpose,
    AttrPattern::Bind(perm),
    vec![TensorPattern::op(
        Op::Negate,
        AttrPattern::Exact(OpAttrs::None),
        vec![TensorPattern::Var(x)],
    )],
);

// Build negate(transpose(x)), reusing x and the captured attributes.
let rhs = TensorExpr::op(
    Op::Negate,
    AttrExpr::Exact(OpAttrs::None),
    vec![TensorExpr::op(
        Op::Transpose,
        AttrExpr::Captured(perm),
        vec![TensorExpr::Var(x)],
    )],
);

// Check replacement metadata and restrict the rule to f32 inputs.
let rule = tensor_rewrite_checked(
    "transpose-negate",
    lhs,
    rhs,
    tensor_info,
    infer_tensor_output,
    move |graph, matched| {
        let input = tensor_info(graph, matched.tensors[x])?;
        (input.dtype == DType::F32).then_some(Default::default())
    },
).unwrap();
```

Even this small identity needs separate match and replacement trees, attribute
bindings, and metadata checks. Larger patterns bury the mathematical rule in
bookkeeping, making it harder to review and maintain.

</details>

<br>

**3. Custom e-class analysis requires substantial shape/dtype boilerplate.**

<details markdown="1">
<summary><strong>👉 Check in detail</strong></summary>

Tensor operations usually need shape and dtype analysis to check their inputs
and infer their outputs. This information is critical for both **valid rewrites**
and **efficient search**: a matcher can reject incompatible candidates early,
before exploring the rest of a pattern or building a replacement.

egg supports this through e-class analysis, but a tensor analysis requires
substantial boilerplate. It must track metadata, propagate unknown or invalid
results, and combine facts when e-classes merge.

Shape and dtype inference can also be complex. Transpose permutes dimensions,
broadcasting aligns shapes, and contractions check axes and dimension sizes.
Dtype inference must follow each operation's type restrictions and promotion
policy.

Even simple addition needs the analysis code below. This standalone egg 0.11
example supports `f32` and `i32` inputs and requires identical shapes and dtypes,
without broadcasting or promotion:

```rust
use std::collections::{BTreeSet, HashMap};
use egg::{define_language, Analysis, DidMerge, EGraph, Id, Symbol};

define_language! {
    enum Tensor {
        "add" = Add([Id; 2]),
        Input(Symbol),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DType { F32, I32 }

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TensorType {
    shape: Vec<u64>,
    dtype: DType,
}

// Retain known evidence even when another alternative has unknown metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Facts {
    observed: BTreeSet<TensorType>,
    unknown: bool,
    invalid: bool,
}

impl Facts {
    fn known(ty: TensorType) -> Self {
        Self { observed: BTreeSet::from([ty]), ..Self::default() }
    }

    fn is_invalid(&self) -> bool {
        self.invalid || self.observed.len() > 1
    }

    fn info(&self) -> Option<&TensorType> {
        if self.unknown || self.is_invalid() {
            None
        } else {
            self.observed.iter().next()
        }
    }
}

#[derive(Default)]
struct TensorAnalysis {
    inputs: HashMap<Symbol, TensorType>,
}

impl Analysis<Tensor> for TensorAnalysis {
    type Data = Facts;

    fn make(graph: &mut EGraph<Tensor, Self>, node: &Tensor, _id: Id) -> Facts {
        match node {
            Tensor::Input(name) => match graph.analysis.inputs.get(name) {
                Some(ty) => Facts::known(ty.clone()),
                None => Facts { unknown: true, ..Facts::default() },
            },
            Tensor::Add([lhs, rhs]) => {
                let a = &graph[graph.find(*lhs)].data;
                let b = &graph[graph.find(*rhs)].data;
                // An invalid operand takes priority over an unknown one.
                if a.is_invalid() || b.is_invalid() {
                    return Facts { invalid: true, ..Facts::default() };
                }
                match (a.info(), b.info()) {
                    (Some(a), Some(b)) if a == b => Facts::known(a.clone()),
                    (Some(_), Some(_)) => {
                        Facts { invalid: true, ..Facts::default() }
                    }
                    _ => Facts { unknown: true, ..Facts::default() },
                }
            }
        }
    }

    fn merge(&mut self, target: &mut Facts, incoming: Facts) -> DidMerge {
        let before = target.clone();
        target.observed.extend(incoming.observed.iter().cloned());
        target.unknown |= incoming.unknown;
        target.invalid |= incoming.invalid;
        // Tell egg whether the result differs from either original value.
        DidMerge(*target != before, *target != incoming)
    }
}

fn main() {
    let mut analysis = TensorAnalysis::default();
    analysis.inputs.insert("X".into(), TensorType {
        shape: vec![2, 3], dtype: DType::F32,
    });
    analysis.inputs.insert("Y".into(), TensorType {
        shape: vec![2, 3], dtype: DType::I32,
    });
    let mut graph = EGraph::new(analysis);
    let x = graph.add(Tensor::Input("X".into()));
    let y = graph.add(Tensor::Input("Y".into()));
    let sum = graph.add(Tensor::Add([x, y]));
    graph.rebuild();

    // Matching shapes are insufficient: the operand dtypes disagree.
    assert!(graph[graph.find(sum)].data.is_invalid());
}
```

🔎 **What the example does:**

- `Facts` tracks observed types and unknown or invalid metadata. It exposes a
  type only when all alternatives are known and agree.
- `make` reads input types or checks an addition's operands. If their types match,
  the output has the same type. A mismatch marks the result invalid.
- `merge` combines evidence and reports changes to egg through `DidMerge`.
  Its result must be independent of merge order and repeated merges.
- `main` adds `f32[2, 3]` to `i32[2, 3]`. Their shapes match, but their dtypes
  differ, so the analysis marks the addition invalid.

Most of this code maintains analysis state for one small inference rule.
Because facts accumulate, supplying missing input types later requires a fresh
graph. Marking a class invalid cannot undo an incorrect union. Rewrite checks
must use the metadata before applying a transformation.

TEPL provides this analysis infrastructure and generates shape/dtype inference
from dialect declarations. Generated matchers use tensor constraints to reject
incompatible candidates early, and replacement checks reuse the same inference.

</details>

## ✨ TEPL, a language dedicated to tensor rewrites

The transpose-negate rule becomes a single TEPL declaration:

```javascript
from "../dialects/tensor.tepl" import TensorLang as t;

rule transpose_negate {
    X: f32[Dims...]

    (t.transpose[@perm] (t.negate X))
    =>
    (t.negate (t.transpose[@perm] X))
}
```

Save this rule under `examples/sample/rules/`. Its relative import loads operations and
metadata definitions from the [tensor dialect](../examples/sample/dialects/tensor.tepl).

- **Graph structure:** `LHS => RHS` describes the pattern and replacement.
  Operations use nested expressions such as `(t.negate X)`. `X` refers to the
  same captured tensor on both sides.
- **Tensor constraints:** `X: f32[Dims...]` matches an `f32` tensor of any rank.
  Named dimensions enforce equality; dimension sequences support varying ranks.
  Dtype constraints do not insert casts.
- **Operation attributes:** `@perm` captures and reuses the transpose's
  attributes as a descriptor. These are separate from tensor shape and dtype.
- **Checked replacements:** Generated checks validate replacement operations
  and preserve the root's shape and dtype. The author remains responsible for
  the transformation's mathematical correctness.

🧠 **Generated analysis:** Dialects define operations and their shape/dtype
inference. This dialect applies the Rust example's addition checks to TEPL's
numeric dtypes:

```javascript
dialect Elementwise {
    op add(lhs: tensor, rhs: tensor) -> tensor {
        dtype(a, b) {
            assert a == b;
            assert is_numeric(a);
            yield a;
        }
        shape(l, r) {
            assert l == r;
            yield l;
        }
    }
}
```

TEPL generates inference code for both e-class analysis and replacement
validation. Its runtime handles unknown, invalid, and conflicting metadata.
The host supplies input types and any required inference missing from the dialect.

🔒 **Conditions and derivations:** `where` states legality conditions; `derive`
computes replacement attributes. Calls like `$is_reassociable(...)` run
application-specific Rust or C++ functions. The [LoRA rule](../examples/sample/rules/lora.tepl)
uses these blocks to check reassociation and derive contraction attributes.
Floating-point reassociation requires explicit host permission.

🔌 **Host integration:** Define or import dialects, write rules, generate code,
and run it with an e-graph optimizer. TEPL supports multiple dialects and custom
analysis hooks. The host chooses rules, numerical policy, search limits, and
extraction cost model.

## 📚 Learn more

- [Design philosophy](design_philosophy.md): principles and core concepts.
- [Example rules](../examples/sample/rules): tensor rewrites in TEPL.
- [Rust runtime guide](../labs/rust-egg/README.md): analysis and rewrites with egg.
- [C++ runtime guide](../templates/cpp/README.md): generated code for egg-c.
