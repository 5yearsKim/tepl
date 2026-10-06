# 🚀 Run your first TEPL code

Let's build a small optimizer for addition and vector dot products. You will
write the operations, rewrite rules, and starting graph in TEPL, then run the
generated code with Rust and egg. This guide covers the basic example;
tensor shape and dtype analysis are covered separately.

## 🔧 Make `tepl` available

From the TEPL repository root, add the build directory to your `PATH`:

```sh
export PATH="$PWD/bazel-bin:$PATH"
tepl --help
```

You can now run `tepl` from any directory in this terminal. To keep it available
in new terminals, add the same `export` command to `~/.bashrc` or `~/.zshrc`,
replacing `$PWD/bazel-bin` with the absolute path to your build directory.

You can also move or copy the binary to `~/.local/bin` or another directory.
Make sure that directory is on your `PATH`.

## 🦀 Set up a Rust project

You can try the finished example in
[`labs/tutorial_rust`](../../../labs/tutorial_rust/README.md). From the repository root:

```sh
cd labs/tutorial_rust
tepl generate pattern_basic --out src/tepl_pattern_basic
cargo run --bin pattern_basic
```

The steps below assume you are starting from scratch. With Rust and Cargo
installed, create a project using edition 2024 and add egg:

```sh
cargo new my_optimizer --edition 2024
cd my_optimizer
cargo add egg@0.11
```

All remaining commands run from your new Rust project root.

## 📁 Add the TEPL files

Initialize a TEPL project, rename its starter files, and add a graph directory:

```sh
tepl init pattern_basic
mv pattern_basic/dialects/your_dialect.tepl pattern_basic/dialects/my_dialect.tepl
mv pattern_basic/rules/your_rule.tepl pattern_basic/rules/add_dot.tepl
mkdir -p pattern_basic/graphs
```

Your TEPL project will contain:

```text
pattern_basic/
├── dialects/
│   └── my_dialect.tepl
├── rules/
│   └── add_dot.tepl
└── graphs/
    └── add_dot_sample.tepl
```

### 1. Define the operations

Replace `pattern_basic/dialects/my_dialect.tepl` with:

```javascript
dialect MyDialect {
    op add(lhs: tensor, rhs: tensor) -> tensor;

    op dot(lhs: tensor, rhs: tensor) -> tensor {
        attrs {
            axis: index;
        }
    }
}
```

The dialect names the operations and their operands. Both take two tensors and
return a tensor. `dot` also carries an `axis` attribute of type `index`.
This basic dialect has no shape or dtype inference. These declarations describe
nodes; they do not implement numerical execution.

### 2. Write the rewrite rules

Replace `pattern_basic/rules/add_dot.tepl` with:

```javascript
from "../dialects/my_dialect.tepl" import MyDialect as d;

rule associate_add {
    (d.add (d.add X Y) Z) => (d.add X (d.add Y Z))
}

rule swap_add {
    (d.add X Y) => (d.add Y X)
}

rule swap_dot {
    (d.dot[@dot] X Y) => (d.dot[@dot] Y X)

    where {
        $supports_axis(@dot.axis);
    }
}
```

The import gives the dialect the short name `d`. Expressions put the operation
first: `(d.add X Y)` means `X + Y`. Capitalized names capture matched tensors,
and `=>` adds an equivalent expression to the e-graph.

The first two rules regroup addition and swap its operands. The last rule
captures the dot attributes as `@dot` and reuses them on the right. Its `where`
block calls a Rust host function, `$supports_axis`, to decide whether swapping
is allowed. This demo assumes semantics where these rewrites are valid.

### 3. Build the starting graph

Create `pattern_basic/graphs/add_dot_sample.tepl`:

```javascript
from "../dialects/my_dialect.tepl" import MyDialect as d;

graph add_dot_sample {
    input x;
    input y;
    input z;

    let a = (d.dot[axis = 0] x y);
    let b = (d.dot[axis = 0] x z);
    let c = (d.dot[axis = 0] y z);
    // An unused node lets us see the host condition reject axis 1.
    let unsupported = (d.dot[axis = 1] x y);
    yield (d.add (d.add a b) c);
}
```

`input` declares external tensors, `let` names nodes, and `yield` selects the
output. The initial expression is `(dot(x, y) + dot(x, z)) + dot(y, z)`.
Inputs have no shape or dtype annotations here. All declared nodes are inserted,
including the unused `unsupported` node.

## ⚙️ Generate Rust

Check the TEPL project and generate a Rust module:

```sh
tepl check pattern_basic
tepl generate pattern_basic --out src/tepl_pattern_basic
```

The first command prints `CheckedProgram`; the second writes operation types,
rule builders, and graph builders. Run generation again after editing TEPL.
Add `/src/tepl_pattern_basic/` to your project's `.gitignore`.

## 🦀 Run the optimizer

Download [utils.rs](../../../labs/tutorial_rust/src/utils.rs) and save it as
`src/utils.rs` for the readable e-graph printer.

Replace `src/main.rs` with:

```rust
// Generated helpers may be unused.
#[allow(dead_code, unused_imports, unused_variables)]
mod tepl_pattern_basic;
mod utils;

use egg::{AstDepth, EGraph, Extractor, Runner, StopReason};
// Module alias for utils.rs.
use tepl_pattern_basic as tepl;
use tepl_pattern_basic::OpNode;
use tepl_pattern_basic::graphs::add_dot_sample::graph_add_dot_sample;
use tepl_pattern_basic::rules::add_dot::{rule_associate_add, rule_swap_add, rule_swap_dot};
use utils::print_egraph;

struct Host;

impl rule_swap_dot::HostFunctions for Host {
    // Implements TEPL's $supports_axis.
    fn supports_axis(&self, axis: u64) -> Option<bool> {
        Some(axis == 0)
    }
}

fn main() {
    // Build the starting graph.
    let graph_def = graph_add_dot_sample::build().unwrap();

    // Build rules; swap_dot uses the host callback.
    let rules = [
        rule_associate_add::build(()).unwrap(),
        rule_swap_add::build(()).unwrap(),
        rule_swap_dot::build(Host).unwrap(),
    ];

    // No tensor metadata analysis.
    let mut graph = EGraph::<OpNode, ()>::default();
    let built = graph_def.insert_nodes(&mut graph).unwrap();

    // Run rewrites to saturation.
    let runner = Runner::default().with_egraph(graph).run(&rules);

    println!("Stop reason: {:?}", runner.stop_reason);
    assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    print_egraph(&runner.egraph, built.root);

    // Extract a minimum-depth expression.
    let (depth, best) = Extractor::new(&runner.egraph, AstDepth).find_best(built.root);
    println!("\nMinimum AST depth: {depth}");
    println!("Best expression:\n{}", best.pretty(80));
}
```

`Host` accepts axis `0`; `Some(false)` or `None` rejects a rewrite candidate.
Rules without host functions use `build(())`. The `tepl` alias lets `utils.rs`
access the generated types; it is not another dependency.

The program creates an e-graph with unit analysis (`()`), inserts the starting
nodes, and runs the rules. The extractor chooses a minimum-depth expression
from the yielded root's e-class. Now run it:

```sh
cargo run
```

## 🔎 Read the result

One run produces this excerpt; e-class IDs and tied extraction choices may vary:

```text
Stop reason: Some(Saturated)

Saturated e-graph (nodes in each e-class are equivalent):
...
e3
  dot[axis=0](e0, e1)
  dot[axis=0](e1, e0)
...
e6
  dot[axis=1](e0, e1)
...
e8 (root)
  add(e16, e4)
  add(e3, e9)
  add(e4, e16)
  add(e5, e7)
  add(e7, e5)
  add(e9, e3)
...

Minimum AST depth: 4
Best expression:
(add None
  (dot MyDialect(DotAttrs { axis: 0 })
    <input> Input { name: "x" }
    <input> Input { name: "z" })
  (add None
    (dot MyDialect(DotAttrs { axis: 0 })
      <input> Input { name: "x" }
      <input> Input { name: "y" })
    (dot MyDialect(DotAttrs { axis: 0 })
      <input> Input { name: "y" }
      <input> Input { name: "z" })))
```

`Saturated` means the rules found no more new equivalences. Each `e` block is an
e-class: its nodes are equivalent, and their child IDs refer to other e-classes.
Here, `e0`, `e1`, and `e2` contain inputs `x`, `y`, and `z`. The two nodes in `e3`
show an accepted dot swap. The axis-`1` node in `e6` has no swapped alternative,
because the host rejected it. The root class contains several addition orders.

The extracted expression above is `dot(x, z) + (dot(x, y) + dot(y, z))`.
`None` means an addition node has no attributes. Input leaves have depth `1`,
dot products depth `2`, and the two levels of addition give depth `4`.
All arrangements of these three dot products tie at that depth, so returning
the original arrangement is also valid. This demo explores equivalent forms;
it does not evaluate tensor values or promise a faster computation.
