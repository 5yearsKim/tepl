# TEPL quick start

This lab has two binaries: `pattern_basic` for structural rewriting and
`pattern_analyze` for rewriting with tensor shape and dtype analysis.

## Basic example

The basic demo saturates an e-graph with three rules, then extracts an expression
with minimum AST depth. The initial expression contains addition and vector dot
products:

```text
(dot(x, y) + dot(x, z)) + dot(y, z)
```

The TEPL project contains:

- [`my_dialect.tepl`](pattern_basic/dialects/my_dialect.tepl) defines `MyDialect` with `add` and `dot`. Only `dot` has an attribute: `axis`.
- [`add_dot_sample.tepl`](pattern_basic/graphs/add_dot_sample.tepl) builds the initial graph and yields its root.
- [`add_dot.tepl`](pattern_basic/rules/add_dot.tepl) defines `associate_add`, `swap_add`, and `swap_dot`.

The addition rules demonstrate reassociation and swapping operands. The dot
rule captures its attribute as `@dot`, reuses it on the right, and calls
`$supports_axis(@dot.axis)`.

Rust implements that host function in [`src/basic.rs`](src/basic.rs). It accepts
axis `0` for this demo. Returning `Some(false)` or
`None` rejects a rewrite candidate.

## Run

From the repository root, build the compiler once:

```sh
bazel build //:tepl
```

Then check the TEPL project, generate Rust, and run the demo:

```sh
cd labs/quick_starts
../../bazel-bin/tepl check pattern_basic
../../bazel-bin/tepl generate pattern_basic --out src/tepl_pattern_basic
cargo run --bin pattern_basic
```

Run the generation command again after editing a TEPL file. Generated
sources and Cargo build outputs are ignored by Git.

## 🔄 Saturate and extract

The graph builder is imported at the top of [`src/basic.rs`](src/basic.rs).
`main` prepares the graph definition and rules, inserts the graph, and runs
`egg::Runner` until no new equivalences are found.

The program prints the stop reason and saturated e-graph, then uses
`Extractor::new(&runner.egraph, AstDepth)` to select a minimum-depth expression
from the yielded root's e-class. The dump includes unused nodes, such as the
axis-`1` dot rejected by the host condition. Extraction starts from the root.

Leaves have depth `1`. This example has minimum depth `4`; all equivalent
arrangements of its three dot products tie at that depth, so the extractor may
return the original arrangement. AST depth is a structural cost, independent
of execution speed.

There are no input shape or dtype declarations in the TEPL project. The demo
uses `EGraph<OpNode, ()>` and structural rewrites. A project with tensor metadata
can use generated tensor analysis and the same rule builders.

## Tensor analysis example

[`pattern_analyze`](pattern_analyze) keeps the same `dialects/`, `graphs/`, and
`rules/` structure. Its three files cover the core concepts:

- [`my_dialect.tepl`](pattern_analyze/dialects/my_dialect.tepl) defines shape and dtype inference for `add` and `dot`. Addition preserves equal input shapes; matrix dot maps `[M, K]` and `[K, N]` to `[M, N]`. Both operations require matching numeric dtypes.
- [`add_dot_sample.tepl`](pattern_analyze/graphs/add_dot_sample.tepl) declares `x: i32[2, 3]` and `y, z: i32[3, 4]`, then yields `add(dot(x, y), dot(x, z))`.
- [`add_dot.tepl`](pattern_analyze/rules/add_dot.tepl) includes the basic example's `associate_add`, `swap_add`, and `swap_dot` rules, plus an abstract factoring rule and a concrete rule that extends it. `T` binds the shared dtype, and `M`, `K`, and `N` bind matrix dimensions. The concrete rule limits the contracting dimension with `K <= 8`.

Each dot contracts the left matrix's axis `1` with the right matrix's axis `0`.
The dialect checks that these dimensions agree. The initial dots infer
`i32[2, 4]`; after factoring, `add(y, z)` infers `i32[3, 4]`, and the final dot
still produces `i32[2, 4]`. The Rust output prints these inferred shapes.

The inherited rule captures each dot's attributes and checks that their axes
match. This demo uses `i32` matrices with wrapping arithmetic.

The binary runs all four concrete rules. `swap_dot` retains its
`$supports_axis(@dot.axis)` host condition from the basic example. Rust permits
swapping only axis-`0` vector dots, so it rejects the axis-`1` matrix dots in
this graph. Matrix multiplication generally does not commute. `swap_add` adds
both operand orders to the saturated graph; either order may be extracted.

From `labs/quick_starts`, generate and run the second binary:

```sh
../../bazel-bin/tepl check pattern_analyze
../../bazel-bin/tepl generate pattern_analyze --out src/tepl_pattern_analyze
cargo run --bin pattern_analyze
```

The application collects declared metadata with `input_bindings()`, creates
`TensorAnalysis`, constructs the egg e-graph, and calls `insert_nodes()`:

```rust
let graph_def = graph_add_dot_sample::build()?;
let bindings = graph_def.input_bindings()?;
let analysis = TensorAnalysis::new(bindings);
let mut graph = EGraph::new(analysis);
let built = graph_def.insert_nodes(&mut graph)?;
```

The basic example creates an `EGraph<OpNode, ()>` and uses the same
`insert_nodes()` method. The binary in [`src/analyze.rs`](src/analyze.rs)
prints the inferred dtype and shape for each named node, the saturated
e-graph, and the expressions before and after extraction:

Both binaries use `print_egraph` from [`src/utils.rs`](src/utils.rs). It lists
equivalent nodes on separate lines and marks the root. The basic example has
no tensor metadata; the analysis example also displays inferred tensor types.
Each binary aliases its generated module as `tepl` for the shared utilities.

The analysis graph display lists each e-class with its tensor type and equivalent
nodes on separate lines. References such as `e3` identify child e-classes,
and `(root)` marks the output. Dot attributes remain visible:

```text
e5: i32[2, 4] (root)
  add(e3, e4)
  add(e4, e3)
  dot[axis=1](e0, e7)
```

| Stage | Expression | AST size | Output metadata |
| --- | --- | --- | --- |
| Before | `add(dot(x, y), dot(x, z))` | 7 | `i32[2, 4]` |
| After | `dot(x, add(y, z))` | 5 | `i32[2, 4]` |

AST size counts expression nodes, including repeated inputs. It is a simple
structural cost rather than a runtime estimate. The binary checks that the
graph saturates and preserves output metadata. It rewrites expressions and
infers their types; it does not evaluate tensor values.

Generated sources live in `src/tepl_pattern_basic/` and
`src/tepl_pattern_analyze/`; both directories are ignored by Git. Plain
`cargo run` still runs the basic demo. Generate both modules before using
`cargo build --bins`.
