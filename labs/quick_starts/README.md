# TEPL quick start

This demo saturates an e-graph with three rules, then extracts an expression
with minimum AST depth. The initial expression contains addition and vector dot
products:

```text
(dot(x, y) + dot(x, z)) + dot(y, z)
```

The TEPL project contains:

- [`my_dialect.tepl`](my_pattern/dialects/my_dialect.tepl) defines `MyDialect` with `add` and `dot`. Only `dot` has an attribute: `axis`.
- [`example.tepl`](my_pattern/graphs/example.tepl) builds the initial graph and yields its root.
- [`your_rule.tepl`](my_pattern/rules/your_rule.tepl) defines `associate_add`, `swap_add`, and `swap_dot`.

The addition rules demonstrate reassociation and swapping operands. The dot
rule captures its attribute as `@dot`, reuses it on the right, and calls
`$supports_axis(@dot.axis)`.

Rust implements that host function in [`src/main.rs`](src/main.rs). It accepts
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
../../bazel-bin/tepl check my_pattern
../../bazel-bin/tepl generate my_pattern --out src/generated
cargo run
```

Run the generation command again after editing a TEPL file. Generated
sources and Cargo build outputs are ignored by Git.

## 🔄 Saturate and extract

The graph builder is imported at the top of [`src/main.rs`](src/main.rs).
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
