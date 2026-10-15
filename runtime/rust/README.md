# TEPL Rust runtime

Shared execution support for TEPL-generated egg programs. The crate uses egg
0.11.0 and Rust edition 2024. Generated modules contain operation types and
program-specific functions; the runtime contains reusable algorithms.

Add the crate to an application:

```toml
[dependencies]
tepl-runtime = { path = "/path/to/tepl/runtime/rust" }
```

Generate the example module from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl generate examples/rewrite --out /path/to/app/src/generated
```

Run its rules:

```rust
mod generated;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let built = generated::graphs::transpose::graph_example::build()?;
    let rules = generated::rules::all()?;
    let runner = tepl_runtime::runner::runner(built.graph)?;
    let result = tepl_runtime::runner::run(runner, &rules)?;
    println!("{:?}", result.egraph[built.root].data);
    Ok(())
}
```

For manual graph construction, create `TensorAnalysis`, register input facts,
and then insert input and operation nodes. `OpNode::operation` accepts a dialect
operation, optional typed attributes, and operand IDs. Attribute schema structs
convert to the combined attribute enum through `Into`.

```rust
use generated::dialects::tensor_lang::{Op, TransposeAttrs};

let node = generated::OpNode::operation(
    Op::Transpose,
    Some(TransposeAttrs { permutation: vec![1, 0] }.into()),
    vec![input_id],
)?;
```

`TensorFacts` holds independent shape and dtype facts: `Unknown`, `Known`, or
`Conflict`. Equal extents do not identify axes. Axis identities are temporary
values used to evaluate operation contracts and synthesized attributes.

False conditions skip a rewrite. Missing required metadata also skips it.
Evaluation errors produce diagnostics, and conflicting metadata stops execution.
A replacement with a different known output shape or dtype is rejected before
insertion. Use `runner::runner` and `runner::run` to receive errors from egg
callbacks; direct egg integration can inspect `graph.analysis.check()`.

`runner::runner` returns an egg runner, so applications can set its iteration,
node, and time limits before passing it to `runner::run`. Cost models and
extraction remain application choices.

The generated/runtime ABI is checked at Rust compile time. No dynamic semantic
library, host callback implementation, JSON interpreter, or runtime solver is
required.

```sh
cargo test --manifest-path runtime/rust/Cargo.toml --locked
```
