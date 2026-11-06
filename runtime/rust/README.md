# TEPL Rust runtime

Shared execution support for generated egg programs. Requires Rust edition 2024
and egg 0.11.0. Generated modules contain program-specific types and functions;
this crate supplies reusable execution algorithms.

## Generate and consume

Add the runtime to the application:

```toml
[dependencies]
tepl-runtime = { path = "/path/to/tepl/runtime/rust" }
```

```sh
bazel build //:tepl
bazel-bin/tepl generate examples/projects/transpose --out /path/to/app/src/generated
```

Run a generated graph and its rules:

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

See the [Rust/egg application](../../labs/rust-egg/README.md) for a complete demo.

## Integration API

`Program` is the generated/runtime trait. `Node<Program>`, `TensorAnalysis`,
`TensorFacts`, and rule helpers use it without depending on the compiler or JSON.
For manual construction, register input facts before inserting input nodes.
`OpNode::operation` accepts a dialect operation, optional typed attributes, and
operand IDs; schema structs convert to the combined attribute enum with `Into`.

Fallible functions return `EvalResult<T>`. Invalid candidates are recorded in
`graph.analysis.rejections()` with rule/source information. `runner::runner`
returns an egg runner so applications can set iteration, node, and time limits;
`runner::run` propagates callback errors. Direct egg users can inspect
`graph.analysis.check()`. Extraction and cost models remain application choices.

The [shared execution contract](../../design/code-generation.md#shared-execution-contract)
defines fact merging, missing metadata, rejection, evaluation order, and
replacement validation. Runtime ABI 2 is checked at compile time; regenerate
modules when updating it. C++ and Rust ABI versions are independent.

## Validate

```sh
cargo test --manifest-path runtime/rust/Cargo.toml --locked
bazel test //tests:rust_codegen_test
```
