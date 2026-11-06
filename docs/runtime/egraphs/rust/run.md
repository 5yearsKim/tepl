# Run generated rules with egg

After [installing TEPL](../../../getting-started/install.md), use the
[transpose tutorial project](../../../../examples/projects/transpose/README.md) with egg.
Read [how e-graphs work](../overview.md) for the search and extraction model.

TEPL generates a Rust module for an existing edition 2024 application. The module
contains operation types, metadata evaluators, and rule functions. Reusable egg
integration lives in the shared `tepl-runtime` crate.

## Set up the application

```sh
cargo new my_app
```

Add the runtime dependency to `my_app/Cargo.toml`, using your checkout's path:

```toml
[dependencies]
tepl-runtime = { path = "/path/to/tepl/runtime/rust" }
```

## Generate the example

From the TEPL repository root:

```sh
bazel build //:tepl
bazel-bin/tepl generate examples/projects/transpose --out /path/to/my_app/src/generated
```

The compiler constructs the composed permutation from the dialect's semantics.
The identity rule removes a transpose only when its permutation preserves axis
order. The generated code evaluates completed attribute expressions on each match.

You can generate the same module from compiled JSON:

```sh
bazel-bin/tepl compile examples/projects/transpose --out /tmp/program.json
bazel-bin/tepl generate /tmp/program.json --out /path/to/my_app/src/generated
```

## Run egg

Replace `my_app/src/main.rs` with:

```rust
mod generated;
use tepl_runtime::egg::{AstSize, Extractor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let built = generated::graphs::transpose::graph_example::build()?;
    let rules = generated::rules::all()?;
    let runner = tepl_runtime::runner::runner(built.graph)?.with_iter_limit(20);
    let result = tepl_runtime::runner::run(runner, &rules)?;
    let (size, best) = Extractor::new(&result.egraph, AstSize).find_best(built.root);
    println!("Best expression ({size} nodes): {}", best.pretty(80));
    println!("Output metadata: {:?}", result.egraph[built.root].data);
    Ok(())
}
```

`AstSize` favors the expression with fewer nodes. Here the best result is the
original input `X`, with output shape `[2, 3]` and dtype `F32`. Unlike the direct
engine, egg also retains the original expression as an equivalent alternative.

Then run:

```sh
cargo run --manifest-path /path/to/my_app/Cargo.toml
```

The generated graph constructor registers input metadata before insertion.
`rules::all()` builds every generated rule. To select one, call its
`definition().build()` method, such as
`generated::rules::transpose::rule_compose_transposes::definition().build()?`.

The checked runner propagates errors from egg callbacks. Applications choose
iteration limits, extraction costs, and how to use the resulting graph.
Candidate evaluation failures are skipped; inspect
`result.egraph.analysis.rejections()` for their rule names, source locations,
and errors. Conflicting metadata and resource limits stop execution.

Next: [supply tensor metadata](analysis.md) when building graphs manually.
