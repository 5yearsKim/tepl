# TEPL examples

Start with one small file, learn an advanced feature, then explore a complete project.
Run all commands from the repository root after `bazel build //:tepl`.

| Start here | What you get |
| --- | --- |
| [Basic](basic/README.md) | Standalone programs for patterns, shapes/dtypes, attributes, and sharing |
| [Advanced](advanced/README.md) | Focused projects for verification, axis inference, reusable rules, and coordinates |
| [Complete projects](projects/README.md) | Transpose simplification and a LoRA transformation |

## Run your first example

```sh
bazel-bin/tepl rewrite examples/basic/double_negation.tepl --trace -o /tmp/result.tepl
```

The result is a graph that yields its original input `X`.
Each example's README explains its purpose, command, and expected result.

## Choose a file or a project

Basic examples contain their dialect, rules, and graph in one file. Run each file
individually; `basic/` is a collection, not one combined TEPL project.

Advanced examples and complete projects each have their own `tepl.toml`,
`dialects/`, `rules/`, and `graphs/`. Run the individual project directory.
Imports are local to that project, and arithmetic policy is set in its manifest.

Generate Rust or C++ from the same input when you need an e-graph integration:

```sh
bazel-bin/tepl generate examples/projects/transpose --out /tmp/app/src/generated
```

See [runtime integration](../docs/runtime/index.md). Host applications live in
[labs](../labs/README.md); broad regression catalogs live in
[tests/fixtures](../tests/fixtures/README.md).
