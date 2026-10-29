# Example projects

These directories contain runnable TEPL projects. A `tepl.toml` marks each root;
`dialects/`, `rules/`, and optional `graphs/` hold its declarations.

| Project | Demonstrates |
| --- | --- |
| [sample](sample/README.md) | Multiple dialects, metadata contracts, pure expressions, descriptors, concrete graphs |
| [semantic](semantic/README.md) | Axis/value semantics, verification, and field-level attribute synthesis |
| [rewrite](rewrite/README.md) | Portable compilation, greedy rewriting, and transpose composition |

Run from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl check examples/sample
bazel-bin/tepl generate examples/sample --out /tmp/app/src/generated
```

Use `--target cpp` for C++ output. Both generated targets support completed rule
expressions and descriptor construction. Add the appropriate
[Rust](../runtime/rust/README.md) or [C++](../runtime/cpp/README.md) shared runtime.

Bare imports resolve against the nearest `tepl.toml`; explicit `./` and `../`
imports remain relative to the importing file. `rewrite/auto.tepl` explicitly
shares the semantic dialect from a neighboring project. Create another project
with `bazel-bin/tepl init examples/<name>`.

Research notes and finite theorem checks now live in
[research/](../research/README.md); measured experiments and applications live in
[labs/](../labs/README.md).
