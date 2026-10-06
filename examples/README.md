# Example projects

Each subdirectory is a separate TEPL project with its own `dialects/` and
`rules/` directories, plus optional `graphs/`. Pass the project directory to `tepl check` or
`tepl generate`.

- [sample](sample/): tensor and scalar dialects, shape and dtype programs,
  reusable rules, LoRA rewrites, and concrete graph builders.

Run the sample from the repository root:

```sh
bazel-bin/tepl check examples/sample
bazel-bin/tepl generate examples/sample --out my_app/src/generated
```

To add another project, create `examples/<name>/dialects/` and
`examples/<name>/rules/`, then add your `.tepl` files. TEPL projects need no
Bazel configuration.
