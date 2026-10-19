# Example projects

Each TEPL project has a `tepl.toml`, `dialects/` and `rules/` directories, plus
optional `graphs/`. The config records `tepl-version = "0.1.0"` and marks the
project root. Pass the project directory to `tepl check` or `tepl generate`.

- [semantic](semantic/): operation semantics, symbolic verification, and explicit or synthesized attributes.
- [sample](sample/): tensor and scalar dialects, shape and dtype programs,
  reusable rules, LoRA rewrites, and concrete graph builders.
- [rewrite](rewrite/): compiled programs and greedy graph rewriting.

Rules and graphs use project-root imports:

```tepl
from "dialects/tensor.tepl" import TensorLang as t;
from "rules/abstract.tepl" import {commute};
```

Paths resolve from the nearest `tepl.toml`, independently of the importing
file's depth or your working directory. `rewrite/auto.tepl` uses an explicit
cross-project import, `"../semantic/dialects/tensor.tepl"`, to share the semantic
dialect. The `research/` directory contains notes and scripts rather than a TEPL
project.

Run the sample from the repository root:

```sh
bazel-bin/tepl check examples/sample
bazel-bin/tepl generate examples/sample/rules/simple.tepl --out my_app/src/generated
```

To add another project, run `bazel-bin/tepl init examples/<name>`, then add your
`.tepl` files. This creates the config and starter dialect and rule. TEPL projects
need no Bazel configuration.

Extended rule expressions and descriptor constructors currently support frontend
checking. Their code generation and execution support is deferred.
