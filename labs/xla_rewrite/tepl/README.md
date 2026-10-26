# HLO TEPL Project

This directory contains all TEPL sources used by the XLA rewrite evaluation.
[tepl.toml](tepl.toml) defines the project root. Imports such as
`from "dialects/hlo.tepl" import HLO as hlo;` resolve from here at any rule depth.

| Directory | Purpose |
|---|---|
| [dialects/](dialects/) | Local HLO profiles with and without value semantics |
| [rules/semantic/](rules/semantic/) | Explicit rules checked for semantic equivalence, plus a false control |
| [rules/structural/](rules/structural/) | The same rule bodies using declarations without value semantics |
| [graphs/](graphs/) | Input graphs for concrete rewrite execution tests |
| [rules/full_auto/](rules/full_auto/) | Generated variants inferring every RHS descriptor |
| [rules/partial_auto/](rules/partial_auto/) | Generated variants inferring a subset of RHS descriptors |

The semantic and structural baselines supply RHS attributes explicitly.
Their counterpart rule bodies differ only in their dialect import. Generated
inference variants omit descriptor definitions; current TEPL uses no `auto`
keyword. Some semantic rules intentionally remain
unproved, so run individual cases rather than checking the whole project.

From the repository root:

```sh
bazel-bin/tepl check labs/xla_rewrite/tepl/rules/semantic/add_zero.tepl --semantic-policy=algebraic
bazel-bin/tepl compile labs/xla_rewrite/tepl/rules/structural/add_zero.tepl -o /tmp/xla-add-zero.json
```

Case IDs, conditions, scope, measured outcomes and reproduction commands live in
[coverage/](../coverage/README.md). The dialect is an algebraic evaluation model,
not a complete HLO importer or an exact IEEE/backend equivalence model.
