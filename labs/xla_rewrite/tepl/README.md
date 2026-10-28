# HLO TEPL Project

[tepl.toml](tepl.toml) defines the project root. Imports resolve from here at
any rule depth.

| Directory | Purpose |
|---|---|
| [dialects/](dialects/) | HLO semantic/structural profiles and singleton-view adapter |
| [rules/axis/](rules/axis/) | Canonical axis-only evaluation inputs |
| [rules/semantic/](rules/semantic/) | Original explicit corpus fixtures and negative control |
| [rules/structural/](rules/structural/) | Original compilation-only counterparts |
| [graphs/](graphs/) | Input graphs for concrete rewrite execution tests |

`hlo_structural.tepl` keeps operation, attribute, dtype, and declared shape
contracts for structural compilation. `hlo.tepl` additionally supplies axis and
value semantics where modeled, enabling equivalence proofs and axis inference.
Shape declarations and `shape(X)` queries remain supported. Shape extents stay
supplied; only axis fields can be automatic.

The active inference tasks use field-level `auto` only for original RHS axis
fields. Extent, coordinate, and general fields remain supplied. There are no
active full-auto or partial-auto descriptor variants.

Unit-only reshape is normalized to supplied-unit broadcast and a singleton-axis
removal adapter. This does not implement general reshape. Generated adapter
positions remain fixed. Some cases are unresolved, so run individual rules
rather than checking the whole project:

```sh
bazel-bin/tepl check labs/xla_rewrite/tepl/rules/axis/broadcast_multiply.tepl --semantic-policy=algebraic --rule broadcast_multiply
```

Case IDs, scope, outcomes, and reproduction commands live in
[coverage/](../coverage/README.md). The dialect is an algebraic model, not a
complete HLO importer or an exact IEEE/backend equivalence model.
