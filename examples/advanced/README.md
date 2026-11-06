# Advanced examples

Each directory is an independent project with a concrete graph and runnable rules.
Start after the [Basic examples](../basic/README.md).

| Project | Learn | Default arithmetic policy |
| --- | --- | --- |
| [semantic_verification](semantic_verification/README.md) | Declare axes/value contracts and distinguish strict from algebraic proofs | `algebraic` |
| [axis_inference](axis_inference/README.md) | Compare explicit and inferred replacement permutations | `strict` |
| [reusable_rules](reusable_rules/README.md) | Parameterize an operation and specialize a rule | `strict` |
| [coordinate_rewrites](coordinate_rewrites/README.md) | Compose slices using explicit coordinate offsets | `strict` |

Run commands from the repository root. A project's `tepl.toml` sets its default
policy; pass `--semantic-policy` to override it. The algebraic policy uses supported
ideal-arithmetic laws and does not promise IEEE floating-point equivalence.

Each project uses only the operations needed for its topic. See
[complete projects](../projects/README.md) for larger applications.
