# Basic examples

Each file is a complete program: dialect, rule, and concrete graph.
Read them in this order. These examples use explicit shape/dtype programs and
attribute calculations, without `axes` or `semantics` blocks.

| File | Learn | Expected result |
| --- | --- | --- |
| [double_negation.tepl](double_negation.tepl) | Match a nested pattern and reuse a capture | Both negations disappear; output is `X` |
| [shapes_and_dtypes.tepl](shapes_and_dtypes.tepl) | Bind a shared dtype and matrix dimensions | Output adds `X` and `Y`, without negations |
| [attributes_and_conditions.tepl](attributes_and_conditions.tepl) | Capture attributes, use `where`, and compute a descriptor with `derive` | Two transposes simplify to `X` |
| [shared_graphs.tepl](shared_graphs.tepl) | Share intermediate computations with `let` | One shared addition feeds both operands of the final addition |

From the repository root:

```sh
bazel-bin/tepl check examples/basic/double_negation.tepl
bazel-bin/tepl rewrite examples/basic/double_negation.tepl --trace -o /tmp/result.tepl
```

Replace the filename to try another row. Run files individually: combining all
of `basic/` would load unrelated declarations with repeated names.
These rules express identities chosen by the author; metadata checking alone
does not establish value equivalence.

Continue with [advanced examples](../advanced/README.md), or read
[Basic language concepts](../../docs/language/core-concepts.md).
