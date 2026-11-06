# LoRA transformation

Rewrite `X(W + AB)` as `XW + (XA)B`. The graph uses `X: f32[2, 4]`,
`W: f32[4, 3]`, and a rank-one adapter `A: f32[4, 1]`, `B: f32[1, 3]`.
Both forms produce shape `[2, 3]`.

## Run

```sh
bazel-bin/tepl check examples/projects/lora
bazel-bin/tepl rewrite examples/projects/lora --trace -o /tmp/lora.tepl
```

The result contains the two branches `XW` and `(XA)B`, followed by addition.
TEPL constructs the replacement contraction attributes from the declared
conditions and operation contracts.

`tepl.toml` explicitly selects `algebraic`: this transformation requires ideal
arithmetic distribution and reassociation. It does not promise bit-identical
floating-point results or lower cost for every shape and target.

To explore alternatives using a cost model, generate the same rules for
[egg or egg-c](../../../docs/runtime/index.md).
