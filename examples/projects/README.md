# Complete projects

These projects separate dialect definitions, rules, and graphs. Each has its own
`tepl.toml`; imports resolve from that project's root.

| Project | Transformation | Default arithmetic policy |
| --- | --- | --- |
| [transpose](transpose/README.md) | Compose two transposes and remove an identity transpose | `strict` |
| [lora](lora/README.md) | Rewrite `X(W + AB)` as `XW + (XA)B` | `algebraic` |

From the repository root:

```sh
bazel-bin/tepl rewrite examples/projects/transpose --trace -o /tmp/result.tepl
```

For individual language features, start with [Basic](../basic/README.md) or
[Advanced](../advanced/README.md). Rust/C++ application setup lives in
[runtime integration](../../docs/runtime/index.md).
