# Reusable rules

`rules/abstract.tepl` declares `cancel_twice`, parameterized by a unary operation.
`rules/simplify.tepl` binds that parameter to negation and adds a specialization
for small `f32` vectors.

## Run

```sh
bazel-bin/tepl check examples/advanced/reusable_rules
bazel-bin/tepl rewrite examples/advanced/reusable_rules --rule cancel_small_vectors \
  --trace -o /tmp/reused.tepl
```

The two negations disappear, leaving `yield X`. The concrete rule requires an
`f32` vector with at most 1,024 elements. The operation signature alone does not
establish that applying an operation twice is an identity; choose valid instances.

Read [reusable rules](../../../docs/language/reusable-rules.md).
