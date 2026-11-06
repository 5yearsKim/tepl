# Reusable rules

An abstract rule can parameterize operations. Concrete rules bind those
parameters to operations from a dialect:

```tepl
abstract rule cancel_twice(F: op<(tensor) -> tensor>) {
    (F (F X)) => X
}

rule cancel_double_negate extends cancel_twice(F = t.negate)

rule cancel_small_vectors extends cancel_twice(F = t.negate) {
    X: f32[N]
    where N <= 1024
}
```

Here `t` is an imported dialect alias. The operation parameter's signature
requires one tensor operand and a tensor result. It does not establish that
applying every such operation twice is an identity: each concrete instance must
be valid for the chosen operation and numerical policy.

A concrete rule inherits the pattern and replacement. It can add constraints
and conditions; it cannot replace the inherited graph or `derive` block.

Use abstraction when several rules share a transformation. Keep a single rule
concrete when a parameter would obscure its meaning. Run the
[reusable rules project](../../examples/advanced/reusable_rules/README.md), whose
[abstract definition](../../examples/advanced/reusable_rules/rules/abstract.tepl)
and [instances](../../examples/advanced/reusable_rules/rules/simplify.tepl)
are kept in separate files.
