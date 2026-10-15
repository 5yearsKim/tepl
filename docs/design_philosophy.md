# TEPL design philosophy

TEPL is a declarative language for tensor graph rewrites, guided by four principles.

- **Make rewrites readable.**
  Express graph structure, transformation intent, and tensor metadata clearly,
  with explicit shape, dtype, and legality constraints.

- **Keep rules self-contained.**
  Express guards and attribute transformations with pure expressions, reusable
  builtins, descriptor constructors, and semantic synthesis. Applications select
  the numerical policy and supply input metadata.

- **Extensible architecture.**
  Use shape-generic rules to express rewrites across tensor sizes and ranks, and
  rule polymorphism to reuse transformations across operations.
  Extend the operation vocabulary through dialects while keeping the core language small.

- **Validate early, search efficiently.**
  Validate rewrite definitions before search wherever possible. Use explicit
  constraints to reject candidates as soon as the necessary information is
  available, retaining runtime checks for match-dependent metadata.

## Core concepts

### Declarative graph rewrites

A rule describes a matched graph and its replacement with `LHS => RHS`.
Operations use S-expressions; repeated names refer to the same captured value.

The [attribute examples](../examples/sample/rules/attributes.tepl) combine
patterns, builtins, and explicit attribute construction:

```tepl
from "../dialects/tensor.tepl" import TensorLang as t;

rule compose_transposes {
    (t.transpose[@outer] (t.transpose[@inner] X))
    => (t.transpose[@combined] X)
    derive {
        @combined = {
            ..@outer,
            permutation: gather(@inner.permutation, @outer.permutation)
        };
    }
}
```

Import paths are relative to the rule file. Constructors are checked by the
frontend; their Rust/C++ execution support is deferred.

### Tensor constraints and shape-generic rules

Declarations constrain matches. Named dimensions express equality; named
sequences support varying ranks.

```javascript
X: [Batch..., M, K]   // Any batch rank, followed by dimensions M and K
W: [K, N]            // Shares dimension K with X
Y: bf16[M, N]        // Restricts both shape and dtype
Z: f32[...]          // Any shape, with dtype f32
S: f32[]             // A rank-zero tensor
```

`_` matches one arbitrary dimension; `...` matches zero or more.
Each shape permits at most one sequence. Dtype annotations never insert casts
or imply promotion; an omitted dtype is unrestricted.

### Builtins and operator descriptors

Calls resolve to reusable builtins. Rules do not call application functions.

Descriptors hold operation attributes. A pattern captures them with `@name`;
the replacement can reuse them or use derived descriptors.

```javascript
(dot[@outer] X W)
```

Here, `@outer` captures the dot operation's attributes, separately from `X`
and `W`. Descriptors contain operation attributes, not output tensor metadata.

### Explicit conditions and derivations

`where` states when a rewrite is legal. `derive` computes attributes for
replacement operations.

```javascript
where {
    K % 128 == 0
}
derive {
    @xa = { ..@outer };
}
```

Both blocks use matched inputs, captured descriptors, and shape variables.
Derivations run in source order and cannot inspect newly constructed RHS values.
Numerical transformations such as floating-point reassociation require explicit
numerical policy.

### Extensible dialects and polymorphic rules

Dialects define operations and attribute schemas without expanding the core
language syntax.

```javascript
dialect TensorLang {
    op add(lhs: tensor, rhs: tensor) -> tensor;
    op multiply(lhs: tensor, rhs: tensor) -> tensor { alias: mul; }
    op concatenate(first: tensor, second: tensor, rest: tensor...) -> tensor {
        attrs { axis: index; }
    }
}
```

Abstract rules parameterize operations (`op`) and host functions (`fn`).
Concrete rules bind those parameters and may add constraints.

```javascript
abstract rule commute(F: op<(tensor, tensor) -> tensor>) {
    (F X Y) => (F Y X)
}

// Requires host semantics in which swapping add operands is valid.
rule commute_add extends commute(F = t.add);

rule commute_small_vectors extends commute(F = t.add) {
    X: [N]
    Y: [N]
    where { N <= 1024 }
}
```

Here, `t` is the imported dialect alias from the LoRA example.
Instances preserve the inherited graph and combine its constraints with their
own; they cannot replace the graph or its `derive` block.

### Validation and constrained search

- **Before search:** validate operation names, arity, types, descriptor references,
  and concrete rule instances.
- **During search:** use shape and dtype constraints and eligible pure `where`
  conditions to reject candidates as bindings become available.
- **Before insertion:** finish legality checks and derivations, validate the
  complete replacement, and require its root shape and dtype to match the original.

A failed check or missing required metadata rejects the match. Rejected matches
leave no partial replacement in the graph.
