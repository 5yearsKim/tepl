# TEPL design philosophy

TEPL is a declarative language for tensor graph rewrites, guided by four principles.

- **Make rewrites readable.**
  Express graph structure, transformation intent, and tensor metadata clearly,
  with explicit shape, dtype, and legality constraints.

- **Design for host integration.**
  Fit into the host's architecture and semantic model. Delegate complex logic,
  including tensor semantics, numerical legality, and metadata inference, to
  host functions while keeping rewrite expressions simple.

- **Extensible architecture.**
  Use shape-generic rules to express rewrites across tensor sizes and ranks, and
  rule polymorphism to reuse transformations across operations and host functions.
  Extend the operation vocabulary through dialects while keeping the core language small.

- **Validate early, search efficiently.**
  Validate rewrite definitions before search wherever possible. Use explicit
  constraints to reject candidates as soon as the necessary information is
  available, retaining runtime checks for match-dependent and host-defined semantics.

## Core concepts

### Declarative graph rewrites

A rule describes a matched graph and its replacement with `LHS => RHS`.
Operations use S-expressions; repeated names refer to the same captured value.

The [LoRA example](../examples/sample/rules/lora.tepl) combines graph structure,
shape constraints, and host functions:

```javascript
from "../dialects/tensor.tepl" import TensorLang as t;
use t::{add, dot};

rule lora {
    X: [Batch..., M, K]
    W: [WeightBatch..., K, N]
    A: [WeightBatch..., K, R]
    B: [WeightBatch..., R, N]

    (dot[@outer] X (add W (dot[@inner] A B)))
    =>
    (add (dot[@xw] X W)
         (dot[@out] (dot[@xa] X A) B))

    where {
        $is_broadcastable(Batch, WeightBatch);
        $is_reassociable(X, A, B, @outer, @inner);
    }

    derive {
        @xw = $infer_dot(X, W, @outer);
        @xa = $infer_dot(X, A, @outer);
        @out = $infer_lora_out(X, A, B, @outer, @inner);
    }
}
```

Import paths are relative to the rule file. This example uses the layout under
`examples/sample/rules/`.

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

### Host functions and operator descriptors

`$name(...)` calls a host function. The host supplies tensor semantics,
numerical legality, and specialized inference.

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
    K % 128 == 0;
    $is_reassociable(X, A, B, @outer, @inner);
}
derive {
    @xa = $infer_dot(X, A, @outer);
}
```

Both blocks use matched inputs, captured descriptors, and shape variables.
Derivations run in source order and cannot inspect newly constructed RHS values.
Numerical transformations such as floating-point reassociation require explicit
host permission.

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
    where { N <= 1024; }
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
