Here’s the summarized architecture I’d use for your Tensor DSL.

## 1. Core goal

The DSL is for **declarative tensor graph rewrites** over an e-graph.

The design principle is:

```text
S-expression  -> graph structure
shape section -> lightweight tensor constraints
binder        -> shared/intermediate expressions
tuple/get     -> multi-output values
@attr         -> compact operator-semantic capture
where         -> legality
derive        -> RHS metadata synthesis
host code     -> complicated tensor semantics
e-graph       -> equivalence + search
```

The DSL should stay small. StableHLO-specific complexity belongs in C++/Rust.

---

## 2. Core expression syntax

All computation is structural and S-expression based:

```lisp
(op[@attr] arg1 arg2 ...)
```

Examples:

```lisp
(add X Y)

(dot[@d] X W)

(reduce[@r] X)

(transpose[@t] X)
```

The S-expression should emphasize **topology**, not detailed operator attributes.

So prefer:

```lisp
(dot[@d] X W)
```

over verbose inline attribute descriptions.

---

## 3. Variable classes

Use distinct concepts:

```text
X, W, A, B      tensor expression variables
?XA             named expression binder
@d              captured operator descriptor / attribute
M, K, N         symbolic dimensions
Batch...        named dimension sequence
...             anonymous dimension sequence
_               exactly one anonymous dimension
Xs...           variadic expression sequence, if needed
```

---

## 4. Tensor shape notation

Shape declarations are optional **pattern constraints**, not a complete type system.

Examples:

```javascript
X: [Batch..., M, K]
W: [K, N]
B: [N]
```

Repeated names unify:

```text
K in X == K in W
N in W == N in B
```

Supported notation:

```text
M            one symbolic dimension
_            one arbitrary dimension
...          zero or more anonymous dimensions
Batch...     zero or more named dimensions
```

Allow the variadic part anywhere:

```javascript
X: [..., M, K]
X: [M, Batch..., K]
X: [M, K, Tail...]
```

Initially, allow at most one variadic segment per tensor pattern to avoid ambiguous matching.

Shape arithmetic stays in `where`:

```javascript
where {
    K % 128 == 0;
}
```

not inside shape syntax.

---

## 5. Generic tensor semantics underneath

The shape notation is only sugar over a generic semantic representation:

```cpp
struct TensorDesc {
    Shape shape;
    ElementType dtype;
    DimensionLineage dims;
};
```

For simple matmul-like cases:

```javascript
X: [..., M, K]
W: [K, N]
```

is useful.

For fully generic `dot_general`, use generic tensors:

```javascript
A: [...]
B: [...]
C: [...]
```

and let captured attributes plus host logic handle batching, contracting, and free dimensions.

---

## 6. Expression binders

Use:

```javascript
?XA = (dot[@xa] X A)
```

This binds the matched or constructed expression result to `?XA`.

Binders can be used on either side.

Example:

```javascript
(add
  (?Y = (dot X W))
  (mul ?Y Z))
```

Repeated references mean the same e-class.

So binder semantics are about **expression identity in the e-graph**, not textual duplication.

---

## 7. Tuple support

Support tuples minimally:

```lisp
(tuple A B C)

(get[0] T)
(get[1] T)
```

This handles multi-output operations while preserving the invariant that every e-node still produces one value.

For example:

```lisp
(get[0] (topk X))
(get[1] (topk X))
```

No stateful programming model is needed.

---

## 8. `where` = legality only

`where` answers:

> Is this rewrite valid for this match?

Examples:

```javascript
where {
    broadcastable(Batch, WeightBatch);
    reassociable(A, B, C, @inner, @outer);
    rank(X) >= 2;
    K % 128 == 0;
}
```

It should be:

- pure
- side-effect free
- boolean-valued

Simple conditions can be built in:

```javascript
K == R
rank(X) == 3
```

Complex logic goes to host code.

---

## 9. `derive` = RHS metadata synthesis

`derive` constructs metadata required by newly created RHS operators.

Example:

```javascript
derive {
    @xw  = infer_dot(X, W, @outer);
    @xa  = infer_dot(X, A, @outer);
    @out = infer_dot(?XA, B, @inner);
}
```

Conceptually:

```text
where  -> bool
derive -> typed values
```

Typical derived values:

```text
DotAttr
ReduceAttr
TransposeAttr
TensorType
Shape
Dimension mapping
```

---

## 10. Operator attributes

Keep S-expressions compact:

```lisp
(dot[@d] X W)
```

`@d` is the authoritative operator descriptor.

For example:

```cpp
struct DotAttr {
    std::vector<int> lhs_batch;
    std::vector<int> rhs_batch;
    std::vector<int> lhs_contract;
    std::vector<int> rhs_contract;
    PrecisionConfig precision;
};
```

Detailed conditions belong in host predicates:

```javascript
where {
    reassociable(A, B, C, @inner, @outer);
}
```

not in the S-expression.

---

## 11. Region-bearing operators

Do not expose arbitrary lambdas or nested regions initially.

Represent common StableHLO region semantics using semantic descriptors.

For example:

```lisp
(reduce[@r] X)
```

with:

```cpp
struct ReduceAttr {
    Dimensions dims;
    ReductionKind kind;   // add, max, min, mul, ...
};
```

Then:

```javascript
where {
    additive(@r);
}
```

This keeps TensorLang focused on tensor graphs instead of becoming a general region language.

---

## 12. Host function registry

Complex semantics live outside the DSL:

```cpp
registry.predicate("reassociable", reassociable);
registry.predicate("broadcastable", broadcastable);

registry.function("infer_dot", inferDot);
registry.function("infer_reduce", inferReduce);
```

Execution model:

```text
DSL call
  ↓
registry lookup
  ↓
C++ / Rust implementation
```

This is where StableHLO-specific reasoning belongs.

---

## 13. Multi-pattern support

Allow multiple roots when needed:

```javascript
rule fuse_projection {
    match {
        ?Y1 = (dot[@d1] X W1)
        ?Y2 = (dot[@d2] X W2)
    }

    where {
        compatible(@d1, @d2);
    }

    rewrite {
        ...
    }
}
```

All patterns share one substitution environment.

Initially, require patterns to be connected through shared variables.

---

## 14. Variadic operands

For n-ary operators, optionally support expression sequences:

```javascript
(concat Xs...)
```

or:

```javascript
(concat A Xs... B)
```

This is useful for:

```text
concat
tuple
generic n-ary operators
```

but it can remain optional for the first implementation.

---

## 15. Example: LoRA

```javascript
rule lora {
    X: [Batch..., M, K]
    W: [WeightBatch..., K, N]
    A: [WeightBatch..., K, R]
    B: [WeightBatch..., R, N]

    (dot[@outer] X
      (add W
        (dot[@inner] A B)))
    =>
    (add
      (dot[@xw] X W)
      (dot[@out]
        (?XA = (dot[@xa] X A))
        B))

    where {
        broadcastable(Batch, WeightBatch);
        reassociable(X, A, B, @outer, @inner);
    }

    derive {
        @xw  = infer_dot(X, W, @outer);
        @xa  = infer_dot(X, A, @outer);
        @out = infer_dot(?XA, B, @inner);
    }
}
```

---

## 16. Internal compilation pipeline

```text
DSL source
   ↓
Parser / AST
   ↓
Structural matcher
   ↓
Shape unification
   ↓
TensorDesc resolution
   ↓
where evaluation
   ↓
derive evaluation
   ↓
RHS type / semantic validation
   ↓
RHS construction
   ↓
E-graph insertion
   ↓
Extraction
```

That ordering is important: invalid RHS operators should be rejected before insertion.

---

## 17. Scope boundary

I would explicitly define TensorLang as:

> A declarative rewrite DSL for single-graph tensor expressions, with optional tuples, symbolic shape constraints, semantic operator descriptors, and host-defined legality and metadata inference.

Initially out of scope:

```text
arbitrary control flow
arbitrary nested regions/lambdas
stateful algorithm descriptions
full symbolic shape arithmetic
general programming-language semantics
```

That boundary keeps the DSL compact while still being strong enough for LoRA, dot/reduce interchange, normalization-linear rewrites, projection rewrites, and many StableHLO-level graph transformations.

The design is now coherent enough that the next step should be to freeze the surface syntax and define the AST/types formally.