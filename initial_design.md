# TEPL design

TEPL describes declarative tensor graph rewrites for an e-graph. Graph syntax
captures structure; shapes constrain matches; operator descriptors carry
attributes; host functions establish legality and infer metadata. Tensor and
numerical semantics belong in the host implementation.

## Rules and graph expressions

A concrete rule contains optional declarations, one `LHS => RHS` rewrite,
then optional `where` and `derive` blocks, in that order. Declarations have no
semicolon; statements inside those blocks require one. For example,
[examples/rules/lora.tepl](examples/rules/lora.tepl):

```tepl
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
        broadcastable(Batch, WeightBatch);
        reassociable(X, A, B, @outer, @inner);
    }
    derive {
        @xw = infer_dot(X, W, @outer);
        @xa = infer_dot(X, A, @outer);
        @out = infer_lora_out(X, A, B, @outer, @inner);
    }
}
```

Graph expressions are names, numeric literals, and S-expression applications.
LHS patterns additionally allow `let` bindings: `(let Y = (dot[@d] X W))`
captures the matched operation's result e-class as `Y`; a root binding can omit
parentheses. Repeated tensor names refer to the same e-class. RHS expressions
are trees of operations, literals, and LHS capture references. They cannot
introduce bindings; `let` is rejected on the RHS, including nested occurrences.
See [examples/rules/binders.tepl](examples/rules/binders.tepl).

Numeric operands and roots accept integers (`1`) and decimals (`1.0`), with
an optional sign (`-0.5`). Decimals require digits on both sides of the point;
exponents and attached numeric suffixes are unsupported. Graph literals may carry
an explicit dtype (`1:i32`, `1.0:f32`, `-0.5:bf16`) and denote rank-zero tensors.
Bare literals remain available in the structural frontend; typed runtime
construction requires a concrete dtype, with no implicit default. Structural
matching preserves kind, spelling, and annotation, so `1:i32`, `1:f32`, and
`1.0:f32` are distinct. Constraint numbers remain host values. See
[examples/rules/basic.tepl](examples/rules/basic.tepl).

Tuple syntax is `(tuple X Y)` and `(get[0] T)`; projection has one operand and
a nonnegative integer index. These forms are parsed but have no Rust tensor
runtime implementation yet.

## Shapes and host semantics

Declarations are optional match constraints, rather than a complete type system:

| Syntax | Meaning |
| --- | --- |
| `X: [M, K]` | Tensor with named dimensions and unrestricted dtype; repeated names must agree |
| `X: bf16[M, K]` | Same shape constraint, restricted to bf16 |
| `X: f32[...]` | Any shape, restricted to f32 |
| `X: []` | Rank-zero tensor |
| `_` | One arbitrary dimension |
| `...` / `Batch...` | Anonymous / named sequence of zero or more dimensions |
| `S: scalar` / `S: []` | Rank-zero tensor with unrestricted dtype |
| `S: f32[]` | Rank-zero f32 tensor |
| `@d` | Captured or derived operator descriptor |

Supported dtype names are `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`,
`u32`, `u64`, `f16`, `bf16`, `f32`, and `f64`. Dtype names remain ordinary
identifiers outside annotation positions. An annotation constrains matching and
never inserts a cast; omission has no default. Unknown names, duplicate local
tensor declarations, and declarations that do not refer to LHS captures are
semantic errors. Inherited declarations validate annotations immediately and
defer capture checks until expansion. Expansion must combine restrictions:
unconstrained dtype may be narrowed, while conflicting explicit dtypes are errors.

Integer graph literals must fit their annotated dtype. Boolean numeric literals
use `0:bool` and `1:bool`; negative or fractional boolean spellings are rejected.
Floating literals preserve decimal spelling without machine conversion; the
host defines format rounding and representability. Complex, FP8, quantized
types, dtype variables, and implicit promotion are outside v1. Supporting a
dtype in the IR does not require every operation to support it.

A shape has at most one sequence, anywhere in its dimension list. Arithmetic
such as `K % 128 == 0` belongs in `where`.

`where` contains boolean legality checks, intended to be pure. `derive` assigns
attributes to RHS descriptors. Both support host calls, names, descriptor
references, integers, decimals, booleans, and conventional arithmetic,
comparison, and
logical expressions. Host checks must establish shape and numerical validity,
including any permission to reassociate floating-point operations.

The Rust host interface uses `TensorInfo { shape, dtype }` for tensor metadata.
Predicates return `Option<bool>`; descriptor derivations such as `infer_dot`
return `Option<OpAttrs>`. A descriptor contains only the attributes needed to
construct an operation. `None` rejects a match.

`where` and `derive` read LHS tensor captures, captured descriptors, and
shape variables. Descriptor derivations are evaluated in source order; they do
not read newly constructed RHS values. LoRA derives its final dot descriptor
through `infer_lora_out(X, A, B, @outer, @inner)`, allowing the host to choose
attributes using only existing inputs. There is no runtime dependency scheduler
for descriptor derivation. Output validation is separate: `OutputInference`
checks the resolved RHS from its leaves upward without inserting nodes.

The host's e-class analysis and pre-insertion `OutputInference` should share
operation semantics. Unsupported operand dtypes or invalid descriptor contents
return `None`; there is no implicit promotion. An operation with a selectable
result dtype must encode that choice in its attributes. Storage dtype,
accumulation dtype, and numerical legality are separate contracts. Derived descriptors contain no
output metadata. Metadata supplied for a matched e-class must hold for all its
alternatives. The runtime finishes host checks and descriptor derivations, then
validates the whole RHS tree before adding any nodes. A rejected match leaves no
partial RHS behind.

## Dialects and imports

Dialects declare named operands, result types, aliases, and inline or shared
attribute schemas. The current validator supports tensor operands/results and
`index`, `string`, and their list forms as attribute types. A final variadic
operand permits additional inputs. For example:

```tepl
dialect TensorLang {
    attrs CollectiveReduce { kind: string; }
    op add(lhs: tensor, rhs: tensor) -> tensor;
    op multiply(lhs: tensor, rhs: tensor) -> tensor { alias: mul; }
    op all_reduce(input: tensor) -> tensor { attrs: CollectiveReduce; }
    op concatenate(first: tensor, second: tensor, rest: tensor...) -> tensor {
        attrs { axis: index; }
    }
}
```

[examples/dialects/tensor.tepl](examples/dialects/tensor.tepl) defines the
reference tensor dialect, including `dot` as an alias of `dot_general`.
`from "path" import TensorLang as t;` allows qualified calls such as
`(t.add X Y)`. `use t::{add, dot};` opens selected operations; `use t;` opens
all. The legacy `import "path";` opens all imported dialect operations.
Imports resolve relative to the importing file and may be nested; missing,
invalid, cyclic, or ambiguous imports produce diagnostics.

## Reusable rules

Abstract rules parameterize operations (`op`) and host functions (`fn`) with
signatures. Concrete instances bind parameters by name:

```tepl
abstract rule commute(F: op<(tensor, tensor) -> tensor>) {
    (F X Y) => (F Y X)
}
rule commute_add extends commute(F = t.add);
rule commute_small_vectors extends commute(F = t.add) {
    X: [N]
    Y: [N]
    where { N <= 1024; }
}
```

Templates can be selected with `from "abstract.tepl" import {commute};`.
Instances may add declarations and `where` restrictions. Their
expansion preserves the inherited graph and combines conditions; instances
cannot provide a replacement graph or `derive` block. Parsing and importing
preserve the source AST; the core analyzer expands concrete instances and
validates bindings and signatures. See
[abstract.tepl](examples/rules/abstract.tepl) and [inherited.tepl](examples/rules/inherited.tepl).

## Implementation and execution

The C++ frontend uses an ANTLR4 grammar without embedded actions and builds an
owning AST with source spans. The `parse` command checks syntax and resolves
imports while preserving source rules and annotations. The `check` command
performs semantic analysis through `src/core/`, validates operation names,
arity, descriptors, and types, and expands concrete instances into the core IR.
`generate FILE --target rust --out DIR` emits a standalone crate from checked
dialects and concrete rules through the shared `src/codegen/` backend interface.

[labs/rust-egg](labs/rust-egg/README.md) implements the tensor IR, matching,
binders, host callbacks, and e-graph rewrite application. Dialect and rule
modules remain manually maintained reference examples; generated crates use the
shared templates under `runtime/rust/` and their own dialect definitions.
Its execution model is:

```text
structural match -> metadata lookup and shape/dtype checks -> where checks
-> derive descriptors from LHS inputs
-> validate RHS structure and infer every RHS output
-> require root shape/dtype compatibility -> insert and union with matched root
-> saturation and cost-based extraction
```

RHS operation, attribute, and arity checks occur before insertion. LoRA applies
its declared shape constraints and explicit `where` predicates, then derives
three descriptors from LHS inputs. The checked rewrite path verifies every RHS operation through `OutputInference`
and requires root shape/dtype equality with matched-root metadata before any
insertion or union. Unknown metadata rejects the match. Numerical equivalence
still requires host legality predicates. The structural `tensor_rewrite` API
retains an explicit caller obligation to prove the entire replacement valid.
The LoRA saturation example uses the checked path.

`TensorBindings` registers immutable symbol and named-constant types; conflicting
registrations are rejected. An identity must have one tensor type per graph.
E-class metadata must describe every alternative, including dtype; hosts must
return `None` when this cannot be established. Dtype is part of literal node
identity, so differently typed literals cannot be hash-consed into one node.

The C++ core analyzer resolves operation and rule-local symbols, expands
inherited rules, checks tensor/host types, and validates descriptor references.
`check FILE` prints its checked IR; `generate` emits Rust dialect/rule code.
C++ and Python backends remain future work. Abstract bodies are checked on
instantiation, and the initial analyzer explicitly rejects tuple/projection semantics.
Multiple-root patterns, variadic graph captures such as `Xs...`, arbitrary
regions/control flow, and full symbolic shape algebra are outside current
scope. Example files demonstrate syntax; their tensor equivalence depends on
host semantics and is exercised separately by the Rust lab.
