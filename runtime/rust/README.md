# Rust runtime templates

These files are the maintained runtime portion of each generated Rust module.
Bazel embeds them into the compiler. `src/pattern/` is copied into the output's
`pattern/`; it contains matching, attribute witnesses, shape checks,
metadata access, and checked rewrite application. It has no dependency on a
particular dialect.

`src/op_node.rs` supplies the shared node implementation and typed `DialectOp`
constructor. Generation inserts project-specific `Op` and `OpAttrs` sum types
at the marker below the imports, before the node implementation, and writes
`op_node.rs`. `src/types.rs` is copied as `types.rs`.
`src/analysis/shape_builtins.rs` is copied as `analysis/shape_builtins.rs`.
The generator exposes it through an inline `analysis` module. It implements
every builtin in `examples/shape_guide.md`, including those not yet used by the
sample evaluators, with no dependency on dialects, host semantics, or egg:

| Helpers | Runtime behavior |
| --- | --- |
| `len`, `range` | List length and eager zero-based axes; range bounds must be nonnegative and fit `usize` |
| `concat` | Concatenate two or more lists, preserving nested list boundaries |
| `gather`, `exclude` | Select checked positions (duplicates allowed) or remove matching values |
| `slice`, `replace` | Copy a checked slice or replace one checked entry; bounds never clamp |
| `sum`, `product` | Checked integer reductions; empty identities are zero and one |
| `all`, `any` | Reduce already evaluated Boolean lists; empty identities are true and false |
| `contains`, `is_disjoint` | Membership and absence of shared values |
| `is_valid_axis_list` | Unique, nonnegative axes within the supplied rank |
| `broadcast_shape` | Right-aligned broadcasting; broadcasting zero with one produces zero |
| `min`, `max` | Integer extrema |
| `floor_div`, `ceil_div` | Rounded integer division, including negative numerators; divisors must be positive |

The additional `ensure` helper returns assertion errors in both debug and
release builds. `product` returns zero when any entry is zero, even if a
preceding prefix would overflow. Incompatible broadcasting, arithmetic overflow,
and invalid indices produce errors. `range` and `concat` also report capacity
failures rather than panicking on unrepresentable list sizes.

List helpers are generic over their element type, including nested lists.
Rust calls pass slices; a TEPL call `concat(a, b, c)` becomes
`concat(&[&a, &b, &c])`, and `contains(xs, x)` becomes `contains(&xs, &x)`.
`len` returns `usize`; `range` produces `u64` axes. Index helpers accept integer
types with checked conversion to `usize`, rejecting negative indices.
`sum`, `product`, `min`, `max`, `floor_div`, and `ceil_div` support primitive
signed and unsigned integers through `ShapeInteger`. The evaluator selects the
integer representation; a future generator must explicitly convert mixed
signed/unsigned operands and check that final dimensions fit `u64`.

Fallible helpers return `ShapeResult<T>` (`Result<T, ShapeError>`). Operation
evaluators compose them with `?`; their dispatcher converts errors to invalid
inference. Missing operation definitions are handled separately as unknown
inference.
The lab's handwritten operation evaluators demonstrate this interface; the
compiler does not yet generate evaluators from TEPL shape blocks.

Generation emits files directly into the selected module directory of an
existing crate. Internal imports use `super`, so the enclosing module can have
any name and location. The lab's `src/ir/` uses this output, regenerated
by `tools/regenerate_lab.sh`; its host semantics are maintained in `src/host/`.
Changes to these templates reach the lab through regeneration.

Hosts supply `TensorMetadata`, `OutputInference`, and the generated per-rule
`Functions` traits. Shapes and index values use `u64` consistently with core's
64-bit Index contract. Rust slice positions and table IDs use `usize`.

Literal patterns carry `Option<DType>`: `None` accepts any dtype while preserving
exact spelling. RHS literal requests also retain `None` until
`OutputInference::infer_literal` resolves them. Its default uses the matched
root's dtype as context; a host can override it or return `None`. Explicit
annotations cannot be overridden. Output inference must accept the resulting
literal as rank zero with that dtype. All RHS validation and output inference
finish before any node is inserted.

Host functions are fallible: `None` rejects a match. Generated integer
arithmetic uses checked operations; overflow and division/remainder by zero
reject matches in debug and release. Nonfinite host floating values/results
reject matches. Boolean operators short-circuit. Derived attributes are checked
against their inferred schema and evaluated in source order.

The runtime preserves distinct attribute witnesses even when they share the
same tensor substitution. Tensor metadata must describe all alternatives in an
e-class; missing or incompatible metadata rejects the match. Numerical
rewrite equivalence is established by the host's legality functions.

Run compiler-to-runtime integration tests with `./tools/test_codegen.sh` from
the repository root. Rust runtime files and integration fixtures are formatted
with `rustfmt --edition 2024`.

Runtime input leaves use `OpNode::input(name)` and `Op::Input`, independently of
any dialect. `types.rs` also defines `Precision`, `DotAlgorithm`, `ReplicaGroups`,
`Region`, and `Elements` for generated attributes. Region and mesh payloads are
canonical opaque bytes supplied by the host. Elements preserve a canonical type
encoding, shape, and payload bytes; algorithm type encodings can represent formats
outside the built-in `DType` set. All metadata participates in equality and
hashing. The host validates payloads and operation-specific attribute semantics.
