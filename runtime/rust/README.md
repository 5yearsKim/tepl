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
The repository directory is called `runtime/` because it stores boilerplate;
the generated public submodule is `pattern`.

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
