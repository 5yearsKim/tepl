# Code generation

`generate.h` is the public interface for all language backends. Each backend
inherits `Generator` and implements `target()` and `generate(program, options)`.
Input must be a successfully checked `core::Program`. A rule alone is insufficient
because its IDs refer to the program's operation, type, schema, and host tables.

A `GenerationResult` contains relative output paths, file contents, and source
aware diagnostics. Backends perform no filesystem writes. The CLI writes files
only after successful generation. Rust is implemented; selecting C++ or Python
reports an explicit diagnostic and emits no files.

The layout separates shared semantics from target syntax:

- `common/rule_plan.*` computes host and tensor-metadata dependencies, including
  nested expressions. Unconditional shape-check dependencies are tracked
  separately so expression metadata reads remain inside short-circuit control
  flow. Conditions and derivations retain core's source order.
- `rust/backend.*` assembles a standalone Cargo crate.
- `rust/dialect_emitter.*` emits the combined operation enum and attribute enum
  from all checked dialect declarations, with arity, aliases, and schema checks.
- `rust/rule_emitter.*` recursively emits patterns, RHS expressions, shape
  restrictions, and per-rule host interfaces and callbacks.
- `rust/expression_emitter.*` emits typed, fallible host expressions with checked
  arithmetic and short-circuit boolean evaluation.
- `rust/names.*` owns Rust names, string escaping, and type representations.
- `rust/code_writer.h` handles indentation and blocks.
- `runtime/rust/` holds fixed runtime code, embedded by Bazel and copied into the
  generated crate. Runtime algorithms are maintained separately from emitters.

To add another language, create `cpp/` or `python/` with the same backend/emitter
roles, inherit the public `Generator`, and register it in `createGenerator`.
Reuse checked core and `common/` dependency planning. Target syntax, ownership,
names, and runtime integration stay in that backend. `Options` and
`GenerationResult` remain the shared entry and exit types. `tests/codegen_test.cc`
contains an independent example backend using this interface.

The first runtime target is Rust with egg. An additional Rust runtime adapter
can be introduced inside the Rust backend without changing core or the public
language-backend interface.

## Generated Rust API

The emitted crate exposes `ir::{dialects, op_node, types, pattern, rules}`.
Each dialect has its own module and short `Op` and `OpAttrs` enums, such as
`ir::dialects::tensor_lang::Op::Add` and `ir::dialects::scalar::Op::Add`.
`ir::op_node` defines the combined `Op`, `OpAttrs`, and `OpNode` types used by egg.

```rust
use generated_crate::ir::{OpNode, dialects::scalar};
let node = OpNode::new(scalar::Op::Add, scalar::OpAttrs::None, vec![x, y])?;
```

`DialectOp::Attrs` keeps an operation and its dialect's attributes paired at
compile time. Arity and operation-specific schemas are validated at runtime.
`OpNode::from_parts(op, children, attrs)` supports dynamic combined values and
validates their pairing. Local `Op::from_name` accepts local names and aliases;
combined `Op::from_name` requires a qualified name such as `Scalar.add`.

Module names use snake_case; local enum variants use PascalCase. The compiler
diagnoses normalized name collisions, reserved names, and rule file/directory
collisions rather than adding numeric suffixes. Rename the conflicting source
file or declaration. Repeated imports of one dialect produce one output file.

Project input scans `dialects/**/*.tepl` and `rules/**/*.tepl` in sorted order,
resolves imports per file, and deduplicates declarations by source and name.
It preserves each file's imports and host-function scope. `Options.rules_root`
controls relative rule paths for API callers. Nested files become nested Rust
modules. A single input file produces a rule module named after its file stem.
Inherited rules belong to their instance file, independently of diagnostic
origins that can point into a template file.

Attribute schemas become `OpAttrs` variants, shared by every operation using
that schema. Attribute fields and host trait methods use a `tepl_` prefix,
including ordinary names, to avoid Rust keywords and escaping collisions.
Index fields use `u64`, string fields use `String`, and lists use `Vec`.
Schema default annotations are retained as field comments; Rust callers supply
all fields when constructing an attribute value, using an empty vector for an
empty-list default.

Each concrete rule becomes `ir::rules::FILE::rule_NAME` with:

- `Functions`: only host functions actually referenced by that rule.
- `pattern()` and `expression()`: structural match and replacement trees.
- `build_rewrite(metadata, inference, functions)`: a checked egg rewrite.

Rewrite names are qualified as `FILE::NAME`, and rule modules have no broad
re-exports. Host names are scoped to their source file; same-file overloads
remain unsupported. For rules without host calls, pass `()` as `functions`. Abstract rules are
already expanded in core; only concrete instances are emitted. Captures and
attribute variables use core IDs to keep repeated identities and avoid collisions
with user names. Both conditions and derivations read matched LHS metadata;
derivations may also read earlier derived descriptors.

Host argument/result types are:

| Core type | Rust argument | Successful result |
| --- | --- | --- |
| Tensor | `&TensorInfo` | `TensorInfo` |
| Index | `u64` | `u64` |
| IndexList | `&[u64]` | `Vec<u64>` |
| Bool | `bool` | `bool` |
| I64 | `i64` | `i64` |
| F64 | `f64` | `f64` |
| Descriptor | `&OpAttrs` | `OpAttrs` |

Fallible host functions wrap results in `Option`. Inferred descriptor schemas
are checked at runtime, including intermediate results. Bare graph literals use
the common `TeplLiteral` node, separately from any user-declared `literal` op.
An omitted dtype remains unconstrained during matching and generation. See
`runtime/rust/README.md` for contextual RHS resolution and runtime requirements.

The generated package uses Rust edition 2024 and egg 0.11. Cargo can format the
output with `cargo fmt --manifest-path PATH/Cargo.toml`. Keep host implementations
in separate application files so regeneration does not replace them.
