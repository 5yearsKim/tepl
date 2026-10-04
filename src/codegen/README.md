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
- `common/project_plan.*` groups dialect operations and schemas by declaration
  source, and rules by their owning source module. It preserves raw relative
  module paths so every target can apply its own identifier rules.
- `rust/backend.*` assembles module contents for an existing crate and computes
  relative import paths for nested rule modules.
- `rust/dialect_emitter.*` emits the combined operation enum and attribute enum
  from all checked dialect declarations, with arity, aliases, and schema checks.
- `rust/rule_emitter.*` recursively emits patterns, RHS expressions, shape
  restrictions, and per-rule host interfaces and callbacks.
- `rust/analysis_emitter.*` emits shape and dtype dispatch from checked operation
  definitions, including operation shape evaluators and attribute conversion.
- `rust/shape_expression_emitter.*` emits shape expressions through shared
  builtins, checked arithmetic, fallible indexing, and list comprehensions.
- `rust/expression_emitter.*` emits typed, fallible host expressions with checked
  arithmetic and short-circuit boolean evaluation.
- `rust/names.*` owns Rust names, string escaping, and type representations.
- `rust/code_writer.h` handles indentation and blocks.
- `runtime/rust/` holds fixed runtime code, embedded by Bazel and copied into the
  generated module. Runtime algorithms are maintained separately from emitters.
- `write.*` synchronizes generated files, maintains their ownership manifest,
  formats output by default, and checks for drift without changing output.

To add another language, create `cpp/` or `python/` with the same backend/emitter
roles, inherit the public `Generator`, and register it in `createGenerator`.
Reuse checked core and `common/` project and dependency planning. Target syntax, ownership,
names, and runtime integration stay in that backend. `Options` and
`GenerationResult` remain the shared entry and exit types. `tests/codegen_test.cc`
contains an independent example backend using this interface.

The first runtime target is Rust with egg. An additional Rust runtime adapter
can be introduced inside the Rust backend without changing core or the public
language-backend interface.

## Generated Rust API

The emitted module exposes `{analysis, dialects, op_node, types, pattern, rules}`.
`analysis` exposes `TensorAnalysis`, `TensorAnalysisData`, `TensorBindingTable`,
`TensorInfo`, and pure `infer_shape`, `infer_dtype`, and `infer_tensor` functions.
`analysis::shape_builtins` exposes the copied checked shape primitives.
Operation evaluators come from checked shape programs and dtype policies;
missing definitions return `Unknown`. Reusable analysis implementations and
policy helpers are copied from `runtime/rust/src/analysis/`.
Its enclosing name and location are chosen by the consuming application; the
examples below use `generated`.
Each dialect has its own module and short `Op` and `OpAttrs` enums, such as
`generated::dialects::tensor_lang::Op::Add` and `generated::dialects::scalar::Op::Add`.
`generated::op_node` defines the combined `Op`, `OpAttrs`, and `OpNode` types used by egg.

```rust
use crate::generated::{OpNode, dialects::scalar};
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
that schema. Attribute fields and host trait methods preserve their TEPL names.
The `$` host-call sigil is syntax only: `$infer_dot(...)` generates the trait
method `infer_dot` and calls `functions.infer_dot(...)`. Checking resolves host
calls (including bound `fn` parameters) to `HostCall` before code generation;
unknown unprefixed calls never implicitly create host methods.
Rust keywords use raw identifiers, such as `r#type` and `r#match`. Names Rust
cannot escape (`self`, `Self`, `super`, and `crate`) produce a generation
diagnostic; rename them in TEPL. No prefixes or suffixes are added.
Index fields use `u64`, string fields use `String`, and lists use `Vec`.
Schema default annotations are retained as field comments; Rust callers supply
all fields when constructing an attribute value, using an empty vector for an
empty-list default.

Each concrete rule becomes `generated::rules::FILE::rule_NAME` with:

- `Functions`: only host functions actually referenced by that rule.
- `pattern()` and `expression()`: structural match and replacement trees.
- `build_rewrite(functions)`: a checked rewrite using generated `TensorAnalysis`.
- `build_rewrite_with(metadata, inference, functions)`: explicit callbacks for a
  custom analysis or inference policy.

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
the common `Literal` node, separately from any user-declared `literal` op.
An omitted dtype remains unconstrained during matching and generation. See
`runtime/rust/README.md` for contextual RHS resolution and runtime requirements.

The generated module requires Rust edition 2024 and egg 0.11. Generation runs
`rustfmt` on staged output by default, before writing or comparing it. This
requires `rustfmt` on `PATH`; `--no-format` disables formatting. API callers can
set `WriteOptions::format = false` to opt out. `--format` explicitly enables it.
Rustfmt formats layout but does not eliminate redundant expression parentheses;
generated rule modules still allow `unused_parens`.

## Operation inference

Shape expression values use signed `i128` arithmetic and homogeneous nested
lists. Dimensions and index attributes enter as `u64`; signed attributes enter
as `i64`. Generated evaluators widen inputs, call checked helpers, and validate
all yielded dimensions against `u64`. Parameters follow the operation's fixed
and variadic operand signature. Assertions and local bindings execute in source
order; logical operators and conditional branches short-circuit. Comprehensions
propagate element failures without panicking.

Dtype dispatch uses the checked optional policy: common policies require equal
operand dtypes, numeric policies exclude bool, float policies accept only floats,
and concrete policies fix the result dtype. Common policies with no actual
operands return `Unknown`. There is no name-based inference or implicit promotion.

`infer_tensor` combines known shape and dtype; any invalid component takes
priority, and incomplete metadata is unknown. Runtime input and literal nodes
have shared policies. User operations with absent shape or dtype declarations
remain unknown, including typed-payload constants. Attribute-dependent dtype
behavior needs an explicit application policy or a future language extension.
The LoRA demo illustrates a host policy through `build_rewrite_with`.

## Output ownership and regeneration

Generation emits only module contents into `--out`. It does not create
`Cargo.toml`, `src/lib.rs`, or an enclosing `src/` directory. Point `--out` at
any module directory in your application, declare that module in its parent,
and add egg 0.11 to your application's dependencies. There is no module-name
option: relative internal imports make output independent of the enclosing
name and depth. Nested rules use the appropriate number of `super::` segments;
imports of external crates such as egg and std retain their normal paths.

```sh
bazel-bin/tepl generate examples --out my_app/src/generated
./tools/regenerate_lab.sh --check
```

`.tepl-generated-files` records the relative paths owned by generation. A later
generation removes obsolete files listed in that manifest and preserves unlisted
files. Keep application host implementations outside the generated paths. Invalid
manifest paths and symlinked generated paths are rejected before writing.

`--check` compares file contents and the manifest, reporting missing, changed,
or obsolete output without modifying it. It returns 0 when current and 1 on
drift; usage, formatting, and output errors return 2. Use the same
formatting options for generation and checking. The lab checks in its generated
IR and manifest; `tools/regenerate_lab.sh` supplies those options consistently.
