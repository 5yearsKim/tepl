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
  nested expressions. It also plans binding dependencies for the builtin-only
  prefix of `where`, stopping at the first condition containing a host call.
  Unconditional shape-check dependencies are tracked
  separately so expression metadata reads remain inside short-circuit control
  flow. Conditions and derivations retain core's source order.
- `common/project_plan.*` groups dialect operations and schemas by declaration
  source, and rules by their owning source module. It preserves raw relative
  module paths so every target can apply its own identifier rules.
- `rust/backend.*` assembles module contents for an existing crate from the
  validated naming plan.
- `rust/dialect_emitter.*` emits the combined operation enum and attribute enum
  from all checked dialect declarations, with arity, aliases, and schema checks.
- `rust/rule_emitter.*` recursively emits patterns, RHS expressions, shape
  restrictions, and per-rule host interfaces and callbacks.
- `rust/analysis_emitter.*` emits shape and dtype dispatch from checked operation
  definitions, including operation shape evaluators and attribute conversion.
- `rust/shape_expression_emitter.*` emits shape expressions through shared
  builtins, checked arithmetic, fallible indexing, and list comprehensions.
- `rust/expression_emitter.*` emits typed rule builtin/host expressions with
  shared checked arithmetic and short-circuit boolean evaluation.
- `rust/builtin_emitter.*` emits builtin calls for both rule and shape contexts,
  including borrowing, variadics, numeric adapters, and error propagation.
- `rust/names.*` constructs the complete validated Rust naming plan before
  emission: dialects, operations, schemas, fields, host methods, rules, source
  modules, and output indexes. It also owns string escaping and type representations.
- `rust/paths.h` constructs relative generated paths and absolute external crate
  paths. Emitters use qualified dialect references and explicit runtime imports.
- `rust/code_writer.h` handles indentation and blocks.
- `templates/rust/` holds reusable Rust templates, embedded by Bazel and copied
  into the generated module. Runtime algorithms are maintained separately from emitters.
  Bazel builds `tools/embed_templates.cc` and invokes it directly through
  `tools/embed_templates.bzl` to generate `rust/template_files.h` in its output
  directory. This build step needs no shell; the compiled TEPL executable contains
  the templates and does not need the embedding tool at runtime.
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

The emitted module exposes `{analysis, builtins, dialects, op_node, types, pattern, rules}`.
`analysis` exposes `TensorAnalysis`, `TensorAnalysisData`, `TensorBindingTable`,
`TensorInfo`, and pure `infer_shape`, `infer_dtype`, and `infer_tensor` functions.
`builtins::{common, shape, dtype}` contains the shared checked runtime helpers.
Common list/arithmetic functions live in `builtins::common`; shape semantics
live in `builtins::shape`. Errors use `builtins::{BuiltinError, BuiltinResult}`.
Both rule and shape emitters call these implementations. Builtin signatures and
section availability live in `src/core/builtins/catalog.*`.
Operation evaluators come from checked shape programs and dtype policies;
missing definitions return `Unknown`. Reusable analysis implementations and
policy helpers are copied from `templates/rust/src/analysis/`.
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

Module names use snake_case; local enum variants use PascalCase. Each identifier
has a semantic key and a Rust token: dialect `Type` generates `pub mod r#type;`
and the file `dialects/type.rs`. Keyword source filenames and directories use the
same policy. Filesystem paths and rewrite names never contain the `r#` escape.
The compiler validates names after normalization, so dialect, operation, or
schema `self_` fails with a TEPL source diagnostic for generated variant `Self`.
Names Rust cannot escape must be renamed. Collisions are checked in each actual
Rust declaration scope, including normalized parent directories and rule
file/directory conflicts. `mod.rs`, runtime union variants (`None`, `Literal`,
`Input`), and schema variant `None` are reserved in their respective scopes.
Generation reports the conflicting declarations before formatting or writing
output; it never adds numeric suffixes. Repeated imports of one dialect produce
one output file.

Generated code uses explicit runtime imports and qualified dialect paths.
Dialects named `Std`, `Egg`, `Op`, or `DType` therefore cannot shadow runtime types
or external crates. External references start with `::std::` and `::egg::`;
generated references remain relative so the output can be relocated.

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
unprefixed builtin calls lower to `BuiltinCall` and never create host methods.
Bound template `fn` calls retain their host binding; unknown unprefixed calls
are errors. See [builtin signatures and availability](../core/builtins/README.md).
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
- `constraints()`: shape/dtype restrictions indexed by capture, including inheritance.
- `match_checks(metadata)`: declarations and ordered pure condition evaluators;
  metadata is shared through `Arc`.
- `build_rewrite(functions)`: a checked rewrite using generated `TensorAnalysis`.
- `build_rewrite_with(metadata, inference, functions)`: explicit callbacks for a
  custom analysis or inference policy.

Rewrite names are qualified as `FILE::NAME`, and rule modules have no broad
re-exports. Host names are scoped to their source file; same-file overloads
remain unsupported. For rules without host calls, pass `()` as `functions`. Abstract rules are
already expanded in core; only concrete instances are emitted. Captures and
attribute variables use core IDs to keep repeated identities and avoid collisions
with user names. Rule integer arithmetic calls shared checked builtin helpers. Builtin errors
reject candidates through `Option` propagation, while shape evaluator errors
use `Result` and become invalid inference. Both conditions and derivations
read matched LHS metadata;
derivations may also read earlier derived descriptors.

Generated builders use `tensor_rewrite_checked_with_checks`. `MatchChecks`
combines the existing tensor declarations with ordered condition dependencies
and generated Rust evaluators. Shapes bind dimensions; conditions consume those
bindings. Each branch holds one `next_condition` cursor alongside tensor,
attribute, and shape bindings. At traversal start and after new bindings, the
matcher advances through ready conditions in source order. Missing bindings wait;
false or evaluation failure rejects the branch. Readiness never eagerly fetches
metadata or evaluates a skipped expression branch. A later condition waits for
an earlier condition even when its own bindings are ready.

Only the builtin/operator prefix before the first host-containing condition is
eligible. Nested host calls and calls in short-circuited branches still end that
prefix. Host conditions, subsequent conditions, and derivations execute only
during application, retaining their source order and host-call behavior.
Application rematches with the same plan, then rechecks shapes and the early
prefix against current metadata before executing the remaining callback and
validating the RHS. The same generated evaluators serve search and final checks;
there is no runtime expression interpreter. Search limits count only surviving
substitutions, and attribute witnesses are filtered before substitution deduplication.
Rewrite builders require an owned (`'static`) analysis type and callbacks.
Handwritten callers can use `matches_at_with_checks` or
`tensor_rewrite_checked_with_checks`; custom early evaluators must be pure and
stable during traversal. The tensor-only APIs remain wrappers with an empty
condition plan.
Rules with no early conditions emit `MatchChecks::tensors(constraints())` instead
of generating a condition-dispatch closure. Rule planning collects dependencies
once per expression and retains only host functions and early-condition bindings.

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
`templates/rust/README.md` for contextual RHS resolution and runtime requirements.

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
external crate imports use absolute paths such as `::egg::` and `::std::`.

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
