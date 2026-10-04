# C++ code generation templates

These handwritten C++20 runtime headers are embedded in TEPL and copied into
generated output. Declaration-dependent code belongs in `src/codegen/cpp/`.
The directories mirror the Rust runtime: `analysis`, `builtins`, and `pattern`,
alongside `types.h` and `op_node.h`. The compiler replaces namespace placeholders
and inserts project-specific operation and attribute unions into `op_node.h`.

Consumers need egg-c headers and GCC or Clang with signed `__int128` support.
The tested egg-c revision is `0c28bd5050b85ed10e915b5348c27f760ed31ae5`.
TEPL's compiler itself does not depend on egg-c. Generated code uses relative
includes and a configurable namespace, defaulting to `tepl_generated`.
Every header identifies its namespace so GCC's `#pragma once` heuristic keeps
multiple generated libraries distinct even when wrapper contents and file
timestamps would otherwise match.

## Generated API

`generated.h` includes the full generated library. The public namespaces are
`analysis`, `builtins::{common,shape,dtype}`, `dialects`, `pattern`, and `rules`.
Individual headers can be included directly. Non-template definitions are
inline, allowing multiple translation units to include the library.

Each dialect defines scoped `Op` values, named attribute structs, and an
`OpAttrs` variant wrapper. The root `Op` and `OpAttrs` combine dialect values
without erasing their identities. For example:

```cpp
auto node = generated::OpNode::make(
    generated::dialects::tensor_lang::Op::Add, {}, {x, y});
```

`make` is the typed counterpart of Rust's `OpNode::new` (`new` is a C++ keyword).
Its attribute parameter uses that operation's dialect type. `from_parts` accepts
combined values dynamically. Both check arity and operation-specific schemas;
invalid construction throws `NodeError`. Runtime leaves use `OpNode::input`
and `OpNode::literal`. Opaque region, elements, replica-group, and algorithm
payloads retain structural equality and hashing; hosts validate their semantics.

Each rule namespace exposes `pattern`, `expression`, `constraints`,
`match_checks`, and rewrite builders. `Functions` is a C++ concept checking only
the host methods used by that rule. Methods are callable on a const host object
and return `std::optional<T>` for fallible results. Empty hosts use the default
`build_rewrite()` argument. Builders own hosts and metadata callbacks; a move-only
host can be passed by move. Shared callback state must remain valid and stable
through each read-only traversal.

| Host value | Argument | Successful result |
| --- | --- | --- |
| Tensor | `const TensorInfo&` | `TensorInfo` |
| Index | `std::uint64_t` | `std::uint64_t` |
| IndexList | `std::span<const std::uint64_t>` | `std::vector<std::uint64_t>` |
| Bool | `bool` | `bool` |
| I64 | `std::int64_t` | `std::int64_t` |
| F64 | `double` | `double` |
| Descriptor | `const OpAttrs&` | `OpAttrs` |

`build_rewrite()` uses generated `TensorAnalysis`. A custom analysis uses
`build_rewrite_with<MyAnalysis>(metadata, inference, functions)`, where metadata
is callable with `(const EGraph&, Id)` and returns `optional<TensorInfo>`.
Inference may be callable with `(Op, span<const TensorInfo>, const OpAttrs&)`,
or supply an `infer_output` method with that signature. An optional
`infer_literal(value, requested_dtype, expected)` method overrides contextual
literal resolution. Explicit dtype annotations always apply.

## Runtime contracts

`TensorAnalysisData` joins known observations, unknown alternatives, and invalid
or conflicting evidence. Metadata is exposed only when every alternative is
known and agrees. Conflicting evidence returns an egg-c changed/unchanged join;
it does not use egg-c's union-rejecting conflict result. Register inputs through
`TensorBindingTable::register_symbol` before graph construction; conflicting
registrations throw. Rebuild a fresh graph when input metadata changes.

Matching streams structural witnesses through branch-local tensor, attribute,
dimension, sequence, and ordered-condition bindings. The builtin-only condition
prefix ends at the first condition containing any host call. Readiness checks
binding presence, preserving short-circuiting and lazy metadata reads. Host
conditions and derivations run during application. C++ expression emission
sequences host arguments and nonlogical operands explicitly from left to right.

One surviving tensor substitution reaches egg-c's custom-search sink regardless
of its attribute witnesses. Rejected branches never consume that sink's match
budget. The matcher honors cooperative cancellation and sink rejection.

egg-c batches applications and prohibits dirty node queries. TEPL therefore
rebuilds a dirty graph at the start of each checked application. Like Rust, it
rematches each e-class once when its application batch starts and groups all
attribute witnesses by the searched tensor substitutions. Later applications
reuse these structural witnesses, rechecking constraints, ordered conditions,
host callbacks, and output inference against current metadata. New witnesses
are considered by the next search, which owns a fresh batch. It prepares and
infers all accepted witnesses for a substitution before inserting any of their
RHS nodes. Multiple accepted
witnesses are unioned with the same rewrite attribution; egg-c performs the final
union. This may rebuild more frequently than the ordinary egg-c pattern runner.
Its statistics count one application per tensor substitution and report the final
union through the runner; earlier witness unions and application-time repairs
are not all represented in its per-rule union/rebuild counters. Saturation still
uses graph revisions and detects these changes.

RHS preparation resolves every tensor and descriptor and validates the entire
structure before output inference. Inference must preserve the root's shape and
dtype. Expected builtin failures, unavailable metadata, bad descriptors, and
invalid RHS nodes reject the candidate before insertion. This is validation
before mutation, not rollback for arbitrary host exceptions or allocation failure
during graph insertion. Host exceptions other than expected TEPL failures propagate.

Checked builtin helpers return `BuiltinResult<T>`; internal `take` propagates
its `BuiltinError` to generated rejection/inference boundaries. Indices use
unsigned 64-bit integers, signed rule integers use 64 bits, and shape arithmetic
uses checked signed 128-bit integers. Overflow, division/remainder by zero,
invalid indexing, and nonfinite floating values fail in every build mode.
MSVC requires a future compatible 128-bit implementation. List builtins support
nested lists; shape-only `all` and `any` retain the existing language availability.

## Validation

Run `./tools/test_codegen_cpp.sh` to generate, compile, and execute fixtures in
temporary applications. By default it checks out the pinned egg-c revision.
Set `EGGC_SOURCE_DIR` to test another local checkout and `CXX` to choose GCC or
Clang. The suite exercises debug and optimized builds, plus UBSan for arithmetic
and runtime failures, namespace relocation, standalone headers, generated-name
collisions, grouped rematching, multiple translation units, typed
attribute rejection, and the example LoRA rewrite through saturation, extraction,
and numerical comparison. It does not regenerate or require a handwritten lab.
To rerun selected fixtures, pass the compiler binary followed by fixture names,
for example `./tools/test_codegen_cpp.sh bazel-bin/tepl hygiene runtime`.
