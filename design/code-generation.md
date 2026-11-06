# Code generation and execution

`codegen::generate(ExecutableProgram, Options)` returns source files and
diagnostics. It performs no source loading, semantic rechecking, attribute
solving, formatting, or output writes. The CLI compiles source or validates and
loads JSON before calling it.

## Ownership

| Layer | Responsibility |
| --- | --- |
| `src/codegen/common/project_plan.*` | Source membership and rule/graph module paths |
| `src/codegen/{rust,cpp}/names.*`, `paths.*` | Target identifiers, paths, and collision validation |
| Backend emitters | Program-specific types, metadata functions, rules, and graph builders |
| `src/codegen/write.*` | Formatting, drift checks, and manifest-owned output synchronization |
| `runtime/{rust,cpp}/` | Values, builtins, axes, analysis, matching, replacement preparation, runners |

Explicit module roots override paths inferred beneath `rules/` and `graphs/`.
Source and JSON generate identical files when executable data and options agree.
Name/path errors clear generated output before synchronization. The writer
formats staged files before writing and removes only obsolete manifest-owned
files. `--check` reports drift without changing the destination.

Runtime source is a separate application dependency. Generated output contains
no copied runtime templates, dynamic semantic libraries, JSON interpreter, or
runtime solver.

## Emitted API

| Purpose | Rust | C++ |
| --- | --- | --- |
| Complete generated interface | `mod.rs` | `generated.h`, `program.h` |
| Operation contracts | `metadata.rs` | `metadata.h` |
| Typed dialect operations and schemas | `dialects/` | `dialects/` |
| Rule definitions and aggregate builders | `rules/` | `rules/` |
| Concrete graph constructors | `graphs/` | `graphs/` |
| Generated/runtime boundary | `Program` trait | `Program` concept |
| Backend dependency | `tepl-runtime`, egg, edition 2024 | C++20 runtime, egg-c, signed `__int128` |

Shared schemas are emitted once per dialect. Combined operation/attribute types
retain dialect identities. Node construction checks arity and attribute schemas.
Each rule exposes `definition().build()`; `rules::all()` builds the complete set.
Graph builders register input metadata before inserting input nodes and return
the graph, root, and named bindings.

Expression emission is exhaustive over completed executable variants. Attribute
synthesis has already resolved `auto`, supplied values, and defaults. Emitters
translate those completed computations into ordinary functions; descriptor
helpers return concrete schema types. Logical axes and sets have runtime type
checks separate from numeric positions and opaque payloads.

Rust modules use relocatable relative paths. C++ uses relative includes and a
configurable `--cpp-namespace`, with inline/template definitions for multiple
translation units. Keywords receive trailing underscores; resulting collisions
are diagnosed. Each C++ header identifies its namespace and path so GCC cannot
coalesce different byte-identical `#pragma once` wrappers with relative includes.
Rust output uses `rustfmt`; C++ output uses `clang-format`.

## Shared execution contract

Shape and dtype are independent facts: unknown means no evidence, known refines
unknown, and differing known facts create a conflict. Explicit shape programs
precede axis contracts. Input metadata must be registered before node insertion;
conflicting registrations fail without changing existing facts.

Matching preserves concrete node/attribute witnesses and tensor captures.
Applications recover witnesses using current canonical tensor bindings. Source
conditions execute first, descriptor derivations have a separate local scope,
and dependent conditions execute afterward with the original `where` locals.
Arguments execute once in source order; Boolean operators and conditionals
short-circuit. Checked arithmetic, conversions, finite floats, and resource
budgets apply in optimized builds as well as debug builds.

All conditions, derivations, and complete RHS inference finish before insertion.
Known output shape/dtype disagreement rejects a candidate. Accepted witnesses
must remain compatible as earlier replacements refine metadata. This is
validation before mutation, not rollback for allocation failures during insertion.

| Outcome | Runtime behavior |
| --- | --- |
| False condition | Skip candidate |
| Unavailable required metadata | Skip candidate |
| Invalid guard/derivation/RHS evaluation | Reject candidate and retain a rule/source diagnostic |
| Conflicting metadata or exhausted internal budget | Stop checked execution |

Rust and C++ expose these through their own result APIs and checked runner
wrappers. Backend ABI versions are independently checked at compile time.
Extraction and cost models are application choices. See the
[Rust integration guide](../runtime/rust/README.md) and
[C++ integration guide](../runtime/cpp/README.md) for concrete APIs and build setup.

## Validation

```sh
bazel test //tests:codegen_test //tests:codegen_cpp_test \
  //tests:rust_codegen_test //tests:cpp_codegen_test
```

The integration scripts generate temporary applications from source and JSON,
compile sample/semantic projects, and exercise guards, locals, typed descriptors,
logical axes, metadata, and rejection before mutation. The native semantic
reference independently compares applicability, descriptor values, replacement
trees, and output facts. Rust runs debug/release tests; C++ covers GCC/Clang,
optimized UBSan, relocatable headers, and multiple translation units. See
[fixture ownership](../tests/codegen/README.md).
