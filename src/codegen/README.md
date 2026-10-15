# Rust code generation

`generate(ExecutableProgram, Options)` returns Rust source files and diagnostics.
It does not read source files, repeat semantic verification, solve attributes, or
write output. The CLI compiles source or validates and loads JSON before calling
this interface.

The generated module depends on the shared [`tepl-runtime`](../../runtime/rust/)
crate and Rust edition 2024. C++ and Python generation are unavailable.

## Boundaries

| Component | Input | Output |
| --- | --- | --- |
| Project planning | Executable tables and optional module roots | Source module membership |
| Rust naming | Project plan and declarations | Validated Rust names and paths |
| Rust emitters | Executable IR and names | Program-specific Rust files |
| Output synchronization | Generated files and destination | Formatted files and ownership manifest |
| Rust runtime | Generated program implementation and graph inputs | egg analysis and rewrite execution |

`common/project_plan.*` preserves module paths under `rules/` and `graphs/`.
Explicit roots override this convention. Naming rejects collisions before any
files are written. Source and JSON inputs generate the same files when their
executable data and generation options agree.

## Generated code

- `mod.rs`: combined operation and attribute enums, the runtime `Program`
  implementation, concrete egg type aliases, and the semantic policy.
- `dialects/`: dialect operation enums and one public struct per attribute
  schema. Shared schemas are emitted once and reused by operations.
- `metadata.rs`: shape, axes, and dtype contract evaluation. Explicit shape
  programs take precedence over axes contracts, matching the executable engine.
- `rules/`: patterns, restrictions, ordered conditions, local bindings, typed
  descriptor derivations, and replacement preparation. `rules::all()` builds
  the egg rewrites. Each rule also exposes `definition().build()`.
- `graphs/`: constructors that register input metadata and build egg graphs.

Expression emission is exhaustive over completed executable expressions.
Generated functions call shared builtins and internal intrinsics, preserve
short-circuit evaluation, and check value ranges. Synthesized attributes become
ordinary functions returning their concrete schema structs.

## Runtime

The runtime crate owns value conversion, checked arithmetic, axis identity,
metadata facts, attribute-aware matching, replacement preparation, and checked
runner integration. Its `Program` trait is the only generated/runtime interface.
The crate does not depend on generated programs or parse compiled JSON.

Shape and dtype facts merge independently. Unknown means no evidence; differing
known facts produce a conflict. Input metadata must be registered before its
node is inserted. Invalid inference and conflicting facts record diagnostics.
Use the runtime's runner functions to propagate callback errors to the caller.

Matching keeps attribute values with their concrete node witnesses. The egg
adapter rematches during application to recover those witnesses using current
canonical tensor bindings. All conditions, derivations, and RHS validation finish
before insertion and union. A known output metadata mismatch rejects a candidate.
Missing required metadata skips it. Evaluation errors and conflicts stop execution.

## Validate

```sh
bazel test //tests:codegen_test //tests:rust_codegen_test
cargo test --manifest-path runtime/rust/Cargo.toml --locked
```

`tools/test_codegen.sh` generates temporary test applications from source and
JSON, compiles the full sample and semantic projects, and executes focused
fixtures in debug and release modes. Runtime tests cover numeric boundaries,
resource limits, metadata merging, and axis identity.
