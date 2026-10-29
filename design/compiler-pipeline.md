# Compiler pipeline

TEPL compiles source through explicit, independently callable stages:

```text
SourceFile
  → syntax::ParsedModule
  → resolution::ResolvedProgram
  → checking::TypedProgram
  → semantics::VerifiedProgram
  → executable::ExecutableProgram
```

[`src/core/pipeline.h`](../src/core/pipeline.h) connects these stages.
`compile(CompileRequest, SourceLoader&)` runs the complete pipeline. Stage
representations live in `program.h`; transformation headers expose the next
boundary. Paths in the following table are relative to `src/core/`.

| Transformation | Header | Representation target |
| --- | --- | --- |
| `resolution::resolve(SourceProject)` | `resolution/resolve.h` | `//src/core/resolution:model` |
| `checking::check(ResolvedProgram)` | `checking/check.h` | `//src/core/checking:model` |
| `semantics::verify(TypedProgram)` | `semantics/verify.h` | `//src/core/semantics:verified_model` |
| `semantics::inspect(TypedProgram)` | `semantics/inspect.h` | `//src/core/semantics:inspection_model` |
| `lowering::lower(VerifiedProgram)` | `lowering/lower.h` | `//src/executable` |

## Stage guarantees

| Stage | Published guarantee |
| --- | --- |
| Syntax | One parsed source module, with original spellings and locations. ANTLR objects remain inside the parser. |
| Resolution | Imports and visibility are established, operation/template references are resolved, and concrete templates are expanded. Source storage is owned. |
| Checking | Types, descriptor schemas, locals, operation metadata, and concrete graphs are checked. Field-level inference requests remain explicit holes. |
| Verification | Required proofs succeeded under the selected arithmetic policy. Constructors and logical-axis expressions are completed. |
| Lowering | Runtime tables and expressions are validated. No holes, proof plans, or compiler-only references enter executable IR. |

Representations have restricted construction and expose immutable data.
`Result<T>` contains a completed program or diagnostics. A failed stage cannot
manufacture a successful input for the next stage. Representation targets do
not link earlier stage implementations; lowering does not link proof search.

Typed rules contain holes, inspection results contain proof attempts, and
verified rules contain completed metadata computations. Inspection may retain
unknown results for diagnostics, but cannot be passed to executable lowering.
Verification consumes its owned inspection result during materialization while
leaving the caller's typed program available for inspection.

## Source loading and ownership

`SourceLoader` supplies imported source. `FileSourceLoader` reads files; tests
can inject virtual modules and project configs. A loading session caches parsed
modules by source identity and caches the nearest enclosing `tepl.toml`.
Project discovery and independent graph/rule inputs share this session.

Bare imports resolve against each file's project root. Explicit `./` and `../`
imports remain relative to the importing file. Without a project config, all
non-absolute imports are file-relative. Virtual loaders can override
`findProjectConfig` and supply config text through `read`.

`loadImports` returns a `SourceProject` without mutating parsed modules.
`loadProject(vector<SourceFile>, SourceLoader&)` supports explicit in-memory
roots. The project owns its module collection, selected declarations, and
normalized import edges. Resolution consumes these edges without repeating
filesystem discovery.

Resolution snapshots rule expressions, operation metadata, and scalar semantics.
Published pointers are const, and mutable aliases from manually constructed ASTs
cannot change resolved definitions. Graph/expression identity uses node IDs.
Checking finalizes the complete type table before verification begins.

## Semantic and executable boundaries

[Semantic verification](semantic-verification.md) establishes legality and
value equivalence. [Attribute synthesis](attribute-synthesis.md) proposes
constructors; an independent verification attempt must accept them before
materialization. Source conditions remain explicit throughout this process.

[`src/executable/`](../src/executable/) owns executable representation,
validation, and serialization. [`src/rewrite/`](../src/rewrite/) reads executable
and shared data without linking parsing, checking, or proof algorithms.
Executable operations retain shape contracts; scalar value semantics stay in
the compiler. The wire format is version 2; older versions are rejected.

Verified and executable rules use `core::completed` expressions and metadata
instructions. They have no symbolic holes or unresolved output-group references.
Validation checks IDs, types, instruction arguments, and bindings at this
boundary. Shared source locations, diagnostics, concrete types, operation/schema
tables, builtin catalogs, and metadata representations live in `src/shared/`.
Inference and proof search remain in their owning stages.

[Code generation](code-generation.md) consumes `ExecutableProgram` directly.
Rust and C++ are implemented; Python generation reports a diagnostic. Neither
generated runtime performs compiler proof search or attribute synthesis.
