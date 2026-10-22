# Compilation pipeline

TEPL compiles source through explicit, independently callable stages:

```text
SourceFile
  → syntax::ParsedModule
  → resolution::ResolvedProgram
  → checking::TypedProgram
  → semantics::VerifiedProgram
  → executable::ExecutableProgram
```

`pipeline.h` connects the stages. `compile(CompileRequest, SourceLoader&)` runs
all of them. Call individual stages to inspect or test a boundary. Each stage keeps its
representation in `program.h` and exposes its transformation separately:

| Transformation | Header | Representation target |
| --- | --- | --- |
| `resolution::resolve(SourceProject)` | `resolution/resolve.h` | `//src/core/resolution:model` |
| `checking::check(ResolvedProgram)` | `checking/check.h` | `//src/core/checking:model` |
| `semantics::verify(TypedProgram)` | `semantics/verify.h` | `//src/core/semantics:verified_model` |
| `semantics::inspect(TypedProgram)` | `semantics/inspect.h` | `//src/core/semantics:inspection_model` |
| `lowering::lower(VerifiedProgram)` | `lowering/lower.h` | `//src/executable` |

Representation targets do not link earlier stage implementations. Lowering uses
the verified representation without linking proof algorithms.

| Directory | Public output | Contract |
| --- | --- | --- |
| `syntax/` | `syntax::ParsedModule` | Valid syntax for one source file. Names and imports retain their written forms and locations. ANTLR objects stay inside the parser. |
| `resolution/` | `resolution::ResolvedProgram` | Imports and visibility are established, operation and template references in rules are resolved, and concrete rule templates are expanded. Source storage is owned. |
| `checking/` | `checking::TypedProgram` | Types and descriptor schemas are finalized; local bindings, metadata programs, and concrete graphs are checked. RHS inference requests are explicit attribute holes. |
| `semantics/` | `semantics::VerifiedProgram` | Required proofs succeeded under the selected policy. RHS attribute constructors and output-group expressions have been materialized. |
| `lowering/` | `executable::ExecutableProgram` | Compiler rules are translated into validated runtime tables and rules. No attribute holes or proof plans enter the runtime representation. |

Typed rules contain attribute holes, inspection rules own proof attempts, and
verified rules contain completed metadata computations. These are distinct data
representations. Only inspection results contain rule proof plans.

The checked program classes expose immutable data and have restricted
construction. `Result<T>` contains a completed program or diagnostics. A failed
stage cannot manufacture a successful output for the next stage.

## Source loading and ownership

`SourceLoader` supplies imported source. `FileSourceLoader` reads files; tests
can provide an in-memory loader. `loadImports` returns a `SourceProject` without
mutating the parsed module. `SourceProject` contains the parsed module collection
and the selected declarations; it does not inherit from a single module.
Imports and visibility remain attached to their owning modules.

Each loading session caches parsed modules by source identity. Root files and
imports share the cache, so repeated imports read and parse each source once.
Project discovery and independent graph/rule inputs use this same session and
accept an injected loader. `loadProject(vector<SourceFile>, SourceLoader&)`
supports explicit roots in memory.

`FileSourceLoader::findProjectConfig` discovers the nearest enclosing
`tepl.toml`. The loading session caches configs and resolves bare import paths
against each file's project root. Explicit `./` and `../` paths stay relative
to the importing file. Without a config, all non-absolute imports remain
file-relative. Virtual loaders can override `findProjectConfig` to supply a
config identity and return its text through `read`; by default they perform no
config discovery.

`SourceProject::import_targets` records normalized import edges. Symbol
resolution consumes these targets without rediscovering configs or reading
the filesystem. Parsed imports retain their original text.

Resolved graph and expression metadata uses node IDs, rather than pointer keys.
Resolution snapshots all source expression trees, including operation metadata
and scalar semantics. Its published pointers are const, and callers cannot alter
resolved definitions through mutable aliases retained from manually built ASTs.
Type checking publishes its complete type table before semantic verification.

## Semantic compilation

`semantics/index_maps/` owns symbolic axis mapping and transport algorithms.
`symbolic/` owns expression construction and normalization, `synthesis/` proposes
attribute constructors, and `proof/` verifies their equivalence and legality.
Each proof context owns its facts, work budget, and caches.

Optional operation labels survive checking as occurrence metadata, separately
from tensor captures and descriptors. Source spellings remain compiler-only
metadata for suggestions. `node.attrs` resolves to a descriptor reference;
`derive` assignments define explicit RHS descriptors and omission creates holes.
Logical axis properties stay typed until both semantic trees exist, then lower
to closed queries over the shared axis analysis. Source group positions can
also use the general symbolic mapper. Membership uses `LogicalAxisSet`,
sequences use `LogicalAxis[]`, and positions use `index[]`.

`suggestion.*` prints a complete named source rule with the generated feasibility
conditions. It preserves operation qualification, existing node names, original
conditions and explicit constructors, and chooses fresh local names. The
`--infer-applicability` path emits the corresponding typed conjunction;
`shared/catalog/axis_conditions.*` owns both its recognition and the shared
source/derived-condition schedule. Runtime evaluation and executable validation
use that same schedule. Actual inferred layouts cannot be read before
construction; boundary membership and required subsequences can.

Symbolic keys are immutable DAGs. Their deterministic digests accelerate lookup;
structural comparison distinguishes hash collisions. Each proof context owns its
canonical-key arena, so repeated DAGs do not expand into trees. No process-wide identity
registry determines expression identity or proof order.

`semantics::inspect` may retain unknown proofs for diagnostics. Its result is not
a `VerifiedProgram` and cannot be passed to executable lowering. Structural
rules have no semantic proof; semantic rules must pass the selected arithmetic
policy. Conditions remain explicit, and synthesis cannot invent applicability
restrictions.

Verification consumes its owned inspection result during materialization. It
moves common tables and converts metadata directly into completed expressions.
Output-group references are resolved during that conversion. The caller's typed
program remains available for inspection.

## Runtime boundary

`src/executable/` owns the runtime representation, validation, and serialization.
`src/rewrite/` depends on executable and shared data, without depending on the
parser, resolver, type checker, or proof algorithms. Compiled programs can be
loaded and executed without the compiler stages. Executable operations contain
shape contracts; their scalar value semantics remain in the compiler. The wire
format is version 2, and older format versions are rejected.

Verified and executable rules use `core::completed` expressions and metadata
instructions. These types have no inferred descriptor kind, symbolic hole, or
output-group reference variant. Compiler-only expressions live in checking, and
proof symbols and keys live in `semantics/symbolic/`. Validation still checks
IDs, types, instruction arguments, and bindings at the executable boundary.

Shared components live in `src/shared/`: source locations, diagnostics, concrete
types, operation/schema tables, builtin catalogs, and metadata representations.
Type inference and proof search stay in their owning stages.

## Deferred code generation

Code generation sources and templates remain for a later migration. Their old
compiler interfaces and legacy inference runtime have been removed. The active
CLI reports generation as unavailable, and backend targets are excluded from
the active build. Backend migration should consume `ExecutableProgram` directly.
