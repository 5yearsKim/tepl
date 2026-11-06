# Compiled program format

`tepl compile` stores a checked executable program as versioned JSON. The TEPL
engine and Rust/C++ code generators consume the same program.

```sh
bazel-bin/tepl compile examples/projects/transpose -o /tmp/program.json
bazel-bin/tepl validate /tmp/program.json
bazel-bin/tepl schema -o /tmp/rewrite-program.schema.json
bazel-bin/tepl generate /tmp/program.json --target rust -o /tmp/generated
```

See [run rewrites](../runtime/engine/run.md) for execution and
[embed the engine](../runtime/engine/embed.md) for the C++ API.

## Contents

The format is declared in
[`rewrite-program.schema.json`](../../schema/rewrite-program.schema.json).
Compile [`double_negation.tepl`](../../examples/basic/double_negation.tepl) to inspect a small
program's JSON output locally.

The envelope contains `format: "tepl.rewrite"`, `version: 4`, an arithmetic
`policy`, required `capabilities`, the `program` tables, and rule-scoped
`axis_definitions`. Rules contain explicit match and replacement trees,
constraints, typed conditions, and descriptor derivations. Expression variants
use `kind` and `data`; references use table IDs. Operation metadata programs
and concrete graphs travel with the rules.

Compilation resolves inferred axis attributes into explicit constructors and
retains semantic proof conditions as executable guards. Unresolved holes and
proof search state are absent. Source locations remain for diagnostics and
exported dialect imports; executing JSON does not read those source files.

Tensor extents and numeric literals use decimal strings to preserve their full
integer range across JSON implementations. IDs are JSON integers bounded by
`2^53 - 1`. Unknown fields, variants, versions, and capabilities are rejected.

## Validation

A standard JSON Schema validator checks structure. TEPL also checks references,
scope, operation arity, expression types, and graph topology. Its built-in schema
validator implements the vocabulary used by this schema; it is not a general
JSON Schema tool. Loading limits input to 64 MiB and JSON nesting to 256 levels.
Duplicate object keys are rejected.

Validation does not prove that a manually edited rule preserves tensor values.
Source rules with axis semantics pass through semantic verification before
compilation; ordinary rules retain their existing checking guarantees.
Serialized shape/dtype caches are not trusted by execution: metadata is inferred
from operation contracts.

## Source export

The graph printer emits inputs, concrete attributes, shared `let` bindings, a
`yield`, and absolute dialect imports. Rechecking exported source requires the
referenced dialect files. If artifact provenance paths are relative, export
from the original source root or adjust imports after moving the source.
The JSON itself remains executable without those source files.

## Implementation boundaries

| Directory | Responsibility |
| --- | --- |
| `src/core/` | Parsing, resolution, checking, semantic verification, and lowering |
| `src/executable/` | Executable program, JSON codec, and validation |
| `src/shared/` | Types, catalogs, and metadata representations |
| `schema/` | Public JSON Schema |
| `src/rewrite/` | Matching, replacement, scheduling, and graph printing |
| `src/codegen/` | Backend generation from the compiled program |

`ConcreteGraph` is the shared DAG representation: indexed nodes, operands,
concrete attributes, optional shape/dtype, named bindings, and a root.

## Changing the format

`tools/build_program_schema.py` contains the explicit field catalog and generates
the schema, codec definitions, and embedded schema. Regenerate them together:

```sh
python3 tools/build_program_schema.py
```

An incompatible change needs a new version and matching loader behavior.
Recompile older executable artifacts. Version 3 introduced `u64` metadata and
removed the `index` type spelling; source using that spelling needs migration.
