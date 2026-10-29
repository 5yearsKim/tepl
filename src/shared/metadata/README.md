# Operation metadata representation

Operation shape/dtype blocks share checked expressions, statements, symbols, type
inference, and builtin resolution. `ProgramKind` selects parameter/yield types:
shape uses integer lists; dtype uses dtype values. Variadic parameters bind lists
of operand metadata. Programs own their tables and retain no source AST pointers.

| Location | Responsibility |
| --- | --- |
| `ir.h`, `types.*` | Checked expressions, bindings, types, and program kind |
| `instruction.h`, `inference.h` | Completed symbolic instructions and native inference representation |
| `src/core/checking/metadata/` | Parameters, statements, contracts, expression/type checks |
| `src/core/checking/dialects/check.cc` | Invoke metadata checking for every operation |
| `src/core/semantics/` | Symbolic lowering and compiler-only reasoning |
| `src/codegen/{rust,cpp}/metadata_emitter.*` | Translate operation contracts into generated functions |

Checking runs even if no rule uses an operation. Completed instructions have no
inference holes. Shape values use checked signed 128-bit computation with unsigned
64-bit output dimensions; dtype values support equality/classification without
implicit promotion. Assertions and locals execute in source order.

See [builtins](../../../design/builtins.md), the
[execution contract](../../../design/code-generation.md#shared-execution-contract),
and the public [shape](../../../docs/references/shape.md) and
[dtype](../../../docs/references/dtype.md) references.
