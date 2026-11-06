# Compiler core

`pipeline.h` connects source loading, parsing, resolution, checking, semantic
verification, and executable lowering. The full stage contracts and ownership
rules live in [the compiler pipeline design](../../design/compiler-pipeline.md).

| Directory | Owns |
| --- | --- |
| `syntax/` | Parsed modules, source AST, ANTLR integration |
| `resolution/` | Import visibility, source ownership, template expansion |
| `checking/` | Typed rules, schemas, metadata programs, graph checks |
| `semantics/` | Inspection, independent proof, verified programs |
| `attr_synth/` | Unverified constructor candidates and strategy adapters |
| `lowering/` | Verified-to-executable translation |

Depend on a stage's representation target when only reading its output.
Published programs are immutable; restricted constructors prevent bypassing
stage guarantees. Inspection results cannot enter executable lowering.

See [semantic verification](../../design/semantic-verification.md) for proof
obligations, [attribute synthesis](../../design/attribute-synthesis.md) for field
roles and strategies, and [code generation](../../design/code-generation.md) for
the runtime boundary. Source conditions remain explicit; failed proof attempts
cannot become hidden applicability guards.

Use [the test guide](../../tests/README.md) to select checks for a changed stage.
