# Builtin catalogs and execution

The [public reference](../docs/references/built-ins.md) defines signatures,
permitted sections, examples, and failure behavior. Catalog-generated tables are
checked against compiled definitions. This document describes implementation
ownership and the change workflow.

## Identities and source visibility

| Definition | Responsibility |
| --- | --- |
| `src/shared/builtins/catalog.def` | Public IDs, names, signatures, arities, feature/domain grouping, section permissions |
| `src/shared/builtins/catalog.*` | Signature patterns and lookup by name or ID |
| `src/shared/builtins/axes.*` | Public axes signatures and explicit lowering targets |
| `src/shared/builtins/intrinsics/catalog.*` | Internal operation IDs, names, arities |

The expression checker resolves public names once and stores `Builtin` IDs in
checked rule/metadata IR. Axes checking uses the axes catalog. A public axes
function can target an existing builtin, an operand query, or an internal
operation; sharing implementation does not create a source alias.

Internal operations are constructed by compiler passes. An intrinsic's existence
never grants source access. Symbolic IR retains textual spellings for structural
keys and proof laws; validation and execution resolve these through catalogs.
Ordinary implementation helpers have no public or intrinsic IDs.

Feature (`list`, `integer`, `shape`, `dtype`), source context (`shape`, `dtype`,
`where`, `derive`), and catalog domain (`common`, `shape`, `dtype`) are independent.
Catalog domains are grouping metadata, not a requirement for matching runtime
subdirectories or namespaces. Current runtimes dispatch through `builtins` and
`intrinsics`.

## Lowering and execution

| Layer | Location |
| --- | --- |
| Rule checking | `src/core/checking/rules/expression.cc` |
| Metadata checking | `src/core/checking/metadata/expression.cc` |
| Axes checking | `src/core/checking/operations/semantics.cc` |
| Native public dispatch | `src/rewrite/builtins/dispatch.cc` |
| Native feature implementations | `src/rewrite/builtins/{list,integer,shape,dtype}.cc` |
| Native operators/internal operations | `src/rewrite/operators.cc`, `src/rewrite/intrinsics.cc` |
| Symbolic lowering/normalization | `src/core/semantics/graph.cc`, `symbolic/expr.cc` |
| Generated expression lowering | `src/codegen/{rust,cpp}/expression_emitter.cc` |
| Shared Rust implementations | `runtime/rust/src/{builtins,intrinsics,values}.rs` |
| Shared C++ implementations | `runtime/cpp/include/tepl/runtime/{builtins,intrinsics,values}.h` |

Both backends support completed rule expressions and typed descriptor
construction. Reusable runtime implementations are separate dependencies;
generated applications contain no copied builtin templates.

Shared axes mappings include `permute` to `gather`, operand shape/rank queries,
position complements, placement, and extent operations. `permute` selects axes
in order; full permutation legality comes from operation contracts. Consult
`axes.cc` for exact targets and the
[intrinsic inventory](../src/shared/builtins/intrinsics/README.md) for internal
operations. Symbolic simplification is partial: a retained symbolic call need
not be provable in every context.

Arguments execute once in source order. Boolean operators and conditionals are
lazy; `all` and `any` consume already evaluated Boolean lists. Metadata arithmetic
uses checked signed 128-bit computation. Rule types retain checked unsigned
positions or signed integers, with explicit conversions. Runtime range checks,
error categories, and budget handling follow the
[shared execution contract](code-generation.md#shared-execution-contract).

## Add or change a function

1. Specify visibility, contexts, types, empty inputs, bounds, overflow, and errors.
2. Add public identities/signatures to `catalog.def`/`catalog.cc`, or internal
   identities to the intrinsic catalog. Add an axes entry only when meaningful.
3. Implement native behavior and update both shared runtimes and emitters.
4. Add conformance cases, including failures, and update public examples.
5. Regenerate catalog tables and run their checks.

```sh
bazel build //:builtin_reference
bazel-bin/builtin_reference --write docs/references/built-ins.md
bazel-bin/builtin_reference --internal --write src/shared/builtins/intrinsics/README.md
bazel test //tests:builtins_test //tests:builtin_reference_test
```

Generated inventory regions belong to their catalogs/generator. Preserve their
markers and update them through the tool. `tools/build_program_schema.py`
regenerates the compiled-program schema when a change also affects that format.
For full generated execution checks, see [code generation](code-generation.md#validation).
