# TEPL built-ins

Start with the [public function reference](../../../docs/references/built-ins.md)
for signatures, permitted sections, examples, and failure behavior. Its function
inventories are generated from the compiler catalogs and checked by Bazel tests.

## Ownership and visibility

| Definition | Purpose |
| --- | --- |
| `catalog.def` | Public expression identities, names, features, signatures, arity, runtime domains, and section permissions |
| `catalog.h`, `catalog.cc` | Catalog types, signature patterns, lookup by name or ID |
| `axes.h`, `axes.cc` | Public axes signatures and explicit targets for shared functions, axis queries, or internal operations |
| [../intrinsics/](intrinsics/README.md) | Internal symbolic operations, separate identities, and arities |

A public function is callable from source in its declared sections. An axes-only
function is public too. An internal operation is constructed by compiler passes;
its existence never grants source access. Ordinary implementation helpers have
neither public built-in IDs nor internal-operation IDs.

The expression checker resolves names once and stores `Builtin` IDs in checked
rule and metadata IR. Native execution and emitters look up the signature by ID.
Axes checking uses the axes catalog instead of a separate handwritten signature
chain. Symbolic IR retains textual names for structural identities and proof
laws; validation and native execution resolve those names through the catalogs.

## Features, contexts, and runtime domains

These are independent properties:

- **Feature:** `list`, `integer`, `shape`, or `dtype`. Groups the specification
  and native implementations.
- **Context:** operation `shape` or `dtype`, rule `where` or `derive`. Controls
  source availability. Axes signatures are declared separately because they
  operate on position sequences, axis identities, and operand references.
- **Runtime domain:** `common`, `shape`, or `dtype`. Selects the namespace in
  generated C++/Rust runtimes. A feature change does not rename runtime APIs.

A dialect contains operation definitions; availability is checked at each
expression section. Shared names do not imply identical types in every context.
Metadata uses checked signed `i128`; rule integer positions use unsigned `u64`
(`u64`), and integer pairs may use a shared `u64` or `i64` type. There is no
implicit signed/unsigned conversion or floating-point built-in overload.

The axes catalog makes these shared implementations explicit:

| Axes function | Execution target | Symbolic lowering |
| --- | --- | --- |
| `len`, `range`, `concat`, `gather`, `is_valid_axis_list` | Same public `Builtin` ID | Same operation name |
| `permute` | `Builtin::kGather`, with axes input | `gather` |
| `ordered_complement` | `Builtin::kExclude`, with position inputs | `ordered_complement`, retained for proof laws |
| `all_axes`, `rank`, `shape` | Operand shape queries | `range(len(shape))`, `len(shape)`, shape value |
| `place` | `Intrinsic::kPlaceAxes` | Axis placement, or a gather/permutation expression for dimensions |
| Extent arithmetic and predicates | Explicit `Intrinsic` targets | Same internal operations |

`permute` selects axes in the supplied order; its name alone does not impose a
complete-permutation requirement. Operation assertions and axis legality supply
additional constraints. Do not introduce aliases merely because implementations
share a loop.

## Implementation map

| Layer | Location |
| --- | --- |
| Rule type checking and bindings | `src/core/checking/rules/expression.cc` |
| Shape/dtype type checking and bindings | `src/core/checking/metadata/expression.cc` |
| Axes type checking | `src/core/checking/operations/semantics.cc` |
| Native public dispatch | `src/rewrite/builtins/dispatch.cc` |
| Native list, integer, shape, dtype behavior | `src/rewrite/builtins/{list,integer,shape,dtype}.cc` |
| Native internal operations and symbolic boundary | `src/rewrite/intrinsics.cc` |
| Operators, indexing, checked primitive helpers | `src/rewrite/operators.cc`, `evaluation_support.cc` |
| Symbolic lowering and simplification | `src/core/semantics/graph.cc`, `expr.cc` |
| C++/Rust call lowering | `src/codegen/{cpp,rust}/builtin_emitter.cc` |
| Generated runtime implementations | `templates/{cpp,rust}/src/builtins/` |

Generated modules copy runtime templates into their own `builtins/` directory.
Public list and integer helpers use `common`; shape helpers use `shape`; dtype
predicates use `dtype`. Source code belongs in templates, not generated lab IR.

## Execution and support

All public expression built-ins have native, generated C++, and generated Rust
implementations. Generated support for an enclosing expression is a separate
question: extended rule expressions and descriptor construction remain deferred
and produce diagnostics. The native engine supports those forms.

Symbolic simplification is partial. When a call cannot simplify, it can remain
symbolic; this does not imply that a proof involving it can be established.
Axes contracts use compiler proof machinery and native inference. Generated
backend support is checked by each backend before output.

Arguments execute once in source order. Boolean operators and conditionals
retain short-circuit evaluation. `all` and `any` consume an already evaluated
Boolean list; they do not make list construction lazy. A public built-in failure
rejects a rule candidate or makes available metadata inference invalid. Native
step and list limits remain enforced separately from evaluation errors.

## Add or change a function

1. Decide whether it is public, axes-only, or internal. Write its contract,
   including empty inputs, bounds, overflow, and failures.
2. For a public expression function, add one `catalog.def` entry. Reuse signature
   patterns in `catalog.cc`, or introduce a pattern when necessary. Choose
   feature, runtime domain, and allowed sections independently.
3. Add native behavior in the appropriate feature file. Extend C++/Rust runtime
   implementations and call lowering as required. Unsupported enclosing syntax
   must produce an explicit diagnostic.
4. Add an axes entry only when its types and semantics are meaningful there.
   Point it at the existing built-in ID when the contract is shared. Internal
   operations belong in the internal catalog; proof laws stay with proof code.
5. Update behavioral examples in the public reference. Add conformance cases and
   failures to `tests/builtins_test.cc` and relevant generated runtime tests.
6. Regenerate the inventories and run validation.

```sh
python3 tools/build_program_schema.py
bazel build //:builtin_reference
bazel-bin/builtin_reference --write docs/references/built-ins.md
bazel-bin/builtin_reference --internal --write src/shared/builtins/intrinsics/README.md
bazel test //tests:builtins_test //tests:builtin_reference_test
```

`tests/builtins_test.cc` covers every public ID and checks visibility, explicit
axes mappings, edge cases, failures, and list limits. Core and metadata tests
check type inference. Rewrite and semantic tests exercise full execution and
proofs. Generated runtime tests cover C++/Rust behavior in their own integer and
error domains; run the integration commands in the
[developer guide](../../../docs/developer_guide.md#validate-changes).
