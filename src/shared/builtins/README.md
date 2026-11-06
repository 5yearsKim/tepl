# Builtin definitions

This directory owns source-callable builtin identities, signatures, availability,
and axes lowering targets. Start with the
[public reference](../../../docs/references/built-ins.md) for language usage.

| File | Responsibility |
| --- | --- |
| `catalog.def` | Public IDs, names, features, signatures, arity, domains, contexts |
| `catalog.*` | Signature patterns and lookup |
| `axes.*` | Public axes signatures and explicit execution targets |
| [intrinsics/](intrinsics/README.md) | Compiler-only IDs and generated inventory |

Source visibility is explicit. Shared execution helpers and intrinsic IDs do
not grant source access. Checked rule/metadata expressions retain resolved IDs;
symbolic spellings are validated at their execution boundary.

The [builtin design](../../../design/builtins.md) maps native and generated
implementations, explains evaluation contracts, and gives the change workflow.
Both generated backends use separate shared runtimes rather than copied templates.

```sh
bazel test //tests:builtins_test //tests:builtin_reference_test
```

Preserve catalog markers in generated inventories. Regenerate their tables with
`bazel-bin/builtin_reference --write` or `--internal --write` as described in the
design guide.
