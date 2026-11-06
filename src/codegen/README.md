# Code generator

`generate(ExecutableProgram, Options)` returns program-specific files and
diagnostics. It does not load source, repeat proof/synthesis, or write output.
Rust and C++ consume the same completed executable IR.

| Location | Owns |
| --- | --- |
| `common/project_plan.*` | Source membership and module paths |
| `rust/`, `cpp/` | Naming, target paths, dialect/expression/metadata/rule/graph emitters |
| `write.*` | Staged formatting, manifest ownership, drift checking |
| `../../runtime/rust/`, `../../runtime/cpp/` | Reusable execution support |

Reject target-name/path collisions before synchronization. Emit every completed
expression variant explicitly; unhandled variants should fail compilation.
Generated source depends on the shared runtime and contains no copied templates.

The [generator design](../../design/code-generation.md) defines emitted layouts,
backend boundaries, ABI checks, and the common execution contract. Application
setup belongs in the [Rust](../../runtime/rust/README.md) and
[C++](../../runtime/cpp/README.md) runtime guides.

```sh
bazel test //tests:codegen_test //tests:codegen_cpp_test \
  //tests:rust_codegen_test //tests:cpp_codegen_test
```

See [fixture ownership](../../tests/codegen/README.md) before adding a case.
