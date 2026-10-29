# Tests

Select focused tests for the changed boundary; run the full suite for broader
changes. Commands below assume the repository root.

| Area | Starting targets |
| --- | --- |
| Syntax, checking, loading | `//tests:syntax_test`, `//tests:core_test`, `//tests:loading_test` |
| Axis properties and synthesis | `//tests:named_nodes_test`, `//tests:attribute_reasoning_test`, `//tests:ordered_synthesis_test` |
| Semantic proofs and suggestions | `//tests:semantic_test`, `//tests:semantic_cli_test`, `//tests:where_suggestions_cli_test` |
| Concrete execution | `//tests:rewrite_test`, operation-specific tests |
| Builtin definitions/inventories | `//tests:builtins_test`, `//tests:builtin_reference_test` |
| Generated code | `//tests:codegen_test`, `//tests:codegen_cpp_test`, `//tests:rust_codegen_test`, `//tests:cpp_codegen_test` |
| Shared runtimes | `cargo test --manifest-path runtime/rust/Cargo.toml --locked`, `//runtime/cpp:values_test` |

```sh
bazel test //tests:all --build_tests_only
```

The full `tests` target set includes CLI and Rust/C++ generation integration.
Standalone runtime and lab-specific checks have their own targets/commands.
See [codegen fixtures](codegen/README.md) for generated-code ownership and
[labs](../labs/README.md) for experiment reproduction.

Keep cases that exercise distinct behavior, known counterexamples, or failure
modes. Prefer independent metadata/tensor oracles, small dense inputs, and reuse
of compiled rules when varying only metadata. Tests support implementation
correspondence; they do not establish mathematical completeness or refinement.
