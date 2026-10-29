# Code generation fixtures

The active integration suite uses `rust/` and the sample and semantic projects
through `tools/test_codegen.sh` and `//tests:rust_codegen_test`.

`current.tepl` covers field-level axis synthesis, preserved supplied/default
fields, logical-axis sets and sequences, and guards on derived descriptors.
`codegen_semantic_reference` evaluates each graph/rule pair with the native
engine and emits independent Rust assertions for applicability, descriptors,
replacement trees, and shape/dtype facts. These run in debug and release modes.

Other sibling fixtures and `cpp/` contain historical tests for the retired
generator interfaces. C++ generation is unavailable; its integration script
reports that status without attempting regeneration. Obsolete host-call fixtures
have been removed. `$` calls and host `fn` parameters are rejected by
`//tests:syntax_test` and `//tests:core_test`.
