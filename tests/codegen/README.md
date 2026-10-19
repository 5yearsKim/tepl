# Code generation fixtures

The active integration suite uses `rust/` and the sample and semantic projects
through `tools/test_codegen.sh` and `//tests:rust_codegen_test`.

Other sibling fixtures and `cpp/` contain historical tests for the retired
generator interfaces. C++ generation is unavailable; its integration script
reports that status without attempting regeneration. Obsolete host-call fixtures
have been removed. `$` calls and host `fn` parameters are rejected by
`//tests:syntax_test` and `//tests:core_test`.
