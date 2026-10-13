# Generated-code runtime compatibility

This package preserves the plan format and native runtime used by existing Rust
and C++ generated modules. It is separate from the compiler's axis-semantic
inference engine in `src/core/index_semantics`.

- `plan.h` and `inference_ir.h` contain the frozen codegen plan types.
- `codec.cc` serializes plans without a solver dependency. Existing codegen
  includes `src/core/semantics/solver.h`, which forwards only to this codec.
- `solver.cc`, `ordered.cc`, and `axis_domains.cc` implement the legacy runtime.
  This target requires Z3; compiler analysis and the CLI binary do not.
- `runtime.cc` preserves `tepl_semantic_solve_v1` and
  `tepl_semantic_verify_v1`. The build still exports
  `//src/core:libtepl_semantic_runtime.so` for generated applications.

Compiler analysis no longer constructs these plans or accepts the old element
semantic syntax. Compatibility fields remain in the checked IR because the
existing code generators reference them. Remove this package and those fields
when code generation is migrated to the new symbolic constructors.

`//tests:legacy_semantic_runtime_test` checks the serialization format and a
runtime request independently of the retired compiler frontend.
