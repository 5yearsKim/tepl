# Code generation fixtures

Rust and C++ share the current source fixtures in `rust/`, along with the language
and semantic catalogs in [fixtures](../fixtures/README.md). `tools/test_codegen.sh`
and `tools/test_codegen_cpp.sh` generate temporary applications and compare
source/JSON output.

`current.tepl` covers field-level synthesis, preserved supplied/default fields,
logical axes and guards on derived descriptors. `codegen_semantic_reference` uses
the native evaluator to generate independent Rust assertions or C++ assertions
(`--cpp`) for applicability, descriptors, replacement trees, and output facts.
Both backends execute them in debug and optimized builds.

`cpp/integration.cc` checks egg-c execution, witness pairing, rejection before
insertion, metadata refinement/conflicts, literals, binders, multiple namespaces,
and typed construction. `cpp/odr.cc` verifies multiple translation units. Runtime
numeric boundaries and logical-axis checks live in `runtime/cpp/tests/values.cc`.
`cpp/strings.cc` preserves embedded NUL bytes; `cpp/witnesses.*` checks metadata
refinement across competing attribute witnesses. The retired host-function runtime
and fixtures have been removed.
