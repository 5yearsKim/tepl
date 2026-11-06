# Semantic regression catalog

This project is the broad operation and rule catalog used by semantic reasoning
tests, scalability workloads, and Rust/C++ conformance applications. Keep rule
filenames stable unless their generated API consumers are updated too.

Generation requires `--semantic-policy algebraic` because the catalog includes
reassociation and distribution. Tests can select individual files and rules under
stricter policies.

See [fixture ownership](../README.md). For focused learning material, use
[Advanced examples](../../../examples/advanced/README.md).
