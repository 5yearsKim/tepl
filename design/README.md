# Implementation design

These documents describe the current compiler and generated runtimes. Start with
[the public manual](../docs/index.md) for language usage and application tutorials.

| Topic | Read when changing |
| --- | --- |
| [Compiler pipeline](compiler-pipeline.md) | Stage boundaries, source ownership, executable lowering |
| [Semantic verification](semantic-verification.md) | Axis/value contracts, normalization, proof obligations |
| [Attribute synthesis](attribute-synthesis.md) | Candidate construction, strategy fallback, ordered solver |
| [Code generation](code-generation.md) | Rust/C++ emitters and their shared execution contract |
| [Builtins](builtins.md) | Catalogs, dispatch, source availability, conformance |

Component READMEs beside source code identify local entry points and invariants.
Language-specific integration belongs in the [Rust](../runtime/rust/README.md)
and [C++](../runtime/cpp/README.md) runtime guides. Mathematical claims and draft
algorithms belong in [research](../research/README.md); experiment measurements
belong beside their inputs in [labs](../labs/README.md).

Maintain one full explanation of each contract. Link to it from local READMEs.
Update a design document when behavior or ownership changes; move a superseded
research proposal to its archive rather than presenting it as current support.
Generated catalogs and recorded measurements retain their own provenance.
