# Compiler and runtime fixtures

These catalogs preserve regression coverage independently of the public learning
examples in [examples/](../../examples/README.md).

| Directory | Coverage |
| --- | --- |
| `language/` | Metadata contracts, descriptors, inheritance, literals, graph bindings, multiple dialects |
| `semantic/` | Axis/value contracts, algebraic proofs, constructors, calculated reads, supported operation families |
| `rewrite/` | CLI compilation, JSON/source parity, rule selection, sharing, and limits |

The language and semantic catalogs retain their original filenames and generated
module names so Rust/C++ conformance applications can exercise the same APIs.
The fixture package exports source files as Bazel data. Generation scripts consume
these directories and keep their output module names `sample` and `semantic`.

Use the public examples for tutorials. Add regression cases here when broad
coverage would distract from a learning example.
