# Developer guide

This guide covers contributing to TEPL's compiler and generated runtimes.
For language concepts and application setup, use the
[core concepts](2_core_concept_tepl.md), [Rust tutorial](tutorial/rust/02_run.md),
and [CLI reference](references/cli.md).

Run the commands below from the repository root.

## Build and inspect

Follow the [build guide](tutorial/rust/01_build.md) for prerequisites.

```sh
bazel build //:tepl
bazel-bin/tepl check examples/sample
bazel-bin/tepl parse examples/sample/rules/lora.tepl --ast
```

Use `parse --tree` to inspect the ANTLR tree, `parse --ast` for the source AST,
and `check` for semantic diagnostics and checked IR. With `bazel run //:tepl --`,
use absolute input paths because the binary starts in its runfiles directory.

## Build and publish a Docker image

Build the TEPL compiler image and check it:

```sh
docker build -t tepl:local .
docker run --rm tepl:local --help
```

To publish, create a `tepl` repository on Docker Hub and replace `yourname`
below with your username. These platforms cover x86 computers and all
Apple Silicon Macs:

```sh
docker login
docker buildx create --driver docker-container --bootstrap --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t {yourname}/tepl:{version} --push .
```

Pass `--no-format` when generating code with the image; it omits formatters.

## Where to make changes

The compiler parses source, resolves imports, checks types, verifies semantics,
and lowers rules into executable programs. See the
[compilation pipeline](../src/core/README.md) for stage contracts. Backend migration
is deferred; runtime templates and code generation sources remain for that work.
Stage representations live in `program.h`; the `resolve.h`, `check.h`, `verify.h`,
and `inspect.h` headers expose transformations. Depend on the representation
target when a module only reads the output of a stage.

| Area | Location |
| --- | --- |
| Language grammar | [grammar/Tepl.g4](../grammar/Tepl.g4) |
| AST and parsing | `src/core/syntax/` |
| Imports and project discovery | `src/core/resolution/` |
| Type checking | `src/core/checking/` |
| Synthesis and verification | `src/core/semantics/` |
| Executable lowering | `src/core/lowering/` |
| Runtime program and serialization | `src/executable/` |
| Shared definitions and builtin catalogs | `src/shared/` |
| Shared generation plans and language emitters | [src/codegen/](../src/codegen/README.md) |
| Reusable runtime algorithms | [runtime/rust/](../runtime/rust/README.md); C++ templates remain deferred |
| Compiler and generated-code fixtures | `tests/`, `tests/codegen/` |

Edit the grammar and AST builder when changing syntax. Bazel regenerates the
ANTLR parser; generated parser files belong in the build output.

Keep language checks in core, declaration-dependent output in emitters, and
shared Rust runtime behavior in `runtime/rust/`. Generated application files
are replaced on regeneration; edit the runtime crate or emitters instead.

## Built-in specifications

Use the [public built-in reference](references/built-ins.md) to check signatures,
section availability, examples, and failures. The [built-in architecture guide](../src/shared/builtins/README.md)
maps each responsibility to its implementation and explains how to add a function.
[Internal operations](../src/shared/builtins/intrinsics/README.md) have separate identities
and never grant source access by themselves.

Function inventory tables come from the compiled catalogs. After changing a
catalog, regenerate them from the repository root:

```sh
python3 tools/build_program_schema.py
bazel build //:builtin_reference
bazel-bin/builtin_reference --write docs/references/built-ins.md
bazel-bin/builtin_reference --internal --write src/shared/builtins/intrinsics/README.md
```

After building `//:tepl`, run the built-in fixtures without regenerating lab IR:

```sh
./tools/test_codegen.sh "$PWD/bazel-bin/tepl" builtins
./tools/test_codegen_cpp.sh "$PWD/bazel-bin/tepl" builtins
```

`bazel test //tests:builtin_reference_test` detects stale inventory tables.
Behavioral prose and examples remain maintained alongside conformance tests.

## Format code

Install `clang-format`, then format or check first-party C++ sources:

```sh
bazel run //:format
bazel run //:format -- --check
```

This covers `src/`, `tests/`, and `tools/` using the repository's
`.clang-format`. Set `CLANG_FORMAT` to choose another executable.
Format Rust runtime changes with `cargo fmt --manifest-path runtime/rust/Cargo.toml`.

## Validate changes

| Command | Coverage |
| --- | --- |
| `bazel test //...` | Parser, imports, semantic analysis, CLI, and code generation. |
| `./tools/test_codegen.sh` | Generate temporary Rust applications and run execution and evaluator-reference tests. |
| `cargo clippy --manifest-path runtime/rust/Cargo.toml --all-targets -- -D warnings` | Check the shared Rust runtime. |

Rust integration needs Cargo, Rust with edition 2024 support, and `rustfmt`.
The integration script uses Cargo's local dependency cache. Run the compiler
suite for frontend changes and the Rust integration suite for emitter or runtime
changes. The script builds TEPL by default; pass a compiler binary and the
`codegen_reference` helper to use existing builds. C++ generation is deferred.

## Develop Rust runtime logic

The [Rust lab](../labs/rust-egg/README.md) exercises `examples/rewrite/`.
Its `src/ir/` is generated and ignored by Git.

1. Edit TEPL declarations, `runtime/rust/src/`, or `src/codegen/rust/`.
2. Keep runtime tests in `runtime/rust/tests/` and generation fixtures in
   `tests/codegen/rust/`.
3. Run `bazel test //tests:codegen_test //tests:rust_codegen_test`.

`tools/test_codegen.sh` generates temporary applications, compares source and
JSON output, checks builtin behavior against the executable evaluator, and runs
Rust tests in debug and release modes. C++ generation remains deferred.

## Language and runtime documentation

- [Shape](references/shape.md), [dtype](references/dtype.md), and
  [builtins](references/built-ins.md): metadata programs and expression support.
- [Attributes and conditions](advanced/attributes_and_conditions.md):
  descriptor capture and derivation.
- [Binding and early pruning](advanced/binding_and_early_pruning.md):
  matching and condition evaluation.
- [Tensor analysis](advanced/custom_analysis.md): input facts and metadata consistency.
- [Graph examples](../examples/sample/graphs/README.md): construction and insertion.
