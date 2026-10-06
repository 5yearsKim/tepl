# TEPL command-line reference

The `tepl` command initializes projects, parses source files, checks declarations,
and generates Rust or C++ code. See the [build guide](../tutorial/rust/01_build.md)
to build the executable. Examples below assume `tepl` is on your `PATH`; from
the repository root, you can use `bazel-bin/tepl` instead.

## Commands

```text
tepl init <directory>
tepl parse <file> [--tree | --ast]
tepl check <file-or-project>
tepl generate <file-or-project> --out <directory> [options]
```

| Command | Purpose | Input |
| --- | --- | --- |
| `init` | Create a minimal project with a dialect and rewrite rule. | A destination directory. |
| `parse` | Parse source and load its imports. | One source file. |
| `check` | Perform semantic analysis and print the checked intermediate representation. | One source file or a project directory. |
| `generate` | Check the input, then generate code and synchronize the output directory. | One source file or a project directory. |

Use `-h` or `--help` for top-level or command-specific help:

```sh
tepl --help
tepl init --help
tepl parse --help
tepl check --help
tepl generate --help
```

## `init`

```sh
tepl init my_project
tepl check my_project
```

The starter project contains:

```text
my_project/
├── dialects/
│   └── your_dialect.tepl
└── rules/
    └── your_rule.tepl
```

`your_dialect.tepl` declares `YourDialect.add`. `your_rule.tepl` imports it and
defines `commute_add`. The starter can be checked and generated immediately.
Add a `graphs/` directory when you want concrete starting graphs.

The destination can be new or an existing directory. `tepl init .` initializes
the current directory. Existing unrelated files are preserved. Initialization
refuses to overwrite either starter file, including a directory or symlink at
the same filename.

| Argument | Required | Meaning |
| --- | --- | --- |
| `<directory>` | Yes | Destination for the `dialects/` and `rules/` starter files. |

## File and project inputs

`check` and `generate` accept a file or a project directory. A project uses:

```text
my_project/
├── dialects/     # Operation and attribute declarations
├── rules/        # Concrete and abstract rewrite rules
└── graphs/       # Concrete computation graphs
```

TEPL recursively discovers `.tepl` files beneath these three directories.
Nested directories are supported. A directory can omit any of the three parts,
but it must contain at least one `.tepl` file under them. Files elsewhere in
the project root are not automatically discovered.

Imports are resolved relative to the file containing each import, independently
of your shell's working directory. Imported dialects and selected abstract rules
are loaded as needed. `parse` accepts only a file and does not perform project
directory discovery.

## `parse`

```sh
tepl parse examples/sample/rules/basic.tepl
tepl parse examples/sample/rules/basic.tepl --tree
tepl parse examples/sample/rules/basic.tepl --ast
```

| Option | Meaning |
| --- | --- |
| No output option | Print a summary of the parsed rules and, when present, graphs. |
| `--tree` | Print the ANTLR parse tree for the input file. |
| `--ast` | Print the structured TEPL AST after loading imports. |

`--tree` and `--ast` are mutually exclusive. Parsing reports syntax and import
errors, but leaves semantic checks such as operation resolution, arity, and
expression typing to `check`.

These tree and AST outputs are diagnostic representations, not formatted TEPL
source.

## `check`

```sh
tepl check examples/sample
tepl check examples/sample/graphs/bindings.tepl
```

The checker resolves operation visibility, expands rule inheritance, and validates
signatures, attribute schemas, expressions, shape and dtype programs, and graph
references. On success, it prints the checked IR, beginning with `CheckedProgram`.

`check` does not write generated code or run an optimizer. Shape and dtype
programs are type-checked here; evaluating them for concrete tensor metadata is
the generated runtime's responsibility. Numerical rewrite legality also depends
on the dialect and application semantics.

The command takes one input and has no additional options beyond help.

## `generate`

### Generate Rust

```sh
tepl generate examples/sample --out my_app/src/generated
```

Rust is the default target. The destination is the generated module directory,
containing `mod.rs`, operation types, dialect modules, rule builders, graph
builders, metadata analysis, and runtime helpers. Generation does not create
a Cargo application or run it. See the [Rust tutorial](../tutorial/rust/02_run.md)
for application setup.

### Generate C++

```sh
tepl generate examples/sample --target cpp --out my_cpp_app/generated
```

The destination contains generated headers and the `generated.h` entry point.
To give an independently generated project its own namespace:

```sh
tepl generate examples/sample --target cpp \
  --cpp-namespace my_app::tensor_rules \
  --out my_cpp_app/generated
```

Namespace components must be valid, nonreserved C++ identifiers. The default
namespace is `tepl_generated`; `std` and namespaces beneath it are rejected.
This option applies to C++ generation.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `-o <directory>`, `--out <directory>` | Required | Destination for the generated module or headers. |
| `--target <target>` | `rust` | Select `rust` or `cpp`. The accepted value `python` currently reports that generation is not implemented. |
| `--check` | Off | Compare the expected generated files with the destination, without updating it. |
| `--format` | On | Format generated Rust with `rustfmt`, or C++ headers with `clang-format`. |
| `--no-format` | Off | Disable generated-output formatting. |
| `--cpp-namespace <name>` | `tepl_generated` | Set the generated C++ namespace. Nested names such as `my_app::tensor_rules` are supported. |

All generation starts with parsing, import loading, and semantic checking. An
invalid input is rejected before output synchronization.

### Formatting

Formatting is enabled by default. Rust generation needs `rustfmt` on `PATH` and
uses edition 2024. C++ generation needs `clang-format` and uses the Google style.
A missing formatter or formatter failure stops generation with an error.

To generate without either formatter:

```sh
tepl generate examples/sample --out my_app/src/generated --no-format
tepl generate examples/sample --target cpp \
  --out my_cpp_app/generated --no-format
```

These flags format generated code, not the input `.tepl` files. The Docker image
omits the formatters, so pass `--no-format` when generating inside it. See the
[Docker instructions](../tutorial/rust/01_build.md#build-with-docker).

### Check generated files

Use the same input, target, namespace, and formatting settings as the command
that produced the output:

```sh
tepl generate examples/sample --out my_app/src/generated --check

# For output originally generated with --no-format:
tepl generate examples/sample --out my_app/src/generated --no-format --check
```

`--check` regenerates the expected output and, when enabled, formats it in a
temporary directory. It compares the results with the destination without
changing the destination. It still needs the selected formatter unless
`--no-format` is supplied.

A current output prints `Generated output is current in ...` and exits with `0`.
Missing, changed, or obsolete files are reported on stderr as
`generated output differs: ...`, and the command exits with `1`. An absent or
different generated-file manifest also counts as a difference. This check compares
generated files; it does not compile or test the resulting application.

### Generated-file ownership

The destination contains `.tepl-generated-files`, a manifest of generated paths.
A normal generation writes the current generated files, updates the manifest,
and removes obsolete files listed in the previous manifest. Unrelated files are
preserved. Handwritten application code should use separate paths from generated
files, since generated paths are overwritten on regeneration.

Project generation preserves nested rule and graph source paths as module paths.
When generating from one file, rule and graph modules use source filename stems.
The backend reports names or paths that cannot be represented without collisions.

## Output, diagnostics, and exit codes

Help, initialization messages, parse results, checked IR, and successful
generation messages go to stdout. Diagnostics and generated-file differences
go to stderr.

Source diagnostics normally include `file:line:column: message`. Inherited-rule
errors can also include notes showing the instantiation and related declarations.

| Exit code | Meaning |
| --- | --- |
| `0` | Success, help displayed, or generated output is current. |
| `1` | Syntax, import or project-loading diagnostics, semantic errors, or backend generation errors; also generated output differs under `generate --check`. |
| `2` | Invalid command-line arguments, top-level file access or filesystem failure, initialization failure, or output/formatter failure. |

For scripting, check the exit code rather than parsing the human-readable IR or
success messages. `tepl check` validates source; `tepl generate --check` verifies
that generated artifacts are current.

See the [CLI implementation](../../src/main.cc) for option definitions and the
[builtin reference](built-ins.md) for functions available in TEPL expressions.
