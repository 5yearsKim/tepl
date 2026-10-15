# TEPL command-line reference

The `tepl` command initializes projects, parses source files, checks declarations,
compiles portable JSON, runs greedy rewrites, and generates Rust or C++ code. See the [build guide](../tutorial/rust/01_build.md)
to build the executable. Examples below assume `tepl` is on your `PATH`; from
the repository root, you can use `bazel-bin/tepl` instead.

## Commands

```text
tepl init <directory>
tepl parse <file> [--tree | --ast]
tepl check <file-or-project>
tepl compile <file-or-project> [--rules file-or-directory] [-o program.json]
tepl validate <program.json>
tepl schema [-o rewrite-program.schema.json]
tepl rewrite <file-or-project-or-json> [--rules file-or-directory] [--rule name] [--graph name] [-o result.tepl]
tepl generate <file-or-project-or-json> --out <directory> [options]
```

| Command | Purpose | Input |
| --- | --- | --- |
| `init` | Create a minimal project with a dialect and rewrite rule. | A destination directory. |
| `parse` | Parse source and load its imports. | One source file. |
| `check` | Perform semantic analysis and print the checked intermediate representation. | One source file or a project directory. |
| `compile` | Check and compile an executable JSON program. | Source file or project directory. |
| `validate` | Validate compiled JSON structure and IR references/types. | Compiled JSON file. |
| `schema` | Export the compiled program JSON Schema. | No input. |
| `rewrite` | Apply rules greedily and export a TEPL graph. | Source file, project, or compiled JSON. |
| `generate` | Generate code and synchronize the output directory. | Source file, project, or compiled JSON. |

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

`check`, `compile`, `rewrite`, and `generate` accept a file or a project directory. A project uses:

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
the greedy engine or generated runtime's responsibility. Numerical rewrite legality also depends
on the dialect and application semantics.

For operations with axis semantics, the checker also constructs RHS `auto`
attributes and proves rewrite correctness under explicit conditions:

```sh
tepl check examples/semantic/rules/algebraic.tepl \
  --semantic-policy=algebraic --rule associate
```

| Option | Meaning |
| --- | --- |
| `--semantic-policy strict` | Default arithmetic policy; algebraic transformations require permission. |
| `--semantic-policy algebraic` | Permit the supported exact-arithmetic algebraic laws. |
| `--rule <name>` | Check and report one semantic rule in the input. |

Without `--rule`, all semantic rules are checked. Missing conditions or
unsupported proofs are errors. The previous `--index-rule`, `--verify-small`,
and `--solver-timeout-ms` options are retired. See the
[semantic examples](../../examples/semantic/README.md) for the supported fragment.
Axis-semantic code generation is not yet implemented.

## `compile`, `validate`, and `schema`

```sh
tepl compile examples/rewrite -o /tmp/program.json
tepl validate /tmp/program.json
tepl schema -o /tmp/rewrite-program.schema.json
```

`compile` checks source and lowers rules to the versioned executable format.
Use `--rules <file-or-directory>` to load rule source separately from the graph.
Directory inputs are scanned recursively for `.tepl` files in sorted path order;
imports remain relative to each file.
`--semantic-policy strict|algebraic` selects the existing proof policy for source
rules with axis semantics. `validate` checks the compiled schema and IR invariants;
it does not rerun numerical proofs. `schema` exports the exact schema embedded
in the loader. `compile` and `schema` write to stdout when `-o` is omitted.

## `rewrite`

```sh
tepl rewrite /tmp/program.json --trace -o /tmp/result.tepl
tepl rewrite examples/rewrite/graphs/transpose.tepl \
  --rules examples/rewrite/rules --rule compose_transposes --trace
tepl rewrite examples/rewrite/auto.tepl --max-steps 100
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-o`, `--out` | stdout | Output TEPL graph file. |
| `--rules <file-or-directory>` | Input's rules | Load additional source rules; with a project input, replace its default `rules/` directory selection. Requires source input. |
| `--rule <name>` | All loaded rules | Execute only the named rule. Also supported for compiled JSON. |
| `--graph <name>` | Only graph | Select a graph; ambiguous selection is an error. |
| `--max-steps <n>` | 1000 | Maximum committed rewrites. |
| `--max-nodes <n>` | 1000000 | Maximum graph nodes, including temporary replacements. |
| `--trace` | Off | Report each applied rule on stderr. |
| `--semantic-policy strict\|algebraic` | strict | Proof policy when compiling source; JSON carries its compiled policy. |

Graph files import dialects independently of rules. For separate `dialects/`,
`rules/`, and `graphs/` directories, pass the graph file (or graph directory) as
the positional input and select rule sources with `--rules`. Every loaded rule
is checked; `--rule` selects execution. Unknown or ambiguous names report an
error. Compiled JSON already contains its rules and rejects `--rules`.

The engine checks nodes in ID order and rules in compiled order, applies the
first effective candidate, then restarts. Output contains shared `let` bindings,
concrete attributes, and dialect imports. Statistics and
`Rewrite completed. Result saved to <path>.` go to stderr. `--trace` reports
`Successfully applied rule '<name>'.` for each application. A limit stops
execution with exit code `3` and exports the current graph.

See [greedy rewriting](../advanced/greedy_rewriting.md) for the compiled format,
C++ API, metadata checks, and export limitations.

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
See the [C++ tutorial](../tutorial/cpp/02_run.md) for application setup with egg-c.

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

Source generation starts with parsing, import loading, semantic checking, and
compilation. A `.json` input loads the same compiled program directly. Both paths
reject invalid or unsupported input before output synchronization. Extended
rule expressions and axis semantics still require future backend lowering.

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
[Docker instructions](../tutorial/rust/01_build.md#option-2-build-with-docker).

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
| `3` | Greedy rewriting stopped at a resource/step limit or cancellation; the current graph was exported. |
| `2` | Invalid command-line arguments, top-level file access or filesystem failure, initialization failure, or output/formatter failure. |

For scripting, check the exit code rather than parsing the human-readable IR or
success messages. Compiled validation and rewrite failures report diagnostics
and exit with `1`. `tepl check` validates source; `tepl generate --check` verifies
that generated artifacts are current.

See the [CLI implementation](../../src/main.cc) for option definitions and the
[builtin reference](built-ins.md) for functions available in TEPL expressions.
