# TEPL command-line reference

The `tepl` command initializes projects, parses source files, checks declarations,
compiles portable JSON, runs greedy rewrites, and generates Rust for egg. C++ generation is deferred. See the [build guide](../tutorial/rust/01_build.md)
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

`check --suggest-where --semantic-policy=algebraic` searches for a verified
sufficient source restriction when a rule is unproved. Select a rule with
`--rule NAME`. Suggestions preserve the original conditions and derivations;
review and add them to source before compiling. The original unproved rule
still exits with status 1. Search is bounded and does not promise a weakest
condition or a suggestion for every rule.

```sh
tepl check examples/semantic/restriction_suggestions.tepl \
  --semantic-policy=algebraic --suggest-where --rule remove_output_permutation
tepl compile examples/semantic/explicit_axis_conditions.tepl \
  --semantic-policy=algebraic -o verified.json
tepl rewrite verified.json --graph feasible --trace
```

`compile` and `rewrite` require proof under the source's written conditions.
The old `--infer-applicability` condition-insertion mode has been removed.
Runtime feasibility programs from that mode are rejected when loading JSON.

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
├── tepl.toml
├── dialects/
│   └── your_dialect.tepl
└── rules/
    └── your_rule.tepl
```

`your_dialect.tepl` declares `YourDialect.add`. `your_rule.tepl` imports it and
defines `commute_add`. The starter can be checked and compiled immediately.
Add a `graphs/` directory when you want concrete starting graphs.

`tepl.toml` identifies the project root and records the supported TEPL version:

```toml
tepl-version = "0.1.0"
```

The starter rule uses a project-root import:

```tepl
from "dialects/your_dialect.tepl" import YourDialect as d;
```

The destination can be new or an existing directory. `tepl init .` initializes
the current directory. Existing unrelated files are preserved. Initialization
refuses to overwrite any of the three starter files, including a directory or
symlink at the same filename.

| Argument | Required | Meaning |
| --- | --- | --- |
| `<directory>` | Yes | Destination for `tepl.toml` and the `dialects/` and `rules/` starter files. |

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

Imported dialects and selected abstract rules are loaded as needed. `parse`
accepts only a file and does not scan the project directories for source files.

### Project configuration and import paths

For each source file, TEPL searches its containing directory and then its
ancestors for the nearest `tepl.toml`. That config's directory is the project
root, independently of your shell's working directory. Nested projects can have
their own config; imports in each file use that file's nearest config.

| Import path | Resolution |
| --- | --- |
| `"dialects/tensor.tepl"` | Relative to the project root when a config is present. |
| `"./helpers.tepl"` | Relative to the importing file's directory. |
| `"../dialects/tensor.tepl"` | Relative to the importing file's directory. |
| `"/absolute/path/tensor.tepl"` | Filesystem absolute path. |

For example, `examples/semantic/tepl.toml` lets
`examples/semantic/rules/basic.tepl` use this import from any nesting depth:

```tepl
from "dialects/tensor.tepl" import TensorLang as t;
```

An empty `tepl.toml` is a valid root marker. The optional `tepl-version` field
must be a string equal to the supported version, currently `"0.1.0"`; version
ranges are not supported. Invalid TOML, invalid version types, unsupported
versions, and unreadable configs report diagnostics.

Without a config, bare paths retain their previous file-relative behavior.
When adding a config to an existing project, change local imports such as
`"abstract.tepl"` to `"./abstract.tepl"`. Configured imports do not fall back to
the importing directory if the project-root target is missing.

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

For operations with axis semantics, the checker also constructs inferred RHS
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
imports follow the project-root and file-relative rules above.
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

Generate Rust modules from TEPL source, a project directory, or compiled JSON:

```sh
tepl generate examples/rewrite --out my_app/src/generated
tepl compile examples/rewrite --out /tmp/program.json
tepl generate /tmp/program.json --out my_app/src/generated
```

The destination contains program-specific operation types, dialects, metadata
functions, rules, and graph builders. Add the shared `tepl-runtime` crate to the
application. Generation does not create a Cargo application or run it. See the
[Rust tutorial](../tutorial/rust/02_run.md).

| Option | Default | Meaning |
| --- | --- | --- |
| `-o`, `--out` | Required | Generated module directory. |
| `--target` | `rust` | Rust is implemented. C++ and Python report a diagnostic. |
| `--rules` | None | Additional rule source file or directory; source input only. |
| `--rules-root` | Inferred | Root for rule module paths. |
| `--graphs-root` | Inferred | Root for graph module paths. |
| `--check` | Off | Compare generated output without updating files. |
| `--format` | On | Format generated Rust with `rustfmt`, using edition 2024. |
| `--no-format` | Off | Skip generated-output formatting. |

Source input is compiled to `ExecutableProgram`; JSON input is validated and
loaded into the same representation. Completed conditions, attribute constructors,
and axis derivations are translated into Rust functions. No source files are
needed when generating from compiled JSON.

By default, source paths beneath `rules/` and `graphs/` retain their nested module
paths. Other source paths use filename stems. Explicit roots override this
convention and must contain the corresponding sources. Use the same root options
when comparing source and JSON output.

Formatting finishes before output synchronization. If `rustfmt` is unavailable,
use `--no-format`. Formatting applies to generated Rust, not TEPL source.

### Check generated files

Use the same input, module roots, and formatting settings as the command
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

Current output exits with `0`.
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

Generation preserves nested source paths under `rules/` and `graphs/`, including
when those paths come from compiled JSON.
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
| `1` | Syntax, import or project-loading diagnostics, semantic errors, or backend generation errors; also output synchronization failures or differences under `generate --check`. |
| `3` | Greedy rewriting stopped at a resource/step limit or cancellation; the current graph was exported. |
| `2` | Invalid command-line arguments, top-level file access or filesystem failure, initialization failure. |

For scripting, check the exit code rather than parsing the human-readable IR or
success messages. Compiled validation and rewrite failures report diagnostics
and exit with `1`. `tepl check` validates source; `tepl generate --check` verifies
that generated artifacts are current.

See the [CLI implementation](../../src/cli/main.cc) for option definitions and the
[builtin reference](built-ins.md) for functions available in TEPL expressions.
