# TEPL command-line reference

The `tepl` command initializes projects, parses source files, checks declarations,
compiles portable JSON, runs greedy rewrites, and generates Rust for egg and C++20 for egg-c. See the [build guide](../tutorial/rust/01_build.md)
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

Applicability combines the written `where` condition with compiled feasibility
checks. False feasibility skips a match before constructing RHS attributes.
`--core` selects the metadata DAG constructor, including supported schema
extensions. Generated C++/Rust currently require the symbolic expression path;
unsupported DAG exports report an error. Compiled artifacts use format version 4.

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
tepl --version
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

`tepl.toml` identifies the project root, records the supported TEPL version,
and configures source discovery and checking. `init` writes these defaults:

```toml
tepl-version = "0.2.0"

[sources]
dialects = ["dialects"]
rules = ["rules"]
graphs = ["graphs"]
exclude = []

[check]
semantic-policy = "strict"
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

By default, TEPL recursively discovers `.tepl` files beneath these three
directories. `[sources]` can replace each category with a list of directories.
Nested directories are supported; discovered paths are sorted and deduplicated.
Missing directories are skipped, but at least one source file must remain after
exclusions. Files outside the selected directories are not automatically discovered.

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
must be a string equal to the supported version, currently `"0.2.0"`; version
ranges are not supported. Invalid TOML, invalid version types, unsupported
versions, unreadable configs, unknown keys, and invalid configuration values
report diagnostics. `tepl --version` prints the compiler version.

All sections and individual fields are optional and use the defaults shown
above. Each `[sources]` field is an array of nonempty strings. An empty source
directory list disables discovery for that category; `exclude = []` excludes
nothing. Relative source directories are resolved
against the directory containing `tepl.toml`; absolute directories are also
accepted. Changing source directories does not change import path resolution.

For example, use custom rule directories and skip experiments and fixtures:

```toml
[sources]
rules = ["rules", "shared/rules"]
exclude = ["rules/experimental/**", "**/fixtures/**"]
```

Exclusions match complete project-relative file paths with `/` separators.
`*` matches within one path component, `?` matches one non-separator character,
and `**` matches across directories. `**/` also matches zero directories, so
`**/fixtures/**` covers both root and nested fixtures. Other characters are
literal. Exclusions apply to directory discovery, including directories selected
with `--rules`; explicitly selected files and imported dependencies still load.
With `--rules`, a project input uses its configured dialect and graph directories
and the explicitly selected rule sources replace its configured rule directories.

`[check].semantic-policy` accepts `"strict"` (default) or `"algebraic"`. For CLI
source commands, the primary input's nearest config provides the default policy
for `check`, `compile`, `rewrite`, `generate`, and `core-emit`. Imported projects
retain their own import roots but do not override this policy. An explicit
`--semantic-policy` flag overrides the configuration, including an explicit
`--semantic-policy strict`. Compiled JSON retains its compiled policy.

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
| `--semantic-policy strict` | Require the strict arithmetic policy, overriding project configuration. |
| `--semantic-policy algebraic` | Permit the supported exact-arithmetic algebraic laws. |
| `--rule <name>` | Check and report one semantic rule in the input. |

Without `--rule`, all semantic rules are checked. Missing conditions or
unsupported proofs are errors. The previous `--index-rule`, `--verify-small`,
and `--solver-timeout-ms` options are retired. See the
[semantic examples](../../examples/semantic/README.md) for the supported fragment.
Both backends generate the completed axis metadata computations.

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
| `--semantic-policy strict\|algebraic` | Config or strict | Proof policy when compiling source; defaults to project configuration, then strict. JSON carries its compiled policy. |

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

Generate Rust modules or C++ headers from TEPL source, a project directory, or compiled JSON:

```sh
tepl generate examples/rewrite --out my_app/src/generated
tepl compile examples/rewrite --out /tmp/program.json
tepl generate /tmp/program.json --out my_app/src/generated
```

The destination contains program-specific operation types, dialects, metadata
functions, rules, and graph builders. Add the shared `tepl-runtime` crate to the
application. Generation does not create a Cargo application or run it. See the
[Rust tutorial](../tutorial/rust/02_run.md).

For C++, use `--target cpp --cpp-namespace my_ir --out my_app/generated` and add
the separate [C++ runtime](../../runtime/cpp/README.md) and egg-c dependencies.
The generated `generated.h` includes the complete program API.

| Option | Default | Meaning |
| --- | --- | --- |
| `-o`, `--out` | Required | Generated module directory. |
| `--target` | `rust` | `rust` and `cpp` are implemented; Python reports a diagnostic. |
| `--cpp-namespace` | `tepl_generated` | Namespace for generated C++ declarations. |
| `--rules` | None | Additional rule source file or directory; source input only. |
| `--rules-root` | Inferred | Root for rule module paths. |
| `--graphs-root` | Inferred | Root for graph module paths. |
| `--check` | Off | Compare generated output without updating files. |
| `--format` | On | Use `rustfmt` (edition 2024) for Rust or `clang-format` for C++. |
| `--no-format` | Off | Skip generated-output formatting. |

Source input is compiled to `ExecutableProgram`; JSON input is validated and
loaded into the same representation. Completed conditions, attribute constructors,
and axis derivations are translated into target-language functions. No source files are
needed when generating from compiled JSON.

By default, source paths beneath `rules/` and `graphs/` retain their nested module
paths. Other source paths use filename stems. Explicit roots override this
convention and must contain the corresponding sources. Use the same root options
when comparing source and JSON output.

Formatting finishes before output synchronization. If the selected formatter is unavailable,
use `--no-format`. Formatting applies to generated source, not TEPL source.

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
