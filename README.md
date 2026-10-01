# TEPL

An ANTLR4 grammar and C++ parser for the tensor rewrite DSL described in
[initial_design.md](initial_design.md).

## Build and try a rule

Install [Bazelisk](https://github.com/bazelbuild/bazelisk) and a C++20 compiler.
The project pins Bazel in `.bazelversion`. Bazel downloads the ANTLR tool,
C++ runtime, and Java runtime on the first build; no separate ANTLR installation
is needed. Both ANTLR dependencies are pinned to 4.13.2 with SHA-256 checksums.

```sh
bazel build //:tepl
bazel test //...
bazel run //:tepl -- --help
bazel run //:tepl -- parse "$PWD/examples/lora.tepl"
bazel run //:tepl -- parse "$PWD/examples/lora.tepl" --tree
bazel run //:tepl -- parse "$PWD/examples/lora.tepl" --ast
```

Use an absolute input path with `bazel run`, which starts the executable from its
runfiles directory. `--tree` prints the ANTLR parse tree; `--ast` prints an
indented AST with grammar wrappers removed. Successful parses return
0; syntax errors return 1 with `file:line:column` diagnostics; usage and file errors
return 2. Lines and columns start at 1. CLI11 handles argument parsing and provides
`-h`/`--help` for the program and `parse` command. `--tree` and `--ast` may each
appear before or after the input path, but cannot be used together. Help exits
with 0.

## Format C++ code

Install `clang-format` and use either the Bazel targets or the shell script:

```sh
bazel run //:format
bazel run //:format_check
# Equivalent commands without Bazel:
./tools/format.sh
./tools/format.sh --check
```

The shared `.clang-format` uses Google's C++ style. Formatting covers C++ sources
and headers under `src/` and `tests/`. Check mode reports violations and exits
nonzero without modifying files. Generated and third-party code are excluded.

These targets wrap the shell script and use the formatter installed on your
machine. To select a specific executable, use
`CLANG_FORMAT=clang-format-21 bazel run //:format`. For consistent formatting in
CI and across developers, install the same clang-format version; the Bazel
wrapper does not pin or download it.

## Current syntax

```text
rule NAME {
    zero or more tensor or scalar declarations
    LHS => RHS
    optional where { CONDITION; ... }
    optional derive { @NAME = EXPRESSION; ... }
}
```

- A file contains one or more rules. Whitespace and newlines are insignificant.
- `//` line comments and `/* ... */` block comments are supported.
- Declarations precede the rewrite and have no semicolon. Tensor declarations
  use brackets (`X: [Batch..., M, K]`); scalar declarations use `S: scalar`.
  A tensor shape may be empty (`[]`), contain named dimensions or `_`, and have
  at most one named or anonymous `...` segment anywhere in the list. Shape
  arithmetic belongs in `where`.
- Graph expressions are variables, binder references, operator applications,
  or bindings: `X`, `?Y`, `(dot[@d] X W)`, `?Y = (dot X W)`,
  and `(?Y = (dot X W))`. Operators may have zero or more operands.
- Tuples use the ordinary operator syntax `(tuple X Y)`; projection uses
  `(get[0] T)` with exactly one operand and a nonnegative integer index.
- `where` precedes `derive` when both occur. Statements within either section
  require semicolons. Empty sections are allowed.
- Constraint expressions support host calls, names, descriptor/binder references,
  integers, booleans, and parentheses. Precedence, highest first:
  unary `! + -`, multiplicative `* / %`, additive `+ -`, comparison
  `< <= > >=`, equality `== !=`, logical `&&`, logical `||`.
  Arithmetic and logical operators associate to the left; comparisons and
  equality cannot be chained at their respective precedence levels.
- Identifiers use ASCII letters, digits, and underscores, and cannot start with
  a digit. `rule`, `where`, `derive`, `scalar`, `get`, `true`, `false`, and `_`
  are reserved.

See [examples](examples/) for simple rules, shape patterns, binders, tuples,
and the complete LoRA example from the design document.

## Scope and next steps

The parser also builds an owning AST with source spans for valid input.
A first structural rewriter is available through `validateSimpleRule` and
`rewriteOnce` in `src/semantic.h` and `src/simple_rewrite.h`. It matches a rule
only at the input root, captures LHS variables, and substitutes them into the
RHS. Repeated variables must match structurally equal subtrees. This first
version supports graph variables and operators without attributes. Validation
reports unsupported declarations, binders, projections, `where`, and `derive`
instead of silently ignoring them. A mismatch returns `std::nullopt`; an
invalid rule throws `std::invalid_argument` from `rewriteOnce`.

Full symbol resolution, operator arity/types (except `get` syntax), general
host-function type checking, legality, derived metadata dependencies, and
e-graph behavior need later semantic passes.
The examples are parser fixtures, not claims of tensor equivalence.
Multiple-root patterns and variadic expression operands are deferred.

The grammar contains no C++ actions. Bazel generates lexer/parser and visitor
sources under the build directory. `AstBuilder` converts their parse tree into
the project AST. The next step is broader symbol and metadata dependency
validation.

The `host-template` command emits a Rust host-function interface from calls in
`where` and `derive`. Use `--impl` for a separate implementation template.
See [the egg lab](labs/rust-egg/README.md) for the runtime and an integration
test. Full rule lowering remains future work.
