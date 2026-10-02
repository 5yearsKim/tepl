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
bazel run //:tepl -- parse "$PWD/examples/dialects/tensor.tepl" --ast
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

## Dialects and imports

The [tensor dialect](examples/dialects/tensor.tepl) declares the operations in
the experimental Rust IR. [lora.tepl](examples/lora.tepl) imports it and selects
the operations used by its rule:

```tepl
from "dialects/tensor.tepl" import TensorLang as t;
use t::{add, dot};
```

`use t;` makes all operations in `TensorLang` available by their bare names.
Both forms also allow qualified calls such as `(t.dot[@d] X W)`. For a named
import without a `use` statement, qualified calls still work, while bare names
stay out of scope. `as t` is optional; without it, use the name `TensorLang`. The older
`import "dialects/tensor.tepl";` syntax remains available and opens all
operations. Names imported from different dialects must be unambiguous.
Paths are relative to the importing file; imports may be nested. The CLI
rejects missing, invalid, or cyclic imports. `--ast` shows imported dialect
declarations as well as rules.

```text
dialect TensorLang {
    attrs CollectiveReduce { kind: string; }
    op add(lhs: tensor, rhs: tensor) -> tensor;
    op multiply(lhs: tensor, rhs: tensor) -> tensor { alias: mul; }
    op all_reduce(input: tensor) -> tensor {
        attrs: CollectiveReduce;
    }
    op concatenate(first: tensor, second: tensor, rest: tensor...) -> tensor {
        attrs { axis: index; }
    }
}
```

Operands are named. Fixed operands determine exact arity; a final `...`
operand permits zero or more additional operands. An `attrs` block defines
operation attributes, and `attrs: Name;` reuses a schema. Attribute fields
currently support `index`, `string`, and list syntax such as `index[]`.
An operation can declare at most one `alias` and one `attrs` property, in
either order. Inline and shared attributes are alternative forms of the same
property. `alias` gives a second spelling to the same operation; for example,
`dot_general` remains the declared name and `dot` refers to it. In files with
a dialect, the CLI checks rule operation names, arity, and whether an attribute
descriptor is required.

The parser and AST now carry this information, but Rust IR generation and
full attribute type checking are future compiler stages. The tuple and binder
examples remain syntax fixtures and include operators outside the Rust tensor
IR.

## Current rule syntax

```text
rule NAME {
    zero or more tensor or scalar declarations
    LHS => RHS
    optional where { CONDITION; ... }
    optional derive { @NAME = EXPRESSION; ... }
}
```

- A file contains imports, `use` statements, dialects, and/or rules. Whitespace
  and newlines are insignificant.
- `//` line comments and `/* ... */` block comments are supported.
- Declarations precede the rewrite and have no semicolon. Tensor declarations
  use brackets (`X: [Batch..., M, K]`); scalar declarations use `S: scalar`.
  A tensor shape may be empty (`[]`), contain named dimensions or `_`, and have
  at most one named or anonymous `...` segment anywhere in the list. Shape
  arithmetic belongs in `where`.
- Graph expressions are names, operator applications, or bindings: `X`, `Y`,
  `(dot[@d] X W)`, `let Y = (dot X W)`, and
  `(let Y = (dot X W))`. Later `Y` references the bound tensor value.
  Bindings are allowed only on the LHS, where they capture matched values.
  RHS expressions contain operations, literals, and LHS capture references;
  they cannot introduce `let` bindings.
  Operators may have zero or more operands.
- Graph operands and roots also accept integer and decimal literals:
  `(add X 1)`, `(add X 1.0)`, and `(mul X -0.5)`. An optional `+` or `-`
  sign is allowed; decimals require digits on both sides of the point.
  Exponents, numeric suffixes, `.5`, and `1.` are currently unsupported.
  Literals preserve their kind and spelling without numeric conversion:
  `1`, `1.0`, and `1.00` are distinct for structural matching. Tensor
  semantics treat graph literals as rank-zero values; element types and
  broadcasting remain host responsibilities.
- Tuples use the ordinary operator syntax `(tuple X Y)`; projection uses
  `(get[0] T)` with exactly one operand and a nonnegative integer index.
- `where` precedes `derive` when both occur. Statements within either section
  require semicolons. Empty sections are allowed.
- Constraint expressions support host calls, names, descriptor references,
  integers, decimals, booleans, and parentheses. Precedence, highest first:
  unary `! + -`, multiplicative `* / %`, additive `+ -`, comparison
  `< <= > >=`, equality `== !=`, logical `&&`, logical `||`.
  Arithmetic and logical operators associate to the left; comparisons and
  equality cannot be chained at their respective precedence levels.
- Identifiers use ASCII letters, digits, and underscores, and cannot start with
  a digit. `rule`, `abstract`, `extends`, `fn`, `from`, `import`, `as`, `use`,
  `dialect`, `op`, `attrs`,
  `alias`, `let`, `where`, `derive`, `scalar`, `get`, `true`, `false`, and `_` are
  reserved. `alias`
  remains valid as an operation name.

See [examples](examples/) for simple rules, shape patterns, binders, numeric literals, tuples,
and the complete LoRA example from the design document.

## Abstract and inherited rules

[abstract.tepl](examples/abstract.tepl) declares reusable graph patterns with
operation (`op`) and host-function (`fn`) parameters:

```tepl
abstract rule commute(F: op<(tensor, tensor) -> tensor>) {
    (F X Y) => (F Y X)
}
```

[inherited.tepl](examples/inherited.tepl) imports templates and binds their
parameters by name:

```tepl
from "abstract.tepl" import {commute};
from "dialects/tensor.tepl" import TensorLang as t;
rule commute_add extends commute(F = t.add);
rule commute_small_vectors extends commute(F = t.add) {
    X: [N]
    Y: [N]
    where { N <= 1024; }
}
```

An inherited rule ends with `;` or a body containing shape declarations and
an optional `where` section. It cannot supply a replacement graph or `derive`
section. Abstract rules have the same rewrite body as ordinary rules.
Parameter lists, signature operand lists, and instance binding lists may be
empty; nonempty lists require commas and do not allow a trailing comma.
Signature types preserve named types (including `scalar`) for later validation.
Bindings are bare or dialect-qualified names.

The AST preserves signatures, bindings, restrictions, and source spans.
`resolveImports` loads selected abstract definitions into `Program::imported_rules`
and keeps root rules in `Program::rules`. Existing dialect import forms still
require files that contain only dialect declarations and imports. Both examples
can be inspected with `parse --ast`.

Inheritance remains unexpanded at this stage: inherited rules have null local
`lhs` and `rhs` pointers. Base-rule resolution, parameter signature checking,
substitution, and merging inherited conditions belong to a later semantic pass.
Dialect graph validation defers abstract and inherited rules until that pass.
Simple rewriting and host-template generation report that expansion is required
when given these rules.

## Scope and next steps

The parser also builds an owning AST with source spans for valid input.
A first structural rewriter is available through `validateSimpleRule` and
`rewriteOnce` in `src/semantic.h` and `src/simple_rewrite.h`. It matches a rule
only at the input root, captures LHS variables, and substitutes them into the
RHS. Repeated variables must match structurally equal subtrees. This first
version supports graph variables, numeric literals, and operators without
attributes. Literal matching compares kind and exact spelling. Validation
reports unsupported declarations, binders, projections, `where`, and `derive`
instead of silently ignoring them. A mismatch returns `std::nullopt`; an
invalid rule throws `std::invalid_argument` from `rewriteOnce`.

Full symbol resolution, operator operand types, general
host-function type checking, legality, descriptor references, and
e-graph behavior need later semantic passes.
The examples are parser fixtures, not claims of tensor equivalence.
Multiple-root patterns and variadic expression operands are deferred.

The grammar contains no C++ actions. Bazel generates lexer/parser and visitor
sources under the build directory. `AstBuilder` converts their parse tree into
the project AST. The next step is broader symbol and metadata dependency
validation.

AST types, printing, and construction live in `src/ast/`. The builder uses one
visitor with separate rule and dialect source files. Parsing and semantic
validation remain in `src/`.

The `host-template` command emits a Rust host-function interface from calls in
`where` and `derive`. Use `--impl` for a separate implementation template.
See [the egg lab](labs/rust-egg/README.md) for the runtime and an integration
test. Full rule lowering remains future work.

`derive` host functions return `Option<OpAttrs>` containing only operation
descriptors. Both `where` and `derive` use matched LHS inputs; derivations are
evaluated in source order. LoRA uses `infer_lora_out(X, A, B, @outer, @inner)`
for its final descriptor, avoiding any dependency on a constructed RHS value.
The runtime validates the entire RHS tree before insertion. Output metadata is
inferred separately by the host's e-class analysis after insertion. Ordinary
LHS graph variables are valid tensor host arguments without shape declarations.

The host template maps decimal arguments to Rust `f64`, unsigned integer
arguments to `usize`, and negative integer arguments to `i64`. These are host
interface types; graph literals retain their spelling and defer tensor element
types to the host.
