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
bazel run //:tepl -- check "$PWD/examples/lora.tepl"
bazel run //:tepl -- check "$PWD/examples/inherited.tepl"
```

Use an absolute input path with `bazel run`, which starts the executable from its
runfiles directory. `--tree` prints the ANTLR parse tree; `--ast` prints an
indented AST with grammar wrappers removed. `parse` checks syntax and loads
imports, preserving source annotations and unresolved operation names in the AST.
Use `check` for semantic validation through core. Successful parses return 0;
syntax and import errors return 1 with `file:line:column` diagnostics; usage and
input-file errors return 2. Lines and columns start at 1. CLI11 handles argument
parsing and provides `-h`/`--help` for the program and both commands. `--tree` and
`--ast` may each appear before or after the input path, but cannot be used
together. Help exits with 0.

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
a dialect, `check` validates rule operation names, arity, and whether an attribute
descriptor is required.

The parser and AST carry this information, and `check` resolves declarations
and checks descriptor schemas. Rust code generation remains future work. The
tuple and binder examples remain syntax fixtures and include operators outside
the Rust tensor IR.

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
  use brackets with an optional dtype prefix (`X: bf16[Batch..., M, K]`).
  Without a prefix, dtype is unrestricted. `S: scalar` means a rank-zero tensor
  equivalent to `S: []`; `S: f32[]` also constrains its dtype.
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
  semantics treat graph literals as rank-zero tensors. Explicit typed forms
  include `1:i32` and `1.0:f32`; annotation also participates in matching.
  Rust tensor literal construction requires a concrete dtype. Broadcasting and
  floating format interpretation remain host responsibilities.
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

Inheritance remains unexpanded in the AST: inherited rules have null local
`lhs` and `rhs` pointers. The `check` command resolves base rules, checks parameter
signatures, substitutes bindings, and combines inherited restrictions in a
separate checked representation.
Abstract body annotations and graphs are checked when instantiated by a concrete
rule; unused abstract bodies remain available in the source AST.

## Semantic analysis and checked IR

`check FILE` parses and loads imports, then builds an owning `core::Program`.
It prints the checked IR on success and source diagnostics on failure. Exit
codes are 0 for success, 1 for syntax/import/semantic errors, and 2 for usage
or input-file errors. Parsing with `parse --ast` still exposes the source AST.

```sh
bazel run //:tepl -- check "$PWD/examples/lora.tepl"
bazel run //:tepl -- check "$PWD/examples/inherited.tepl"
```

The API in `src/core/analyze.h` accepts an AST whose imports have already been
loaded by `resolveImports` from `src/imports.h`. Analysis never mutates the AST
or its shared nodes.
`AnalysisResult::program` is present only when every stage succeeds; diagnostics
include definition locations, instantiation locations, and related declarations.
The checked program remains valid after the AST is destroyed.

```cpp
auto parsed = tepl::parse(source, filename);
// Check parse errors before accessing program.
auto imports = tepl::resolveImports(*parsed.program);
// Check import errors before analysis.
auto checked = tepl::core::analyze(*parsed.program);
if (checked.ok()) {
  std::cout << tepl::core::formatProgram(*checked.program);
}
```

The initial analyzer supports declared tensor operations, fixed and trailing
variadic operands, operation aliases, LHS captures and bindings, graph literals,
shape/dtype restrictions, typed conditions, descriptor derivations, and
concrete instances of abstract rules. Operations must have visible dialect
declarations; syntax-only rules using undeclared operators remain parseable but
cannot produce checked IR. Tuples and projections are explicitly unsupported.
Abstract parameter declarations are checked immediately; template bodies are
checked when instantiated. Abstract definitions do not become executable rules.

The checked program contains operation and attribute-schema tables, inferred
host signatures, concrete rules, and semantic types. IDs index these tables;
capture, dimension, and descriptor IDs are local to a rule. Original names,
literal spelling, and source origins are retained. LHS matching and RHS
construction have separate node types. Repeated captures/dimensions use the
same identity. Inherited shape restrictions are retained as additional runtime
constraints; conflicting explicit dtypes and incompatible rank restrictions are
diagnosed. A shape sequence can be empty, so `[N, ...]` requires rank at least
one and conflicts with `scalar`, but is compatible with `[M]`.

Imported template operations resolve in their definition's file scope. Instance
operation bindings resolve in the instance's scope. Private template imports
and unselected rules do not become visible in the importing file. Expansion
copies expression trees so instances cannot change their template or each other.

Host functions have one inferred signature per name across the program; v1 has
no overloads. Explicit `fn` parameter signatures support `tensor`, `bool`,
`index`, `index_list`, `i64`, `f64`, and `attrs`. Successful host values are
distinct from failure: host calls are fallible, and failure rejects a match.
Conditions must produce `Bool`. Numeric operators accept compatible numeric
types; equality supports numbers and booleans. Tensor and dimension-sequence
comparisons must use host functions. Ambiguous calls such as `f(X) == g(X)`
need another typed use or an explicit function parameter signature.

Host integer constants adopt their numeric context; otherwise unsigned constants
default to `Index` and negative constants to `I64` after inference. Decimal
constants use `F64`. Index/signed integer literal ranges are unsigned/signed
64-bit; floating host literals must fit finite `F64`. Tensor captures and their
element dtypes remain separate from these host value types. Graph literals keep
their exact spelling and optional dtype without choosing an implicit tensor dtype.

Captured descriptors are available to `where` and `derive`; derived descriptors
become available after their assignment, in source order. Forward/self
references, duplicate definitions, unknown references, and incompatible schema
uses are rejected. A derived descriptor cannot be read in `where`. Derivations
can read earlier descriptors but cannot reference constructed RHS values.

The implementation lives in `src/core/`. Public IR definitions are in `ir.h`,
`ids.h`, and `types.h`; `analyze.h` exposes the analysis API and `print.h` exposes
the dump formatter. The implementation has these responsibilities:

| File | Responsibility |
| --- | --- |
| `analyze.cc` | Coordinate resolution, expansion, checking, and type finalization |
| `analysis_context.cc` | Shared analysis state, symbol lookup, and diagnostics |
| `resolve.cc` | Register declarations and build each file's import/use scope |
| `expand.cc` | Select templates, validate bindings, and clone expressions with source origins |
| `check.cc` | Check graph patterns, captures, constraints, and derivation order |
| `check_expr.cc` | Lower conditions and derivations into typed expressions |
| `type_inference.cc` | Solve type equations, default literals, and validate type requirements |
| `literal.cc` | Shared graph/host literal range validation |
| `print.cc` | Format the checked program |

Inference and rule-scope headers are private to the analysis target. Expression
checking reads the rule's scope; graph checking introduces symbols and advances
descriptor availability. Failed expressions and rules are excluded from the
result, and unresolved types cannot be materialized into the public IR. Analysis
reports unloaded imports instead of creating empty scopes for them.

Common source locations live in `src/source.h`; AST and core share language
operator identities and spellings from `src/operators.h`. Bazel exposes
`//src/core:ir`, `//src/core:analysis`, and `//src/core:print`.

Runtime shape checks, host numerical legality, descriptor contents, output
metadata inference, and rewrite insertion remain runtime responsibilities.
This stage does not generate executable code or prove tensor equivalence.

## Scope and next steps

The parser also builds an owning AST with source spans for valid input.
The core analyzer resolves symbols, checks declared tensor operation signatures,
infers host-function types, and validates descriptor references. Runtime legality
and e-graph behavior remain host responsibilities.
The examples are parser fixtures, not claims of tensor equivalence.
Multiple-root patterns and variadic expression operands are deferred.

The grammar contains no C++ actions. Bazel generates lexer/parser and visitor
sources under the build directory. `AstBuilder` converts their parse tree into
the project AST. Semantic analysis lowers that AST into the checked core IR.

AST types, printing, and construction live in `src/ast/`. The builder uses one
visitor with separate rule and dialect source files. Parsing lives in
`src/parse.cc`, import loading in `src/imports.cc`, and semantic analysis in
`src/core/`. Bazel exposes parsing through `//:frontend` and import loading
through `//:imports`.

Code generation from the checked `core::Program` remains future work.
See [the egg lab](labs/rust-egg/README.md) for the runtime, reference Rust output,
and integration tests.

In the reference Rust runtime, `derive` host functions return `Option<OpAttrs>`
containing only operation descriptors. Both `where` and `derive` use matched LHS
inputs; derivations are evaluated in source order. LoRA uses
`infer_lora_out(X, A, B, @outer, @inner)`
for its final descriptor, avoiding any dependency on a constructed RHS value.
The checked Rust rewrite path validates the entire RHS tree and infers every
operation's shape and dtype before insertion. It requires output compatibility
with the matched root before union. Host legality predicates still establish
numerical equivalence. Ordinary
LHS graph variables are valid tensor host arguments without shape declarations.

Supported tensor dtypes are `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, and
`f16/bf16/f32/f64`. An annotation never inserts a cast or implicit promotion.
Unknown dtype names, invalid integer literal ranges, duplicate local tensor
declarations, and declarations for missing LHS captures produce diagnostics
through `check`.
See [basic examples](examples/basic.tepl) and the paired Rust behavior tests in
[labs/rust-egg/tests/dtypes.rs](labs/rust-egg/tests/dtypes.rs).
