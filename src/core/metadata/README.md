# Checked operation metadata programs

`dialect/check.cc` calls `metadata::check` for each operation's optional shape
and dtype blocks after checking its signature and attribute schema. Successful
checking stores a `metadata::Program` in `Operation::shape` or `Operation::dtype`.
Programs are checked even when no rule references the operation.

Both kinds share expressions, statements, symbols, type inference, and builtin
resolution. `ProgramKind` determines parameter and yield types: shape parameters
are integer lists and yield an integer list; dtype parameters and results are
dtype values. A trailing variadic parameter binds a list of the corresponding
operand metadata. Each program owns its type and symbol tables, and no AST
pointers survive checking.

| File | Responsibility |
| --- | --- |
| `ir.h` | Checked expressions, symbols, operand bindings, statements, and program kind |
| `types.*` | Integer, Bool, DType, homogeneous lists, and literal ranges |
| `../builtins/catalog.*` | Shared builtin identities, signatures, domains, and availability |
| `check.*` | Parameters, statements, assertions, and yield contracts |
| `check_context.h` | Program state and name scope |
| `expression.*` | Names, attributes, expressions, and builtin calls |
| `type_inference.*` | Type equations, empty-list inference, and cycle detection |
| `print.*` | Resolved expressions and types |

Integer semantics use checked signed 128-bit computation for u64 shapes and
signed i64 attributes. Generated evaluators validate dimensions against u64,
check indices and arithmetic, and preserve short-circuit control flow. Dtype
values support equality and the five classification predicates, with no
implicit promotion or arithmetic. Attributes may supply dtype values and lists.
Assertions and local bindings execute in source order in both backends.

See the [shape guide](../../../examples/shape_guide.md) and
[dtype guide](../../../examples/dtype_guide.md). Checker tests live in
`tests/shape_core_test.cc` and `tests/dtype_program_test.cc`; generated integration
fixtures exercise both Rust and C++ evaluators and rule bindings.
