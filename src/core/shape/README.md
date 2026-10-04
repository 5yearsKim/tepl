# Checked operation shape programs

`dialect/check.cc` calls `shape::check` after checking an operation's signature
and resolving its attribute schema. Successful checking stores a `shape::Program`
in `Operation::shape`. Operations without a shape definition keep an empty
optional. Shape programs are checked even when no rule references the operation.

The flow is: bind parameters, check statements in order, check the yield, and
finish type inference. Core resolves syntax and types without executing shape
programs. Rust code generation turns this IR into checked pure evaluators; host
applications retain e-class facts, input metadata, and merge policies.

| File | Responsibility |
| --- | --- |
| `ir.h` | Owning checked expressions, symbols, operand bindings, and statements |
| `types.*` | Concrete shape types and signed 128-bit literal range |
| `../builtins/catalog.*` | Shared builtin identities, signatures, domains, and section availability |
| `builtins.h` | Compatibility names for the shared catalog |
| `check.*` | Parameter/operand agreement, statement order, assertions, and yield |
| `check_context.h` | State and name scope for checking one operation |
| `expression.*` | Name and attribute resolution, expressions, and builtin calls |
| `type_inference.*` | List type equations, empty-list inference, and cycle detection |
| `print.*` | Resolved expressions and types in checked-program output |

Types are Integer, Bool, and homogeneous nested lists. Each shape program owns
its type and symbol tables; IDs index those tables. Expressions hold constant
shared children, and no AST pointers survive checking. Generic builtin types are
instantiated independently for each call. Empty lists must obtain an element
type from context or later use; unresolved element types are errors.

Integer semantics use checked signed 128-bit computation to accommodate both
u64 shapes and signed i64 attributes. Generated evaluators validate yielded dimensions against u64, check indices
and arithmetic, and honor short-circuit control flow. This checker validates literals and expression types, preserving
runtime assertions and value-dependent checks in source order.

See [the shape guide](../../../examples/shape_guide.md) for syntax and contracts.
Run `bazel test //tests:shape_core_test` for builtin, scope, type, diagnostic, and
complete tensor/scalar dialect coverage.
