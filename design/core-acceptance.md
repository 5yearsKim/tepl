# Conditional metadata constructors

`--core` compiles admitted whole-axis constructors and supported view/reindexing
schemas to a reusable metadata DAG. Admission checks operation semantics,
structure, dtype transport, arithmetic policy, and supplied metadata helpers.
The loader regenerates the canonical DAG and checks schedule/contract binding;
it does not prove that the written guard implies feasibility.

For a legal source match the runtime checks the written condition, evaluates the
DAG predicates, and only then evaluates attribute builders. False feasibility
means non-application. Invalid source metadata, unsupported programs, arithmetic
failures and resource exhaustion retain separate outcomes. Emission and symbolic
verification occur during compilation, never per match.

Constructor soundness applies only to admitted templates and schema assumptions.
Other candidates need independent definedness, legality and semantic-equivalence
verification under C && q. Axis feasibility alone does not establish arbitrary
tensor equivalence. Unresolved candidates do not become executable.

Portable programs use format version 4 and retain DAG predicates and builders.
Canonical constructors remain sound for legal zero/singleton extents; completeness
claims retain their original extent domain. Generated C++/Rust support symbolic
programs; metadata DAG export remains explicitly unsupported.

Validation: `//labs/accepted_core:acceptance_test` and
`//labs/accepted_core:extensions_test` exercise dense reference equivalence,
canonical DAG integrity, dtype contracts, supplied fields and metadata helpers.
