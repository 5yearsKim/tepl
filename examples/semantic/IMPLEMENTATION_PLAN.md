# Semantic constraint-plan implementation

## Objective

Keep `semantics(out_idx)` as the user-facing DSL. Generate constraints from its
primitive expressions and explicit shape contracts. Infer RHS attributes at
match time without recognizing dialect operation names or rewrite templates.
The [README](README.md) documents current behavior and limitations.

## Implemented pipeline

1. **Frontend.** Require the output index parameter. Remove typed-axis
   `generate`/`reduce_sum` binders and their AST/type variants. Preserve optional
   semantics, existing metadata checking, RHS holes, typed pins, and diagnostics.
2. **Semantic lowering.** Resolve operands, attributes, output coordinates, and
   reduction binders; inline lets. Remove `Form` classification. Lower resolved
   shape expressions and assertions into the same primitive bytecode.
3. **Rule composition.** Build one operation DAG for LHS and RHS. Captured tensors
   retain identity; bound intermediates resolve to their expressions. Attribute
   descriptors retain shared field identities.
4. **Compile-time constraint plan.** Recursively substitute intermediate reads,
   freshen reduction binders, distribute scalar products over sums, merge identical
   terms, and extract coefficients, captured reads, and rectangular domains.
   Intern symbolic access expressions referring to node shapes, attribute fields,
   output coordinates, and bound coordinates. Admit pure coordinate routing;
   keep vector lengths symbolic. Normalize reduction names, factor order, and
   coefficients; remove unused literal one-point reductions. Mark remaining
   unused binders for guarded match-time elimination. No operation or attribute name classification
   is used. Unsupported expressions and expansion limits record a fallback reason.
5. **Match-time SMT synthesis.** Derive finite vector domains from explicit
   `is_valid_axis_list` assertions. Instantiate legality, shape, and semantic
   constraints. Generate a joint term bijection and a reduction-coordinate
   bijection for each paired term, requiring matching extents and access maps.
   Add occurrence bijections for repeated factors; all occurrences share the
   reduction mapping. Allow unused RHS reductions to disappear only under
   one-point-domain guards. Preserve nonunit and empty reduction multiplicities.
   Ask Z3 for attributes satisfying all terms simultaneously. Exclude failed
   candidates using model-blocking clauses. The semantic constraints are only
   sufficient: on their UNSAT, discard their solver scope and search the full
   metadata-legal domain. Without a compiled plan, try the existing source-copy
   or identity preference first. Unrestricted search can still be expensive.
   A certified ordered-contraction path now precedes this search. It constructs
   attributes from source logical indices and RHS boundary/order requirements.
   Unrecognized semantics and excluded metadata use this generic solver.
6. **Verification.** Expand tensor reads generically. Compare symbolic index
   polynomials as a sufficient proof; use complete finite entry-polynomial
   expansion when necessary. Use exact rational coefficients. Strict mode checks
   expanded scalar traces. Resource limits yield unknown.
7. **Rust runtime integration.** Embed a per-rule `AUTO_CONSTRAINT_PLAN` constant,
   containing the semantic bytecode and compiled constraint plan. Serialize it
   using the versioned `tepl-constraints-2` format and bind concrete metadata at
   match time through the existing native solver ABI. Report `ConstraintsZ3`
   when the compiled plan supplies the verified witness. Construct descriptors
   only after verification, before graph mutation. Regenerate old applications;
   backward request-format compatibility is not provided. No C++ adapter feature
   work is included in this extension.
8. **Compile-time rejection.** Detect exact scalar counterexamples for
   unconditional, unconstrained, attribute-free rules with supported semantics.
   Preserve ordinary behavior for explicit rules lacking semantics or outside
   this proof fragment. Auto rules require the expanded definitions.
9. **Cleanup.** Remove aligned-axis and ordered-network checkers, their IRs and
   runtime evaluators, shape derivation helpers, template classifier, named
   rewrite transformations, and obsolete fixture dialects. Preserve unrelated
   tensor-analysis inference and research notes.
10. **Measurements.** Add native phase timings, solver/candidate counters,
    configurable preferences, seed, and candidate limits. Compare metadata-only
    and compiled-constraint modes with ordered inference and preferences disabled.
    Also measure production dispatch. Compile rules once, export CSV, and retain
    unknown/invalid outcomes and fallback reasons in the before/after report.

## File ownership

| Responsibility | Files |
| --- | --- |
| Parsing and semantic checking | `grammar/Tepl.g4`, `src/ast/builder_semantics.cc`, `src/core/semantics/check.cc` |
| Resolved semantics | `src/core/semantics/ir.*` |
| Rule lowering and domains | `src/core/inference/check.cc`, `semantic_ir.h`, `ir.h` |
| Compile-time semantic constraints | `src/core/inference/constraints.*` |
| Term normalization and unused binders | `src/core/inference/constraint_normalize.*` |
| Constraint instantiation | `src/core/semantics/constraint_match.h` |
| Primitive constraints and proof | `src/core/semantics/solver.*` |
| Ordered semantic contracts and certificates | `src/core/semantics/ordered.*` |
| Shared runtime ABI | `src/core/semantics/runtime.cc`, Bazel `//src/core:libtepl_semantic_runtime.so` |
| Descriptor code generation | `src/codegen/common/inference_emitter.cc` |
| Generated adapters | `templates/rust/src/builtins/semantic.rs`, `templates/cpp/src/builtins/semantic.h` |
| Example language | `dialects/tensor.tepl` |
| Regression coverage | `tests/constraint_plan_test.cc`, `tests/semantic_test.cc`, `tests/ordered_test.cc`, `tests/codegen/semantic*`, integration scripts |
| Extension regression coverage | `tests/constraint_extensions_test.cc` |
| Performance measurements | `tools/semantic_benchmark.cc`, `tools/summarize_semantic_benchmark.py`, `benchmarks/` |

## Acceptance checks

- Renaming all operations and fields preserves inference behavior.
- Transpose insertion, pinned permutations, distribution, reassociation, and
  batched association use the same solver pipeline.
- LoRA and repeated reductions work through expression composition.
- One compiled distribution plan solves several ranks, including multiple
  batching and contracting axes, without recompilation.
- Renamed operations and attribute fields produce the same semantic constraints.
- Shared unknown descriptors are solved jointly; impossible shared permutation
  roots exhaust the fallback rather than being mislabeled by plan failure.
- Unused static and dynamic singleton reductions succeed through compiled
  constraints; zero and nonunit domains retain their mathematical behavior.
- A valid multiplicity rewrite succeeds through unrestricted fallback when
  reduction-bijection constraints are insufficient.
- Repeated captures preserve occurrence counts and binder independence.
- Alpha-equivalent sums combine coefficients and cancel correctly.
- Strict arithmetic retains fallback reasoning.
- Serialized plans execute through the native ABI and appear in generated Rust.
- Scalar drop-operand rewrites produce compile-time counterexamples.
- Zero and singleton dimensions are interpreted correctly.
- A failed index normalization cannot become an invalid conclusion by itself.
- Timeout, unsupported indexing, or unavailable native runtime declines a match
  before graph mutation.
- Rust and C++ applications exercise the same native implementation.
- Rust applications construct reordered rank-100 contraction attributes and
  decline certified ordering failures without graph mutation.
- The structural and independent generic engines agree for all legal source
  pairings of three one-axis inputs in the reassociation test.
- Ordinary TEPL parser, metadata, analysis, and code-generation tests still pass.

## Remaining extensions

These are separate follow-ups, not claims about the current compiler:

The Rust-focused ordered-contraction implementation completes contract extraction,
logical-index normalization, direct construction, independent witness/obstruction
checks, and native runtime dispatch. Its certifier is conservative and recognizes
specified primitive normal forms. The rank-two compile-time coverage check is
not implemented. Broader semantic equivalence during admission, constrained RHS
attributes, repeated captures, addition, and singleton/empty dimensions require
new reasoning before entering that complete structural fragment.

1. **Guarded metadata lowering.** Encode branch-local legality under implications
   and lower shape/index comprehensions with proved bounds.
2. **More attribute domains.** Add scalar integer/Boolean and noninjective list
   domains, with explicit finite-domain contracts and model reconstruction.
3. **Broader semantic constraints.** Extend sufficient access-map matching to
   used singleton coordinates, unused nonunit multiplicities, coordinate
   arithmetic, reordered reduction domains during term merging, and alternative
   polynomial decompositions. Prove any proposed completeness
   claim separately; the current term/reduction bijections are sufficient only.
4. **Better performance.** Cache results by plan and matched metadata; decompose
   independent attribute components; strengthen rejection clauses using verified
   counterexamples rather than excluding one complete model at a time.
5. **Broader compile-time reasoning.** Translate supported shape declarations and
   guards into counterexample queries. Keep applicability, fixed-metadata
   impossibility, and universal coverage as distinct judgments.
6. **Rank-parametric proofs.** Establish an exact translation and a computable
   small-counterexample theorem for a specified sublanguage before enabling any
   bounded-rank coverage certificate.
7. **Numerical semantics and packaging.** Specify operational floating-point
   behavior before adding error-bound policies; package the native runtime for
   additional platforms.

The [rank research question](math_theorem.md),
[finite-rank restrictions](finite_rank_restrictions.md), and
[research notes](research/ordered_contractions.md) remain mathematical drafts.
The ordered-contract path applies the instance boundary/order reasoning in its
checked fragment. The rank cutoff and broader research proposals do not certify
universal coverage in this compiler.
