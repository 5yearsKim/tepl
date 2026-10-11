# Legacy element semantics migration

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
4. **Constraint generation.** At concrete matched metadata, derive finite vector
   domains from explicit `is_valid_axis_list` assertions. Generate primitive
   length, position, distinctness, extent, legality, and output-shape constraints.
   Represent intermediate shapes symbolically; preserve output axis order.
5. **SMT synthesis.** Ask Z3 for a metadata-legal candidate. Apply deterministic
   initial preferences, then remove those preferences if necessary. Exclude
   failed joint candidates using model-blocking clauses. This remains potentially
   expensive when many legal candidates fail.
6. **Verification.** Expand tensor reads generically. Compare symbolic index
   polynomials as a sufficient proof; use complete finite entry-polynomial
   expansion when necessary. Use exact rational coefficients. Strict mode checks
   expanded scalar traces. Resource limits yield unknown.
7. **Runtime integration.** Export one versioned native solver ABI. Generated Rust
   and C++ adapters serialize the same plan and metadata, load that library, and
   construct descriptors only after a verified result. Existing rule execution
   checks finish before graph mutation.
8. **Compile-time rejection.** Detect exact scalar counterexamples for
   unconditional, unconstrained, attribute-free rules with supported semantics.
   Preserve ordinary behavior for explicit rules lacking semantics or outside
   this proof fragment. Auto rules require the expanded definitions.
9. **Cleanup.** Remove aligned-axis and ordered-network checkers, their IRs and
   runtime evaluators, shape derivation helpers, template classifier, named
   rewrite transformations, and obsolete fixture dialects. Preserve unrelated
   tensor-analysis inference and research notes.

## File ownership

| Responsibility | Files |
| --- | --- |
| Parsing and semantic checking | `grammar/Tepl.g4`, `src/ast/builder_semantics.cc`, `src/core/semantics/check.cc` |
| Resolved semantics | `src/core/semantics/ir.*` |
| Rule lowering and domains | `src/core/inference/check.cc`, `semantic_ir.h`, `ir.h` |
| Primitive constraints and proof | `src/core/semantics/solver.*` |
| Shared runtime ABI | `src/core/semantics/runtime.cc`, Bazel `//src/core:libtepl_semantic_runtime.so` |
| Descriptor code generation | `src/codegen/common/inference_emitter.cc` |
| Generated adapters | `templates/rust/src/builtins/semantic.rs`, `templates/cpp/src/builtins/semantic.h` |
| Example language | `dialects/tensor.tepl` |
| Regression coverage | `tests/semantic_test.cc`, `tests/codegen/semantic*`, integration scripts |

## Acceptance checks

- Renaming all operations and fields preserves inference behavior.
- Transpose insertion, pinned permutations, distribution, reassociation, and
  batched association use the same solver pipeline.
- LoRA and repeated reductions work through expression composition.
- Scalar drop-operand rewrites produce compile-time counterexamples.
- Zero and singleton dimensions are interpreted correctly.
- A failed index normalization cannot become an invalid conclusion by itself.
- Timeout, unsupported indexing, or unavailable native runtime declines a match
  before graph mutation.
- Rust and C++ applications exercise the same native implementation.
- Ordinary TEPL parser, metadata, analysis, and code-generation tests still pass.

## Remaining extensions

These are separate follow-ups, not claims about the current compiler:

1. **Guarded metadata lowering.** Encode branch-local legality under implications
   and lower shape/index comprehensions with proved bounds.
2. **More attribute domains.** Add scalar integer/Boolean and noninjective list
   domains, with explicit finite-domain contracts and model reconstruction.
3. **Stronger semantic constraints.** Derive tensor-access and polynomial-degree
   constraints before requesting models. Current Z3 constraints primarily encode
   metadata legality; the verifier establishes tensor equality afterward.
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
Their theorems are not used to certify this generic implementation.
