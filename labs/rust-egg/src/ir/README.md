# Rust output reference

This directory is an executable reference for the compiler's intended Rust
output. It includes current compiler output and handwritten additions. Its
layout groups code by responsibility; copying versus emission is a producer
concern, not a separate Rust module hierarchy.

## File ownership

| Files | Intended producer | Current status |
| --- | --- | --- |
| `analysis/shape.rs` | Emit from checked operation shape programs | Handwritten reference for all 24 example dialect shape blocks |
| `analysis/shape_builtins.rs` | Copy from `runtime/rust/src/analysis/shape_builtins.rs` | Copied runtime; keep byte-for-byte synchronized |
| `analysis/dtype.rs` | Maintained Rust policies with operation-specific dispatch | Handwritten; no TEPL dtype language is planned |
| `analysis/payload.rs` | Explicit dialect payload policy | Handwritten constant metadata policy |
| `analysis/tensor.rs` | Copy shared tensor inference combination | Handwritten reusable reference |
| `analysis/tensor_analysis.rs` | Copy direct egg integration | Handwritten reusable reference |
| `analysis/tensor_analysis_data.rs` | Copy conservative fact storage and merge | Handwritten reusable reference |
| `analysis/bindings.rs`, `analysis/inference.rs` | Copy input table and inference result | Handwritten reusable reference |
| `analysis/mod.rs` | Fixed public API wiring | Handwritten reference |
| `pattern/*`, `types.rs` | Copy maintained runtime | Current compiler output |
| `op_node.rs` | Copy node template and insert operation sum types | Current compiler output |
| `dialects/*` | Emit from checked dialect declarations | Current compiler output |
| `rules/*.rs` | Emit patterns, conditions, derivations, and default/explicit builders | Current compiler bodies with handwritten default API wrappers |
| Root, dialect, and rule module declarations | Emit module wiring | Root includes the handwritten analysis registration |

The compiler should not read this directory as its template input. Reusable
source implementations belong in `runtime/rust/src/`; project-specific code is
written by `src/codegen/rust/`. Moving the remaining shared reference files into
the runtime and implementing evaluator emission is the next compiler stage.
Each output path should have one producer. `.tepl-generated-files` currently
records ownership by the existing compiler, not a list of all reference files.

## Shape execution contract

`infer_shape` implements each TEPL block with one operation evaluator, in
statement order. Assertions, list comprehensions, and yields call shared
builtins; indexing and scalar arithmetic use checked primitives. Comprehensions
collect fallible results in iteration order. Boolean operators and conditional
branches retain Rust's short-circuit evaluation.

Stored dimensions and index attributes are `u64`; signed padding is `i64`.
Input dimensions and numeric attributes widen to `i128` for shape expressions.
Arithmetic fails on overflow or division by zero. A yield must convert every
dimension back to `u64`, rejecting negative or out-of-range values. Rust's
`usize` remains the collection length/index representation, with checked
conversion at indexing boundaries.

Missing shape definitions return `Inference::Unknown`; a defined evaluator's
failed assertion, bad signature, indexing error, or overflow returns `Invalid`.
Literal nodes have a runtime scalar-shape policy. Constants have no TEPL shape
block: pure `infer_shape` returns `Unknown`, while `payload.rs` supplies their
explicitly typed metadata to combined tensor inference. All-gather and
reduce-scatter remain unknown because the process-grid information is absent.

Shape inference preserves the TEPL contracts; it does not execute opaque
regions, validate constant bytes, or establish numerical rewrite equivalence.

## Dtype policy and public API

Dtype inference is maintained in Rust. Data movement and the reference
single-result reductions/collectives preserve compatible input dtypes;
arithmetic requires matching non-Boolean dtypes. Exponential, log, sqrt, and
rsqrt require floating-point operands. Constants and literals read their
explicit type. Dot/convolution precision overrides and dot algorithm overrides
return `Unknown` rather than assuming output semantics. This policy does not
choose reducer behavior or authorize numerical reassociation. Future dialects
must select an appropriate policy or return `Unknown`.

```rust
use crate::ir::analysis::{TensorAnalysis, TensorBindingTable, TensorInfo};
use crate::ir::rules::simple::rule_commute_add;

let analysis = TensorAnalysis::new(inputs);
let graph = egg::EGraph::new(analysis);
let rewrite = rule_commute_add::build_rewrite(()).unwrap();
```

`build_rewrite(functions)` uses `TensorAnalysis` and the same tensor inference
as its e-class analysis. Callers only supply explicit custom rule functions.
`build_rewrite_with(metadata, inference, functions)` retains explicit callbacks
for custom analyses and runtime validation tests. Both paths perform the same
conditions and RHS validation before inserting nodes.

## Validation and remaining generation gap

Run `cargo test --manifest-path labs/rust-egg/Cargo.toml --all-targets` from the
repository root. `tests/shape_reference.rs` covers all declared shape blocks and
invalid inputs, signed padding, alternate layouts, variadics, overflow, and
missing definitions. Analysis and rewrite tests cover storage, merging, and
conditional rewrite behavior.

The compiler currently emits builtins and callback-only rule builders, without
operation evaluators or the full analysis module wiring. Generating directly
into this reference would replace its root registration and rule wrappers.
`tools/regenerate_lab.sh --check` therefore reports intentional differences.
For current compiler experiments, generate into a separate output directory and
review changes. Once the compiler implements this reference, clean generation
must compile, pass the behavior tests, and reproduce it after formatting.
