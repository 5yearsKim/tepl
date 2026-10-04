# Rust output reference

This directory is executable compiler output generated from `examples/`.
`tools/regenerate_lab.sh --check` verifies that it matches the compiler. The
layout groups code by responsibility; the generator's producer determines
whether a file is copied or emitted.

## File ownership

| Files | Producer |
| --- | --- |
| `analysis/shape.rs` | Emit from checked operation shape programs |
| `analysis/dtype.rs` | Emit from optional checked operation dtype policies |
| `builtins/*` | Copy common, shape, dtype, and error helpers from `templates/rust/src/builtins/` |
| `analysis/tensor.rs` | Copy shared shape/dtype inference combination |
| `analysis/tensor_analysis.rs` | Copy direct egg integration |
| `analysis/tensor_analysis_data.rs` | Copy conservative metadata storage and merge |
| `analysis/bindings.rs`, `analysis/inference.rs`, `analysis/mod.rs` | Copy input bindings, inference result, and public API wiring |
| `pattern/*`, `types.rs` | Copy maintained runtime |
| `op_node.rs` | Copy node template and insert operation sum types |
| `dialects/*` | Emit from checked dialect declarations |
| `rules/*.rs` | Emit patterns, ordered early condition plans, derivations, and builders |
| Root, dialect, and rule module declarations | Emit module wiring |

The compiler uses `templates/rust/src/` as its template input. Project-specific
code is emitted by `src/codegen/rust/`. Each output path has one producer and
is recorded in `.tepl-generated-files`. Application-specific policies live
outside this directory, under `src/host/`.

## Shape execution contract

`infer_shape` implements all 24 example TEPL blocks, with one private operation
evaluator per block. Statements retain source order. Expressions use shared
builtins and checked indexing/arithmetic. Comprehensions collect fallible
results in iteration order; Boolean operators and conditional branches
short-circuit. Generated lists are owned values and cloned when reused.

Stored dimensions and index attributes are `u64`; signed padding is `i64`.
Inputs widen to `i128` for shape expressions. Overflow, division by zero, and
bad indexing are invalid. Every yielded dimension must convert back to `u64`.
Rust list positions use `usize`, with checked conversions at index boundaries.
Assertions include the operation name and TEPL line/column in their message.

Missing shape definitions return `Inference::Unknown`. Literal nodes have
scalar shapes. Constant, all-gather, and reduce-scatter operations have no
example shape block and remain unknown. There is no implicit payload inference.
Shape inference does not execute opaque regions, validate constant bytes, or
establish numerical rewrite equivalence.

## Dtype policy and public API

`infer_dtype` dispatches only on [declared TEPL policies](../../../../examples/dtype_guide.md).
`same`, `same_numeric`, and `same_float` use shared helpers without promotion.
Zero operands under a common policy return `Unknown`; a concrete policy fixes
the result dtype. Missing policies return `Unknown`. Runtime literals read
their explicit dtype; named inputs obtain metadata from the binding table.
Dot, convolution, and constants have no declared dtype policy in the example
dialect, so their default combined tensor inference remains unknown even when
shape inference succeeds.

```rust
use crate::ir::analysis::{TensorAnalysis, TensorBindingTable, TensorInfo};
use crate::ir::rules::simple::rule_commute_add;

let analysis = TensorAnalysis::new(inputs);
let graph = egg::EGraph::new(analysis);
let rewrite = rule_commute_add::build_rewrite(()).unwrap();
```

`build_rewrite(functions)` uses `TensorAnalysis` and its tensor inference.
Callers supply only explicit custom rule functions. Native rule builtins and
checked integer operators call shared helpers in `builtins/`; no host methods
are generated for builtin calls. Builtin errors reject the candidate before
insertion. Shape expressions use `i128`, while rule values keep `u64`/`i64`. The same inference is
used for e-class analysis and RHS validation before insertion.
`build_rewrite_with(metadata, inference, functions)` supports custom analyses.
The LoRA demo uses this explicit path with `host::LoraAnalysis`, whose dot dtype
policy is application-specific rather than inferred from an operation name.

Tensor declarations and the builtin/operator prefix of `where` share a
`MatchChecks` plan. Shapes bind dimensions; one cursor per search branch runs
conditions in source order as their bindings appear. Missing bindings wait;
false or checked builtin failure prunes immediately. The first host-containing
condition ends the early prefix, including nested and short-circuited calls.
Host conditions and all later conditions retain application-time order.
Rematching and final validation recheck the early prefix before host calls,
derivations, or insertion. See the [runtime description](../../../../templates/rust/README.md).

## Validation

From the repository root:

```sh
./tools/regenerate_lab.sh --check
cargo test --manifest-path labs/rust-egg/Cargo.toml --all-targets
./tools/test_codegen.sh
```

Tests cover every example shape block, all language builtins on freshly generated
custom dialects, nested metadata, variadics, checked boundaries, default rewrites,
and metadata merge behavior. The integration suite also checks empty projects,
renamed/nested modules, deterministic regeneration, and custom callback builders.
