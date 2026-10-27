# XLA Rewrite Specification Size and Attribute Synthesis

Across **36 successfully synthesized concrete cases representing 30 catalog schemas**, inference reduces TEPL rule specifications from **243 to 198 noncomment lines (18.5%)** and from **3,370 to 2,277 lexical tokens (32.4%)**. It removes **47 explicit RHS descriptor occurrences**, including copied descriptors and computed descriptors, while preserving the equations, input declarations and written conditions.

## Evaluation population

The implementation contains **209 concrete cases representing 199 XLA rewrite schemas**. Of these, **119 cases** have the operation semantics required by the verifier; **90** lack required semantics. Among supported cases, **43 contain RHS descriptors to synthesize**. The other **76** require no RHS descriptors: their replacement uses attribute-free operations, such as elementwise addition, or returns an existing matched value, as in `add(X, 0) => X`.

| Measurement | Result |
|---|---:|
| Semantically supported cases | 119/209 (56.9%) |
| Supported cases proved with explicit attributes | 94/119 (79.0%) |
| Supported cases containing RHS descriptors | 43/119 (36.1%) |
| Successful full attribute inference | **36/43 (83.7%)** |
| Attribute-bearing cases classified within the theorem fragment | 10/43 (23.3%) |
| Successful inference within that fragment | **10/10 (100.0%)** |
| Successful inference beyond that fragment | **26/33 (78.8%)** |

The theorem classification uses the written operations and read-once/component structure. The mathematical guarantee also assumes extents at least two and ideal arithmetic; the experiment preserves the original case conditions and does not add those extent restrictions. The contraction/permutation implementation has an ordered complete path; sum/addition cases can use general verification. Thus 10/10 is an empirical result on classified examples, not an exhaustive test of completeness. Some excluded operations could potentially be re-encoded using theorem primitives; that alternative encoding was not evaluated.

Each inference success in the archived run required `status=proved`, `additional-guard: true`, the expected inferred descriptor count, and successful executable generation and JSON validation. In **4 cases**, an inferred implementation verifies although the original explicit formulas remain unproved. The seven unsuccessful attribute-bearing cases remain in the coverage denominator; code-size reductions below are conditional on successful full inference.

## Measured representation sizes

| Representation | Total noncomment lines | Total lexical tokens |
|---|---:|---:|
| XLA implementation excerpts, shared lines counted once¹ | 1,398 | 10,849 |
| TEPL with explicit RHS attributes, 36 rule bodies | 243 | 3,370 |
| TEPL with inferred RHS attributes, the same 36 rule bodies | **198** | **2,277** |

¹ The XLA row is manually selected source context for the same 30 represented schemas. Its branches/helpers retain more general ranks, operand variants, graph traversal and policy than several concrete TEPL cases. They are not 36 independent scope-equivalent implementations. General visitor setup and library implementations are excluded. The source row is descriptive; dividing it by the TEPL rows would not isolate a language or inference benefit. The paired TEPL comparison does hold rule scope fixed.

Within TEPL, synthesis removes **45 noncomment lines** and **1,093 tokens**. The source contains **32 removed `derive` statements**, counting both descriptor assignments and their local metadata calculations. Descriptor occurrences include inline reuse such as `dot[attr_inner.attrs]`, so not every inferred descriptor corresponds to a separate assignment or a removed physical line.

| Scope of paired TEPL measurement | Explicit lines | Inferred lines | Line reduction | Explicit tokens | Inferred tokens | Token reduction |
|---|---:|---:|---:|---:|---:|---:|
| Rule bodies; primary measurement | 243 | 198 | 18.5% | 3,370 | 2,277 | 32.4% |
| Theorem-fragment rule bodies | 84 | 67 | 20.2% | 1,019 | 776 | 23.8% |
| Beyond-fragment rule bodies | 159 | 131 | 17.6% | 2,351 | 1,501 | 36.2% |
| Whole rule files, including imports | 314 | 269 | 14.3% | 4,061 | 2,968 | 26.9% |

| Per-case reduction in rule-body size | Median | Minimum | Maximum |
|---|---:|---:|---:|
| Lines | 16.7% | 0.0% | 66.7% |
| Tokens | 24.6% | 7.4% | 87.3% |

Physical lines depend on formatting. Some metadata formulas occupy one long line, while inline attribute erasure can reduce tokens without removing a line. Tokens therefore provide a useful second measurement. Code size measures specification burden, not development time, maintainability, proof-checking speed or optimized-program performance.

## What logic disappears

### Transpose composition

XLA builds a replacement transpose and calls `ComposePermutations`. Explicit TEPL writes `gather(inner.permutation, outer.permutation)` in a descriptor; inferred TEPL leaves the replacement permutation unspecified. Here the benefit is removal of a small but necessary metadata expression.

Selected XLA metadata/construction excerpt (surrounding logic is omitted here; measured ranges are listed below):

```cpp
  if (HloOpcode::kTranspose == operand->opcode()) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateTranspose(
                       transpose->shape(), operand->mutable_operand(0),
                       ComposePermutations(operand->dimensions(),
                                           transpose->dimensions())));
  }
```

Measured rule body: **7 → 3 lines; 53 → 22 tokens**.

Explicit TEPL ([source](../../tepl/rules/semantic/transpose_compose.tepl)):

```tepl
rule transpose_compose {
    (attr_outer: transpose (attr_inner: transpose X)) => (attr_combined: transpose X)
    derive {
        attr_combined.attrs = { ..attr_outer.attrs,
            permutation: gather(attr_inner.attrs.permutation, attr_outer.attrs.permutation) };
    }
}
```

Inferred TEPL ([source](../../tepl/rules/full_auto/transpose_compose.tepl)):

```tepl
rule transpose_compose {
    (attr_outer: transpose (attr_inner: transpose X)) => (attr_combined: transpose X)

}
```

### Transpose of a broadcast

The XLA helper inverts the permutation, maps broadcast axes, checks ordering/layout policy, and builds the replacement. The explicit TEPL algebraic case computes an inverse map, surviving/new axis positions, and output extents. Inference removes those calculations, leaving the input declaration and graph equation. The explicit formula is unproved; the inferred construction has an archived proof. Source layout policy remains outside this TEPL case.

Selected XLA metadata/construction excerpt (surrounding logic is omitted here; measured ranges are listed below):

```cpp
  auto inv_perm = InversePermutation(dimensions);
  absl::InlinedVector<int64_t, 8> new_bcast_dims;
  for (int64_t dim : broadcast->dimensions()) {
    new_bcast_dims.push_back(inv_perm[dim]);
  }
  if (!absl::c_is_sorted(new_bcast_dims)) {
    return absl::OkStatus();
  }
// ... surrounding source omitted ...
  return ReplaceInstruction(
      transpose, MakeBroadcastHlo(broadcast->mutable_operand(0), new_bcast_dims,
                                  transpose->shape()));
}
```

Measured rule body: **12 → 4 lines; 267 → 34 tokens**.

Explicit TEPL ([source](../../tepl/rules/semantic/catalog_broadcast_transpose_broadcast.tepl)):

```tepl
rule catalog_broadcast_transpose_broadcast {
    X: [Dims...]
     (attr_p: hlo.transpose (attr_b: hlo.broadcast X)) => (attr_out: hlo.broadcast X)
     derive {
      let inv = [sum([if attr_p.attrs.permutation[j] == i then j else 0 for j in range(len(attr_p.attrs.permutation))]) for i in range(len(attr_p.attrs.permutation))];
      let d = gather(inv,attr_b.attrs.dimensions);
      let oldnew = exclude(range(len(attr_p.attrs.permutation)),attr_b.attrs.dimensions);
      let newpos = exclude(range(len(attr_p.attrs.permutation)),d);
      let oldshape = [if contains(attr_b.attrs.dimensions,i) then sum([if attr_b.attrs.dimensions[j]==i then Dims[j] else 0 for j in range(len(Dims))]) else attr_b.attrs.new_extents[sum([if k<i then 1 else 0 for k in oldnew])] for i in range(len(attr_p.attrs.permutation))];
      attr_out.attrs = { dimensions: d, new_extents: gather(gather(oldshape,attr_p.attrs.permutation),newpos) };
     }
}
```

Inferred TEPL ([source](../../tepl/rules/full_auto/catalog_broadcast_transpose_broadcast.tepl)):

```tepl
rule catalog_broadcast_transpose_broadcast {
    X: [Dims...]
     (attr_p: hlo.transpose (attr_b: hlo.broadcast X)) => (attr_out: hlo.broadcast X)

}
```

### Swapping dot operands

XLA swaps dimension-number lists, constructs a dot, and computes an output permutation from batch/free axis counts. The TEPL case selects ordinary rank-two matrix products; inference removes the two RHS descriptors. Its LHS rank/axis conditions remain written. XLA's parameter-placement and precision policy are not implemented by this algebraic identity.

Selected XLA metadata/construction excerpt (surrounding logic is omitted here; measured ranges are listed below):

```cpp
  std::vector<int64_t> permutation;
  const int64_t num_batch_dims = dot_dims.lhs_batch_dimensions_size();
  const int64_t lhs_non_contracting_batch =
      new_dot->operand(0)->shape().dimensions().size() - num_batch_dims -
      dot_dims.lhs_contracting_dimensions_size();
  const int64_t rhs_non_contracting_batch =
      new_dot->operand(1)->shape().dimensions().size() - num_batch_dims -
      dot_dims.rhs_contracting_dimensions_size();
  for (int i = 0; i != num_batch_dims; ++i) {
    permutation.push_back(i);
  }
  for (int i = 0; i != rhs_non_contracting_batch; ++i) {
    permutation.push_back(num_batch_dims + lhs_non_contracting_batch + i);
  }
  for (int i = 0; i != lhs_non_contracting_batch; ++i) {
    permutation.push_back(num_batch_dims + i);
  }
  ABSL_ASSIGN_OR_RETURN(HloInstruction * new_transpose,
                        MakeTransposeHlo(new_dot, permutation));
```

Measured rule body: **13 → 8 lines; 123 → 78 tokens**.

Explicit TEPL ([source](../../tepl/rules/semantic/dot_swap_operands.tepl)):

```tepl
rule dot_swap_operands {
    X: [M, K]
    Y: [K, N]
    (attr_d: dot X Y) => (attr_p: transpose (attr_out: dot Y X))
    where { attr_d.attrs.lhs_batching_dimensions == [] && attr_d.attrs.rhs_batching_dimensions == []
         && attr_d.attrs.lhs_contracting_dimensions == [1]
         && attr_d.attrs.rhs_contracting_dimensions == [0] }
    derive {
        attr_out.attrs = { lhs_batching_dimensions: [], rhs_batching_dimensions: [],
            lhs_contracting_dimensions: [0], rhs_contracting_dimensions: [1] };
        attr_p.attrs = { permutation: [1, 0] };
    }
}
```

Inferred TEPL ([source](../../tepl/rules/full_auto/dot_swap_operands.tepl)):

```tepl
rule dot_swap_operands {
    X: [M, K]
    Y: [K, N]
    (attr_d: dot X Y) => (attr_p: transpose (attr_out: dot Y X))
    where { attr_d.attrs.lhs_batching_dimensions == [] && attr_d.attrs.rhs_batching_dimensions == []
         && attr_d.attrs.lhs_contracting_dimensions == [1]
         && attr_d.attrs.rhs_contracting_dimensions == [0] }

}
```

## Counting and source attribution

- **Selection:** every successful full-auto case in the saved synthesis results, with no selection based on code-size reduction. Multiple concrete cases for one schema remain separate in the paired TEPL measurements.
- **TEPL rule bodies:** count from `rule` through its closing brace, including input declarations, both graph patterns, `where` conditions and `derive` blocks. Exclude imports, blank lines and comments. Whole-file totals above show the effect of including imports.
- **Lines:** physical lines containing code after string-aware removal of `//` and `/* ... */` comments. Braces on their own lines count. No reformatting or manual shrinking is performed.
- **Tokens:** use the same lexical convention for C++ and TEPL: identifiers, numbers, quoted strings, recognized multicharacter operators and remaining punctuation. Whitespace and comments do not count. These are reproducible lexical units, not compiler-specific token streams or language-model tokens.
- **XLA excerpts:** manually identify relevant branches or dedicated helpers rather than counting entire visitors such as `HandleDot`. Ranges are inclusive and pinned by the snapshot hash. Union selected lines across all cases before counting: shared branches are counted once. Per-case XLA counts in the appendix overlap and must not be summed.
- **Additional support:** the main XLA row excludes generic libraries and common visitor setup. A selected inventory of ten local helpers adds **163 lines / 1,213 tokens**; the deduplicated union with the excerpts is **1,561 lines / 12,062 tokens**. This is not a complete transitive dependency count. External helpers such as `ComposePermutations`, `InversePermutation`, shape utilities and graph-construction APIs are not expanded.
- **Shared TEPL support:** the local semantic HLO dialect is **423 noncomment lines / 3,875 tokens**, reported once and excluded from per-rule totals. The compiler, solver and shared attribute-construction machinery are also excluded. The comparison measures code written per rewrite after this shared infrastructure is available.

The directly attributable inference saving is the explicit-to-inferred TEPL difference. The XLA excerpts illustrate the imperative work associated with these transformations, but their additional generality prevents an equal-functionality percentage-reduction claim. A future scope-matched XLA baseline would need to implement the same scoped equations and assumptions explicitly.

## Paper-ready result

> We evaluate attribute synthesis on 209 concrete rewrite cases representing 199 XLA schemas. Of the 119 cases with supported operation semantics, 43 contain RHS attribute descriptors. Synthesis succeeds on 36/43 cases (83.7%), including 10/10 cases classified within the theorem fragment and 26/33 beyond it. Across the 36 successful cases, inference eliminates 47 explicit descriptor occurrences and reduces rule-body size from 243 to 198 noncomment lines (18.5%) and from 3,370 to 2,277 lexical tokens (32.4%), preserving the written equations and conditions. Four inferred implementations verify although the original explicit metadata formulas remain unproved. These results quantify reduced per-rewrite specification burden under the evaluated algebraic model.

## Per-case measurements

TEPL columns count rule bodies. `T` denotes the theorem-fragment classification; `G` denotes supported cases beyond it. Source links lead to the actual measured variants. The XLA column gives excerpt lines/tokens and overlaps between cases.

| Case | Scope | RHS descriptors inferred | Explicit lines / tokens | Inferred lines / tokens | XLA excerpt lines / tokens |
|---|:---:|---:|---:|---:|---:|
| [transpose_compose](../../tepl/rules/full_auto/transpose_compose.tepl) | T | 1 | [7 / 53](../../tepl/rules/semantic/transpose_compose.tepl) | 3 / 22 | 7 / 53 |
| [broadcast_add](../../tepl/rules/full_auto/broadcast_add.tepl) | G | 1 | [6 / 59](../../tepl/rules/semantic/broadcast_add.tepl) | 6 / 54 | 72 / 545 |
| [broadcast_multiply](../../tepl/rules/full_auto/broadcast_multiply.tepl) | G | 1 | [6 / 59](../../tepl/rules/semantic/broadcast_multiply.tepl) | 6 / 54 | 72 / 545 |
| [scalar_broadcast_reverse](../../tepl/rules/full_auto/scalar_broadcast_reverse.tepl) | G | 1 | [4 / 36](../../tepl/rules/semantic/scalar_broadcast_reverse.tepl) | 4 / 31 | 18 / 117 |
| [scalar_broadcast_transpose](../../tepl/rules/full_auto/scalar_broadcast_transpose.tepl) | G | 1 | [6 / 87](../../tepl/rules/semantic/scalar_broadcast_transpose.tepl) | 5 / 56 | 18 / 117 |
| [reduce_negate](../../tepl/rules/full_auto/reduce_negate.tepl) | G | 1 | [3 / 26](../../tepl/rules/semantic/reduce_negate.tepl) | 3 / 21 | 13 / 135 |
| [reduce_compose](../../tepl/rules/full_auto/reduce_compose.tepl) | T | 1 | [6 / 70](../../tepl/rules/semantic/reduce_compose.tepl) | 5 / 53 | 30 / 253 |
| [reduce_transpose](../../tepl/rules/full_auto/reduce_transpose.tepl) | T | 1 | [6 / 68](../../tepl/rules/semantic/reduce_transpose.tepl) | 5 / 53 | 41 / 292 |
| [reverse_compose_disjoint](../../tepl/rules/full_auto/reverse_compose_disjoint.tepl) | G | 1 | [6 / 68](../../tepl/rules/semantic/reverse_compose_disjoint.tepl) | 5 / 51 | 20 / 209 |
| [reverse_add_scalar](../../tepl/rules/full_auto/reverse_add_scalar.tepl) | G | 2 | [5 / 52](../../tepl/rules/semantic/reverse_add_scalar.tepl) | 5 / 42 | 37 / 293 |
| [slice_compose](../../tepl/rules/full_auto/slice_compose.tepl) | G | 1 | [9 / 94](../../tepl/rules/semantic/slice_compose.tepl) | 5 / 49 | 15 / 155 |
| [dot_associate](../../tepl/rules/full_auto/dot_associate.tepl) | T | 2 | [14 / 136](../../tepl/rules/semantic/dot_associate.tepl) | 14 / 126 | 269 / 1757 |
| [dot_transpose_output](../../tepl/rules/full_auto/dot_transpose_output.tepl) | T | 1 | [11 / 121](../../tepl/rules/semantic/dot_transpose_output.tepl) | 9 / 90 | 51 / 413 |
| [dot_move_reverse](../../tepl/rules/full_auto/dot_move_reverse.tepl) | G | 2 | [9 / 95](../../tepl/rules/semantic/dot_move_reverse.tepl) | 9 / 85 | 97 / 669 |
| [reduce_compose_leading](../../tepl/rules/full_auto/reduce_compose_leading.tepl) | T | 1 | [6 / 70](../../tepl/rules/semantic/reduce_compose_leading.tepl) | 5 / 53 | 30 / 253 |
| [slice_compose_zero_starts](../../tepl/rules/full_auto/slice_compose_zero_starts.tepl) | G | 1 | [5 / 67](../../tepl/rules/semantic/slice_compose_zero_starts.tepl) | 5 / 62 | 15 / 155 |
| [broadcast_compose_append](../../tepl/rules/full_auto/broadcast_compose_append.tepl) | G | 1 | [8 / 101](../../tepl/rules/semantic/broadcast_compose_append.tepl) | 6 / 67 | 11 / 86 |
| [dot_swap_operands](../../tepl/rules/full_auto/dot_swap_operands.tepl) | T | 2 | [13 / 123](../../tepl/rules/semantic/dot_swap_operands.tepl) | 8 / 78 | 49 / 414 |
| [dot_no_contraction](../../tepl/rules/full_auto/dot_no_contraction.tepl) | G | 2 | [12 / 114](../../tepl/rules/semantic/dot_no_contraction.tepl) | 8 / 75 | 45 / 439 |
| [dot_reduce_product](../../tepl/rules/full_auto/dot_reduce_product.tepl) | G | 1 | [9 / 87](../../tepl/rules/semantic/dot_reduce_product.tepl) | 8 / 72 | 58 / 580 |
| [dot_reduce_batch](../../tepl/rules/full_auto/dot_reduce_batch.tepl) | T | 1 | [10 / 115](../../tepl/rules/semantic/dot_reduce_batch.tepl) | 8 / 84 | 49 / 381 |
| [expr_reverse_compose_general](../../tepl/rules/full_auto/expr_reverse_compose_general.tepl) | G | 1 | [3 / 72](../../tepl/rules/semantic/expr_reverse_compose_general.tepl) | 3 / 28 | 20 / 209 |
| [expr_slice_compose_general](../../tepl/rules/full_auto/expr_slice_compose_general.tepl) | G | 1 | [3 / 127](../../tepl/rules/semantic/expr_slice_compose_general.tepl) | 3 / 66 | 15 / 155 |
| [catalog_elementwise_compare_selected_mask](../../tepl/rules/full_auto/catalog_elementwise_compare_selected_mask.tepl) | G | 1 | [5 / 67](../../tepl/rules/semantic/catalog_elementwise_compare_selected_mask.tepl) | 5 / 62 | 15 / 122 |
| [catalog_elementwise_compare_max](../../tepl/rules/full_auto/catalog_elementwise_compare_max.tepl) | G | 1 | [4 / 42](../../tepl/rules/semantic/catalog_elementwise_compare_max.tepl) | 4 / 37 | 13 / 113 |
| [catalog_elementwise_compare_min](../../tepl/rules/full_auto/catalog_elementwise_compare_min.tepl) | G | 1 | [4 / 42](../../tepl/rules/semantic/catalog_elementwise_compare_min.tepl) | 4 / 37 | 15 / 117 |
| [catalog_broadcast_multiply_constant_into_broadcast](../../tepl/rules/full_auto/catalog_broadcast_multiply_constant_into_broadcast.tepl) | G | 2 | [7 / 101](../../tepl/rules/semantic/catalog_broadcast_multiply_constant_into_broadcast.tepl) | 6 / 78 | 31 / 294 |
| [catalog_broadcast_transpose](../../tepl/rules/full_auto/catalog_broadcast_transpose.tepl) | G | 1 | [5 / 94](../../tepl/rules/semantic/catalog_broadcast_transpose.tepl) | 4 / 32 | 9 / 82 |
| [catalog_broadcast_transpose_broadcast](../../tepl/rules/full_auto/catalog_broadcast_transpose_broadcast.tepl) | G | 1 | [12 / 267](../../tepl/rules/semantic/catalog_broadcast_transpose_broadcast.tepl) | 4 / 34 | 43 / 358 |
| [catalog_broadcast_slice_broadcast_only](../../tepl/rules/full_auto/catalog_broadcast_slice_broadcast_only.tepl) | G | 1 | [6 / 114](../../tepl/rules/semantic/catalog_broadcast_slice_broadcast_only.tepl) | 5 / 78 | 20 / 148 |
| [catalog_indexing_slice_singleton_stride](../../tepl/rules/full_auto/catalog_indexing_slice_singleton_stride.tepl) | G | 1 | [4 / 76](../../tepl/rules/semantic/catalog_indexing_slice_singleton_stride.tepl) | 3 / 21 | 36 / 270 |
| [catalog_dot_transpose_both_operands](../../tepl/rules/full_auto/catalog_dot_transpose_both_operands.tepl) | T | 2 | [5 / 128](../../tepl/rules/semantic/catalog_dot_transpose_both_operands.tepl) | 5 / 118 | 48 / 400 |
| [catalog_dot_move_broadcast](../../tepl/rules/full_auto/catalog_dot_move_broadcast.tepl) | G | 2 | [6 / 150](../../tepl/rules/semantic/catalog_dot_move_broadcast.tepl) | 5 / 109 | 119 / 868 |
| [catalog_dot_reduce_output_free](../../tepl/rules/full_auto/catalog_dot_reduce_output_free.tepl) | T | 2 | [6 / 135](../../tepl/rules/semantic/catalog_dot_reduce_output_free.tepl) | 5 / 99 | 92 / 730 |
| [catalog_dot_scalar_multipliers](../../tepl/rules/full_auto/catalog_dot_scalar_multipliers.tepl) | G | 2 | [6 / 156](../../tepl/rules/semantic/catalog_dot_scalar_multipliers.tepl) | 5 / 129 | 110 / 832 |
| [catalog_reduction_broadcast_original_axes](../../tepl/rules/full_auto/catalog_reduction_broadcast_original_axes.tepl) | G | 2 | [6 / 98](../../tepl/rules/semantic/catalog_reduction_broadcast_original_axes.tepl) | 5 / 71 | 41 / 290 |

Unsuccessful eligible cases: `slice_reverse`, `expr_slice_reverse_general`, `expr_convolution_scale_filter_bias`, `catalog_elementwise_log_power`, `catalog_broadcast_divide_broadcast`, `catalog_broadcast_slice_broadcast`, `catalog_dot_slice_output`.

## XLA source mapping and scope differences

All ranges refer to [algebraic_simplifier.cc](../../catalog/upstream/algebraic_simplifier.cc) in the pinned XLA revision `e02127ba9c9c48912b0e6776fe5bfe3843d81042`. The machine-readable [source map](../data/rule_specification_sources.json) records every inclusive range.

| Catalog schema | Source lines | Attribution and scope |
|---|---|---|
| `transpose-reshape-transpose-compose` | 10248–10254 | Nested-transpose branch; ComposePermutations implementation is outside this snapshot. |
| `broadcast-elementwise` | 6344–6428 | Complete broadcast-sinking helper, shared by addition and multiplication cases; also handles other elementwise functions, arbitrary arity and scalar expansion. |
| `broadcast-scalar-through-indexing` | 5722–5743 | Shared user-scanning branch for scalar broadcast through permutations/subsets; TEPL cases select reverse and transpose. |
| `reduction-negate` | 8973–8985 | Negated sum-reduction branch; source checks floating dtype, zero initializer and additive reducer. |
| `reduction-compose` | 8890–8918, 9077–9081 | MergeReduces and its eligibility branch, shared by both scoped sum cases; source supports more reducers and axis sets. |
| `reduction-transpose` | 9029–9073 | Complete transpose/reduce branch including optional output transpose and policy; TEPL uses a rank-two sum instance. |
| `indexing-reverse-compose` | 7107–7129 | Nested-reverse branch and local setup, shared by general and disjoint-axis cases; includes cancellation. |
| `indexing-reverse-elementwise` | 7107–7110, 7133–7166 | Scalar-broadcast recognizer and binary-elementwise reverse motion, including both operand orders; TEPL selects addition. |
| `indexing-slice-compose` | 7663–7677 | General unit-stride composition branch, shared by general, rank-one and zero-start cases. |
| `dot-associate` | 3829–4180 | Complete nested-dot reassociation helper, including both directions, generalized axes, cost selection and optional output transpose; TEPL selects ordinary matrix multiplication. |
| `dot-transpose-output` | 10262–10328 | Canonical-dot transpose implementation and its operand-swap policy; the alternate noncanonical-dot implementation is excluded. TEPL uses rank-two matrix multiplication. |
| `dot-move-reverse` | 3521–3567, 3570–3607, 3744–3781 | Shared dot-reordering setup, reverse branch and common construction/cost tail; pad and broadcast branches excluded. TEPL uses a rank-one inner product. |
| `broadcast-compose` | 5755–5765 | Nested-broadcast branch; TEPL case restricts broadcasts to appended axes. |
| `dot-swap-parameter-operand` | 2943–2993 | Complete operand-swap helper, including parameter policy, precision and arbitrary free/batch axis bookkeeping; TEPL uses a rank-two identity without policy screening. |
| `dot-no-contraction` | 3783–3827 | Complete no-contraction lowering helper; includes generalized normalization, dtype conversions and graph metadata. TEPL uses same-type rank-one operands. |
| `dot-reduce-product` | 4182–4243 | Complete product/reduction lowering helper; includes generalized normalization and accumulation conversion. TEPL uses a same-type rank-one inner product. |
| `dot-reduce-output-batch` | 9203–9254 | Batch-to-contraction reduction branch; source also handles remaining batch/free reductions, whereas TEPL reduces the sole batch axis. |
| `elementwise-compare-selected-mask` | 5884–5898 | Mask-selection comparison branch; TEPL uses scalar literal branches. Shared compare operand setup is excluded. |
| `elementwise-compare-max` | 5904–5916 | GT guard, captures and maximum branch, including both matched operands; TEPL selects one operand order. |
| `elementwise-compare-min` | 5904–5906, 5917–5928 | Shared GT guard/captures and minimum branch, including both matched operands; TEPL selects one operand order. |
| `broadcast-multiply-constant-into-broadcast` | 5190–5225 | Constant-factor/broadcast branch, including literal and numerical-policy checks; TEPL treats the scalar factor as a capture. |
| `broadcast-transpose` | 5668–5676 | Same-rank broadcast-to-transpose branch; uses the external InversePermutation helper. |
| `broadcast-transpose-broadcast` | 2796–2862 | Complete transpose/broadcast helper, including sorted-map and layout policy; TEPL evaluates a general insertion-only algebraic equation. |
| `broadcast-slice-broadcast-only` | 7679–7698 | Broadcast-only slicing predicate and replacement branch; TEPL selects a rank-one base mapped to axis zero. |
| `indexing-slice-singleton-stride` | 7496–7533, 7990–7994 | Complete redundant-stride helper and caller; source changes qualifying axes selectively, while the TEPL case requires singleton output extents throughout. |
| `dot-transpose-both-operands` | 2864–2925 | Complete operand-transpose helper, with canonical batch-axis constraints and precision handling; TEPL selects rank-two matrix transposes. |
| `dot-move-broadcast` | 3521–3567, 3678–3742, 3744–3781 | Shared dot-reordering setup, broadcast branch and common construction/cost tail; pad and reverse branches excluded. TEPL selects one broadcast-created contraction axis. |
| `dot-reduce-output-free` | 8779–8888 | Complete free-axis reduction/dot helper with mapping and profitability logic; TEPL reduces the left free axis of a rank-two dot. |
| `dot-scalar-multipliers` | 704–853 | Complete scalar-factor relocation helper with cost-based placement, graph traversal and constant folding; TEPL is one local scalar-factor identity. |
| `reduction-broadcast-original-axes` | 9322–9370 | Reduce/broadcast commutation branch with reduction-axis lookup and surviving broadcast-axis renumbering; TEPL is a rank-one-base sum specialization. |

Selected local helper ranges, counted separately:

- `ConstructToDotMaps`: 473–506.
- `ConstructFromDotMaps`: 513–538.
- `IsOpCodeMultiplyCommutative`: 669–679.
- `MakeScalarInstruction`: 681–700.
- `IsParameterLike`: 2930–2940.
- `NormalizeDotOperandToBatchMajorAndContractingMinor`: 2995–3014.
- `AddReduce`: 3016–3026.
- `MakeMultiplyForPrecisionAlgorithm`: 4252–4256.
- `OutputIsPermutationOfOperandElements`: 5544–5557.
- `OutputIsSubsetOfOperandElements`: 5561–5578.

## Provenance and reproduction

Inference outcomes refer to the saved run at **2026-10-26T20:39:44.091717+00:00**, with compiler revision `17d48f05087b540cae87648fed3b43d9f14c2166` and compiler SHA-256 `be7c72d324619a53ea1758b65d009f73e6213f388f9e4ecdb034282e6d8ca465`. This measurement verifies each selected explicit and inferred rule against its archived file hash, checks the saved proof/acceptance evidence, and checks that generated variants differ only by the existing descriptor-erasure transformation. It does not rerun the semantic sweep.

**Current compiler freshness:** 9 compiler source dependencies differ from the archived run. The success rates describe that archived compiler, not a fresh measurement of the current compiler. The selected rule files still match their archived hashes. Changed paths:

- `src/core/attr_synth/synthesize.cc`
- `src/core/attr_synth/synthesize.h`
- `src/core/semantics/context.cc`
- `src/core/semantics/context.h`
- `src/core/semantics/proof/proof_policy.h`
- `src/core/semantics/proof/verify.cc`
- `src/core/semantics/proof/verify.h`
- `src/core/semantics/prove_rule.cc`
- `src/core/semantics/prove_rule.h`

From the repository root:

```sh
python3 labs/xla_rewrite/coverage/scripts/measure_rule_specification.py
python3 labs/xla_rewrite/coverage/scripts/measure_rule_specification.py --check
python3 -m unittest discover -s labs/xla_rewrite/coverage/tests -p 'test_rule_specification.py'
```

The runner writes [measurements and input hashes](../results/rule_specification_complexity.json) and this report. `--check` recomputes both without writing and detects stale outputs. For a fresh compiler evaluation, follow the [existing experiment reproduction steps](../README.md#reproduce), review any changed population/source mappings, then rerun this measurement.
