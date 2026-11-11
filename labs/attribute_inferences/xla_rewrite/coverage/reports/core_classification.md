# Current Constructive Core Coverage

The paper audit classifies **21 cases as main-theorem scope: 13 direct forms + 8 restricted insertion reductions**. The current native `--core` backend compiles and portable-validates all 21. Of these, **18 use constructive metadata DAGs**, while **3 existing reversal forms still dispatch to specialized schemas**.

| Classification | Cases | Measured native implementation path |
|---|---:|---|
| Direct unsigned core | 10 | Existing constructive core DAG |
| Restricted insertion | 8 | Generic LHS normalization + constructive core DAG |
| Direct signed core forms | 3 | Existing specialized reversal schemas |
| Main-theorem scope total | 21 | 18 constructive + 3 specialized |
| Additional structural candidates | 10 | Outside this main-core boundary; not rerun here |

## Populations and interpretation

The audit contains 38 axis-inference forms: 31 historically inferred and 7 not proved. Main-theorem scope is **21/38 (55.3%)**, or **21/31 (67.7%)** among historically successful forms. Actual constructive dispatch is **18/38 (47.4%)**, or **18/31 (58.1%)** among historically successful forms. The previous direct-form count 13 is a mathematical classification; it was not evidence that all 13 executed the unsigned constructive algorithm.

The separate frozen default-backend evaluation reports 30/32. Its population and compiler run differ, so this update does not replace that result with 21/32 or silently change its recorded outcomes.

## Case-level native dispatch

| Case | Paper classification | Current path | Compile / portable validation |
|---|---|---|---|
| transpose_compose | main_theorem_form | Constructive core | Passed |
| scalar_broadcast_reverse | main_theorem_reduction | Generic insertion reduction | Passed |
| scalar_broadcast_transpose | main_theorem_reduction | Generic insertion reduction | Passed |
| reduce_compose | main_theorem_form | Constructive core | Passed |
| reduce_transpose | main_theorem_form | Constructive core | Passed |
| reverse_compose_disjoint | main_theorem_form | Specialized schema | Passed |
| dot_associate | main_theorem_form | Constructive core | Passed |
| dot_transpose_output | main_theorem_form | Constructive core | Passed |
| dot_move_reverse | main_theorem_form | Specialized schema | Passed |
| reduce_compose_leading | main_theorem_form | Constructive core | Passed |
| broadcast_compose_append | main_theorem_reduction | Generic insertion reduction | Passed |
| dot_swap_operands | main_theorem_form | Constructive core | Passed |
| dot_reduce_batch | main_theorem_form | Constructive core | Passed |
| expr_reverse_compose_general | main_theorem_form | Specialized schema | Passed |
| catalog_broadcast_transpose | main_theorem_reduction | Generic insertion reduction | Passed |
| catalog_broadcast_inserted_unit_dims | main_theorem_reduction | Generic insertion reduction | Passed |
| catalog_broadcast_transpose_broadcast | main_theorem_reduction | Generic insertion reduction | Passed |
| catalog_dot_transpose_both_operands | main_theorem_form | Constructive core | Passed |
| catalog_dot_move_broadcast | main_theorem_reduction | Generic insertion reduction | Passed |
| catalog_dot_reduce_output_free | main_theorem_form | Constructive core | Passed |
| catalog_reduction_broadcast_original_axes | main_theorem_reduction | Generic insertion reduction | Passed |

## Evidence and limits

[Current results](../results/core_classification.json) include compiler, audit, input, dialect, and dispatch-source hashes. The runner reads the paper inputs and migrates only old syntax/import qualification; source guards and supplied expressions are retained. Eight insertion forms also have bounded execution tests in `//tests:insertion_test`, including the recorded normalized unit insertion, scalar/singleton cases, incompatible extents, RHS boundary checks, and guard preservation. Compiling 21 forms is not a new full-domain completeness proof.

The constructive extension requires exact-real semantics, original input extents at least two, positive supplied fresh extents, independent unknowns, and the restricted single-component insertion boundary. The broadcast/transpose entry uses the recorded companion. Arbitrary internal RHS insertion remains outside the generic path. Native `--core` support does not add C++/Rust generated-adapter support.

To make **all 21** execute constructively, the remaining implementation step is to route the three pure reversal forms through the compiled signed core machinery already used inside the insertion reduction, and validate that dispatch separately.

```sh
bazel build //:tepl
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/core_classification.py \
  --paper /path/to/TEPL-paper
bazel test //tests:insertion_test --test_output=all
```

Measured UTC: `2026-11-11T08:40:54.888428+00:00`. Existing frozen measurements and paper files were not changed by this runner.
