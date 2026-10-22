# Native test coverage

For changes to attribute inference, start with:

```sh
bazel test //tests:named_nodes_test //tests:inferred_applicability_test \
  //tests:ordered_synthesis_test //tests:rewrite_test \
  //tests:syntax_test //tests:core_test
```

Run the full active native suite before completing a broader change:

```sh
bazel test //tests:all --build_tests_only -- \
  -//tests:rust_codegen_test -//tests:cli_test
```

The exclusions contain deferred backend integration checks. Rust generation and
runtime tests are outside the current native implementation scope.

| Test | Main responsibility |
| --- | --- |
| `named_nodes_test` | Node properties, descriptor reuse/derivation, suggested-rule round trips, and nested feasible/infeasible replacements. |
| `inferred_applicability_test` | Generated conditions and constructors across ranks, JSON replay, unsupported cases, and corrupted-program rejection. |
| `ordered_synthesis_test` | Independent exhaustive attribute oracle, positive/negative certificates, paired contraction order, and high-rank cases. |
| `semantic_test` | Symbolic laws, arithmetic policy, sufficient user conditions, partial routing, and independent dense references. |
| `convolution_test` | Calculated reads, coefficient inference, spatial ranks 1–3, empty batch/channel cases, and illegal or corrupted metadata. |
| `semantic_cli_test` | CLI selection, policy, imports, output, and diagnostic behavior. Operator laws belong to the C++ tests. |
| Other parser, type, loader, and operator tests | Their own syntax, legality, lifecycle, and operation-specific regressions. |

Keep a case when it exercises a distinct branch, known counterexample, or failure
mode. Extend the independent oracle for new fragment support or newly
discovered counterexamples.
Use small dense tensors: larger extents add cost without exercising another
axis relationship. Reuse a compiled rule when varying only match metadata.

The convolution examples combine one ordinary case at each spatial rank with
each empty-axis case at rank one. Empty cases cover input and nested
distribution; kernel and transpose variants use nonempty data. A two-dimensional
coefficient example covers differing strides/dilations and uneven division; nested coefficients and a large
stride have separate cases. The semantic suite retains four-operand explicit
and inferred dot chains and the inexpensive exhaustive permutation/reduction
loops. Duplicate recompilation and repeated CLI checks of whole operator
families are omitted.
