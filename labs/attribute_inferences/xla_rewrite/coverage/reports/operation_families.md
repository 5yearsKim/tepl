# Operation-family coverage

The corpus contains **209 rewrite cases** representing **199 catalog schemas**. The 1 negative control is excluded from family counts and percentages.

| Operation family | Representative operations | Cases involving family, n (%) | Within evaluation scope, n (%) |
|---|---|---:|---:|
| Elementwise | Add, multiply, compare, select, exp, log | 118 (56.5%) | 85 (72.0%) |
| Indexing | Slice, gather, scatter, reverse, pad, concatenate | 70 (33.5%) | 8 (11.4%) |
| Reduction | Reduce sum, reduce min/max, reduce window | 30 (14.4%) | 9 (30.0%) |
| Broadcast | Broadcast in dimension | 42 (20.1%) | 16 (38.1%) |
| Dot | Generalized dot | 18 (8.6%) | 11 (61.1%) |
| Transpose | Transpose | 16 (7.7%) | 10 (62.5%) |
| Reshape | Reshape | 27 (12.9%) | 4 (14.8%) |
| Convolution | Convolution | 7 (3.3%) | 0 (0.0%) |

Each case contributes once to every family appearing on either side of its original rewrite. The cases-involving-family percentages use 209 distinct rewrite cases as the denominator. Memberships overlap; counts may sum to more than the corpus size and percentages may sum to more than 100%.

Within-evaluation-scope percentages use each row's family case count as the denominator. Scope membership is the frozen axis-only classification's `in_scope` flag, including axis-inference and verification-only tasks. It describes eligibility in that evaluation, not successful verification or a general limit on TEPL support. The negative control is excluded from both columns.

## Classification method

Classification parses the original semantic rule graphs, before axis-task migration or adapter normalization, and resolves operation aliases to their canonical HLO names. It includes all cases, regardless of evaluation scope or verification outcome. Repeated operations and operations present on both sides contribute only one membership per family. Captured subgraphs are opaque; operations inside captures, imported declarations, and metadata functions in attributes or conditions are not counted.

The complete mapping below assigns each local HLO operation to exactly one family. Elementwise includes scalar/complex applications, conversion, map, and the constant constructors zero_like, fill, and scalar. Indexing includes coordinate generation (iota), rotation, padding, concatenation, and indexed reads/updates. Generalized dot remains in Dot even though its semantics use summation. Reduce-window operations belong to Reduction; restricted and general convolution belong to Convolution. Transpose and Reshape are separate families.

These are case memberships based on explicit operations, rather than the catalog's single schema category. They characterize the written TEPL rules, not every possible instantiation of each XLA schema.

## Complete operation-to-family mapping

| Family | Canonical local HLO operations |
|---|---|
| Elementwise | `add`, `multiply`, `negate`, `subtract`, `zero_like`, `divide`, `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `power`, `maximum`, `minimum`, `logical_not`, `logical_and`, `logical_or`, `logical_xor`, `real`, `imag`, `complex`, `select`, `remainder`, `convert`, `compare`, `clamp`, `cbrt`, `erf`, `logistic`, `tanh`, `fill`, `scalar`, `map` |
| Indexing | `slice`, `reverse`, `concatenate`, `pad`, `nonnegative_pad`, `dynamic_slice`, `dynamic_update_slice`, `iota`, `rotate`, `scatter`, `scatter_add`, `gather` |
| Reduction | `reduce_sum`, `reduce`, `reduce_min`, `reduce_max`, `reduce_and`, `reduce_or`, `reduce_window`, `reduce_window_sum`, `reduce_window_min`, `reduce_window_max` |
| Broadcast | `broadcast_in_dim` |
| Dot | `dot_general` |
| Transpose | `transpose` |
| Reshape | `reshape` |
| Convolution | `convolution`, `general_convolution` |

## Evidence and reproduction

[Mapping](../data/operation_family_mapping.json) and [per-case memberships](../data/case_families.json) provide the full classification. Generated memberships record input and source hashes. The generator checks source hashes and operation inventories against the frozen synthesis classification; no compiler run is needed.

Excluded negative control: `negative_subtract_to_add`.

```sh
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/operation_families.py
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/operation_families.py --check
```
