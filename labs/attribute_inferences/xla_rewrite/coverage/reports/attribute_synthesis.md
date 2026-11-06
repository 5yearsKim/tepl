# XLA Axis-Only Attribute Inference Evaluation

Axis-field inference succeeds on **30/32 (93.8%) cases**. The run resolves all requested axis fields in those cases: **71/73 fields**. The field total counts only fully verified cases; unresolved cases may contain unaccepted partial candidates.

One canonical migrated rule is evaluated per in-scope case. Every eligible original RHS axis field is `auto`; extent, coordinate, and other fields stay supplied. There is no explicit-baseline comparison, descriptor erasure, partial-auto search, or legacy non-axis inference. Original source conditions are retained.

## Measured grouping

```text
209 XLA rewrite cases
├─ 112 within the supported evaluation scope
│  ├─ 75 with no RHS attributes
│  ├─ 5 with only supplied non-axis attributes
│  └─ 32 axis-inference cases
│     ├─ 30 inferred and verified
│     └─ 2 unresolved
└─ 97 outside the supported evaluation scope
```

## Outcomes

| Task | Cases | Verified success |
|---|---:|---:|
| Axis inference | 32 | 30/32 (93.8%) |
| No RHS attributes | 75 | 61/75 (81.3%) |
| Supplied non-axis attributes only | 5 | 4/5 (80.0%) |
| All in-scope cases | 112 | 95/112 (84.8%) |

Success requires an explicit semantic proof with `additional-guard: true`, all requested original holes resolved, successful executable compilation, and validated JSON without unresolved holes. Any correct inferred assignment is accepted. An unproved rule is unresolved, not evidence of infeasibility. The negative control is checked separately and must fail.

## Normalization and implementation scope

Unit-only reshape is normalized by a shape-directed adapter to insertion broadcast and singleton-axis removal. Adapter-generated axis positions are fixed; they are not counted as synthesis targets. The removal adapter preserves element types without adding HLO reduce_sum's numeric-dtype guard. Every accepted reshape has a known ordered input shape and target shape with identical non-unit coordinates, independently of automatic axis choices.

The audit found three additional axis cases and one verification-only case through unit normalization. `catalog_dot_drop_unit_dims` is excluded: once the dot attributes are automatic, some legal intermediate shapes would require non-unit splitting/merging in the final reshape. Its original handwritten attributes alone do not justify normalizing the complete inference domain. This revises the earlier estimate of 33 axis cases to 32.

`catalog_broadcast_divide_broadcast` has no nonzero guard in the measured rule. Its rank-zero broadcast has a forced empty mapping and is reported separately from nontrivial axis choices. The reciprocal normalization requires exact-real arithmetic and a nonzero divisor.

## Unresolved axis cases

| Case | Reason |
|---|---|
| catalog_broadcast_divide_broadcast | normalized coefficients, tensor accesses, or reduction domains differ |
| catalog_broadcast_transpose_broadcast | metadata definedness: comprehension is outside the pointwise metadata fragment |

Forced-only tasks: **3**; verified: **2**. Nontrivial tasks: **29**; verified: **28**.

## Reproduction and evidence

```sh
bazel build //:tepl
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/synthesis_cases.py
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/run_synthesis.py --jobs 4 --timeout 30
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/synthesis_report.py
```

[Classification](../data/synthesis_classification.json), [measurements](../results/synthesis.json), [per-case outcomes](synthesis_cases.md), and [canonical axis tasks](../../tepl/rules/axis/) record source/task hashes, normalization metadata, requested fields, diagnostics, compile/validation evidence, search-path labels, and timings.

- Run UTC: `2026-10-28T14:57:32.506323+00:00`.
- Compiler SHA-256: `b4e048f0728ce2fd138b71a1c1c5e6b5dbe5fe9e19ea3f70943a75fa9b14efaa`.
- Budget: 30 seconds per compiler command; 4 workers.
- Arithmetic policy: ideal `algebraic`; no strict IEEE/backend equivalence claim.
