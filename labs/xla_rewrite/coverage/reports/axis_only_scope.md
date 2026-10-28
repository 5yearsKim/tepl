# Axis-Only Evaluation Protocol

This protocol replaces the old descriptor-inference experiment. The
[current report](attribute_synthesis.md) contains the measured counts; earlier
112/97 and 33-axis-case proposals were estimates, not frozen target results.

## Scope and migration

Keep all 209 original case IDs, count each once, and check the negative control
separately. Publish one **outside the supported evaluation scope** category;
retain exclusion reasons only in the per-case audit data.

The selected scope excludes slice, split/merge reshape, convolution, cumulative
sum, sort, and tensor-index gather/scatter. Checked unit-only reshape
normalization is allowed. Other missing semantic capabilities remain outside.
Classify cases with no RHS attributes, supplied non-axis attributes only, or
eligible original RHS axis fields separately.

There is one canonical task per in-scope case in `tepl/rules/axis/`:

- Replace every eligible original RHS `axis`/`axis[]` field with `auto`.
- Preserve all supplied extent, coordinate, and general expressions and defaults.
- Expand copied descriptors into field providers; remove unused handwritten
  axis formulas, retaining helpers needed by supplied fields.
- Reject supplied-field dependencies on automatic RHS axes rather than retaining
  the old answer as a hidden pin.

No explicit-baseline variant, whole-descriptor erasure, partial-subset search,
or legacy non-axis inference is part of this experiment. Original explicit files
remain only as source fixtures and provenance.

## Checked normalization and theorem classification

Infer unit-view adapters from known ordered input/target shapes and direct
source-condition conjuncts. Remove singleton dimensions and insert supplied
unit dimensions while preserving non-unit coordinates in order. Generated axis
positions are fixed, not extra inference holes. The removal helper preserves
all element types; using HLO reduce_sum directly would incorrectly add its
numeric-dtype guard.

Normalization must be valid for every legal original axis completion. In
particular, a reshape after an automatic dot cannot be accepted just because
its original handwritten dot attributes produced a unit-only view. Unknown
intermediate shapes are conservatively excluded. Disjunctions and negations
are not treated as positive unit-view premises.

Record structural theorem candidates before execution. Unary linear operations
may surround separated sums, with one common axis choice across their
components. The independent scalar-reciprocal normalization is a restricted
candidate under exact-real nonzero-divisor semantics. General division is not
admitted. The current scalar-division rule has no nonzero guard, so its original
value domain is not fully certified by this theorem route.

Positive extents, supplied-field definedness, read-once components, distinct
capture sets, original legality, and the arithmetic domain remain explicit
admission obligations. Structural candidacy is not a full-domain completeness
claim or a claim that the production compiler implements the complete
mathematical state-table algorithm. Preserve the original conditions in the
main run; any extra assumptions require separately labeled companion tasks.

## Execution and acceptance

Run every in-scope canonical task under the same compiler, algebraic policy,
and per-command timeout. Cases without axis targets are verification-only.
A successful axis case must resolve every requested original hole, preserve
supplied fields, prove the materialized rule without unreported guards,
compile to executable JSON, and pass artifact validation. Any valid inferred
assignment counts. Partially filled cases do not count as successful cases or
resolved fields in the primary totals.

Report case-level inference success as the primary metric, field-level totals
secondarily, verification-only outcomes separately, and forced versus
nontrivial axis assignments separately. An empty scalar-broadcast mapping is
forced. Unproved cases, timeouts, invalid migrated tasks, and implementation
gaps are not infeasibility certificates. The negative control must fail.

Freeze hashes of original and canonical sources, classifier, normalizers,
runner, compiler, and proof before execution. The runner rejects input changes
during measurement. Archive old measurements and publish fresh results with
commands, diagnostics, proof evidence, timings, compile/validation evidence,
and artifact hashes. Reproduction commands are in the [coverage README](../README.md).
