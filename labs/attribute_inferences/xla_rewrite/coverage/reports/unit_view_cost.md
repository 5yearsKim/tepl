# Unit-view proof cost check

The same 92 canonical tasks succeeded before and after the unit-view
proof fixes. Their median compiler check time was **98.2 ms before**
and **99.0 ms after** (+0.9%).
Newly successful cases are excluded from this unchanged-success cohort.
Compile and artifact-validation time are excluded.

This single-run comparison used four workers; system load was not controlled.
It shows no large overhead in this cohort and is not a formal performance
benchmark or a worst-case complexity guarantee. The new rules traverse axis
positions, shape metadata, and bounded symbolic proof search, never tensor-entry
grids.

[Paired timings and compiler/run identities](../results/unit_view_cost.json)
provide the individual observations. The [core classification](core_classification.md)
reports current native core implementation paths separately.
