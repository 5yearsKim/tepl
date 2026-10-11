# Axis-domain propagation measurements

These measurements compare the same generic native solver with propagation enabled
(`compiled`) and disabled (`unreduced`). Both use compiled semantic constraints;
ordered construction and candidate preferences are disabled. Each case has three
measured repetitions after one warmup, a 2000 ms timeout, work budget 10,000,000,
candidate limit 64, and Z3 seed zero. The binary was built with `bazel build -c opt`.
Runs were sequential with no concurrent builds or tests. Timings include native
propagation, formula construction, solving, and verification. They exclude graph
matching, Rust descriptor construction, and ABI serialization. Compilation runs
once per distinct rule and is excluded from match time. Source/binary fingerprints
and toolchain details are recorded in [axis_provenance.json](axis_provenance.json).

Permutation fixtures reverse an equal-extent tensor. Other fixtures use one
high-rank operand with matrix weights and no batching. These are useful cases
with mostly forced attributes, not a coverage claim for arbitrary rank-100 inputs.

| Case | Unreduced result | Unreduced ms | Propagated result | Propagated ms | Propagation ms | Residual entries | Z3 calls |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| axis_permutation_r10 | found | 6.32 | found | 5.27 | 0.13 | 0 | 0 |
| axis_distribution_r10 | found | 312.16 | found | 12.67 | 3.71 | 0 | 0 |
| axis_association_r10 | found | 252.55 | found | 14.31 | 5.22 | 0 | 0 |
| axis_lora_r10 | unknown | 2062.23 | found | 19.18 | 8.23 | 0 | 0 |
| axis_permutation_r32 | found | 37.15 | found | 7.66 | 0.63 | 0 | 0 |
| axis_distribution_r32 | unknown | 2432.02 | found | 30.15 | 14.49 | 0 | 0 |
| axis_association_r32 | unknown | 2198.52 | found | 30.21 | 15.09 | 0 | 0 |
| axis_lora_r32 | unknown | 2309.84 | found | 54.82 | 35.70 | 0 | 0 |
| axis_permutation_r100 | found | 332.87 | found | 19.39 | 5.90 | 0 | 0 |
| axis_distribution_r100 | unknown | 2015.80 | found | 133.62 | 93.39 | 0 | 0 |
| axis_association_r100 | unknown | 2013.20 | found | 97.28 | 65.96 | 0 | 0 |
| axis_lora_r100 | unknown | 2020.68 | found | 297.67 | 255.57 | 0 | 0 |
| axis_permutation_r200 | found/unknown | 2085.15 | found | 47.85 | 25.86 | 0 | 0 |
| axis_distribution_r200 | unknown | 2030.95 | found | 436.37 | 375.25 | 0 | 0 |
| axis_association_r200 | unknown | 2030.52 | found | 288.97 | 235.74 | 0 | 0 |
| axis_lora_r200 | unknown | 2029.59 | found | 1192.06 | 1122.43 | 0 | 0 |

All propagated fixtures returned verified witnesses with zero residual attribute
entries and zero Z3 calls. The generic verifier still uses Z3 expressions for
symbolic equality; zero calls means no solver search, not zero use of the Z3 library.
Unreduced permutation cases can succeed, but their formula construction grows
substantially with rank; some rank-200 repetitions time out. Most larger contraction cases exhaust their construction
budget. Do not interpret `Unknown` as a proof that no attributes exist, or report
a conclusive speedup ratio against an unknown result. A timeout can be exceeded
by an individual expression simplification or construction step.

Propagation itself remains rank dependent and can be costly. The rank-200 LoRA
case takes over one second. Highly batched inputs, unresolved couplings, repeated
captures, and ambiguous term matching can still need SMT or return `Unknown`.

Raw data: [axis_propagation.csv](axis_propagation.csv),
[axis_unreduced.csv](axis_unreduced.csv). Reproduction commands are in
[README.md](README.md). These measurements are separate from the earlier
historical [REPORT.md](REPORT.md).
