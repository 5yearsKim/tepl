# Semantic constraint benchmark

Recorded 2026-10-11T17:16:42.979581+00:00. Host: Linux x86_64.

CPU: Intel(R) Core(TM) Ultra 7 155H.

Z3: 4.15.2.0; compiler: 15.2.0 (optimized). Timeout: 500 ms; work budget: 1000000; controlled candidate limit: 64; seed: 0.

Times are medians of repeated runs, in milliseconds. Rules are parsed and analyzed once per distinct rule per invocation; compilation is excluded from matching time. Both controlled modes disable ordered inference and candidate preferences. Production uses normal dispatch and has no separate candidate-count cap. These measure the native solver, excluding Rust graph matching and ABI serialization.

Metadata time includes source-legality Z3 calls; phase times overlap. Unknown outcomes include resource limits and are retained in the tables. Timing ratios are shown only when both modes consistently return the same conclusive status. This is a local benchmark, not a universal performance guarantee.

## Constraint-generation extensions

| Case | Before result | After result | Before ms | After ms | Before/after solver calls | Before/after rejected candidates | Before/after fallback | After method |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| transpose_r2 | found | found | 21.967 | 21.659 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| distribution_r2 | found | found | 34.924 | 34.801 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| association_r2 | found | found | 107.745 | 107.014 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| lora_r2 | found | found | 121.988 | 130.912 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| transpose_r3 | found | found | 29.246 | 28.036 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| distribution_r3 | found | found | 51.226 | 50.001 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| association_r3 | found | found | 231.710 | 190.227 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| lora_r3 | found | found | 319.055 | 270.489 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| transpose_r5 | found | found | 74.418 | 71.258 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| distribution_r5 | found | found | 161.019 | 134.498 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| association_r5 | unknown | unknown | 609.153 | 644.659 | 1/1 | 0/0 | 0%/0% | z3 |
| lora_r5 | unknown | unknown | 557.297 | 528.439 | 1/1 | 0/0 | 100%/100% | z3 |
| distribution_extent0 | found | found | 26.072 | 25.513 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| unused_extent0 | found | found | 2.414 | 1.959 | 3/3 | 0/0 | 100%/100% | z3 |
| distribution_extent1 | found | found | 27.535 | 28.395 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| unused_extent1 | found | found | 2.592 | 2.210 | 3/2 | 0/0 | 100%/0% | constraints-z3 |
| distribution_extent2 | found | found | 28.185 | 28.789 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| unused_extent2 | invalid | invalid | 3.188 | 2.393 | 4/4 | 1/1 | 100%/100% | z3 |
| multi_axes | found | found | 89.680 | 93.926 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| equal_extents | found | found | 53.411 | 54.042 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| pinned_cycle | found | found | 23.086 | 26.636 | 2/2 | 0/0 | 0%/0% | constraints-z3 |
| ordering_impossible | invalid | invalid | 20.227 | 22.792 | 3/3 | 0/0 | 100%/100% | z3 |
| drop_impossible | invalid | invalid | 6.946 | 7.419 | 4/4 | 1/1 | 100%/100% | z3 |
| repeated_symmetry | found | found | 3.887 | 5.255 | 2/2 | 0/0 | 100%/0% | constraints-z3 |
| repeated_asymmetric | invalid | invalid | 12.714 | 15.477 | 3/4 | 1/1 | 100%/100% | z3 |
| unit | found | found | 2.713 | 2.205 | 3/2 | 0/0 | 100%/0% | constraints-z3 |
| dynamic_rhs_unit | found | found | 2.417 | 2.197 | 3/2 | 0/0 | 100%/0% | constraints-z3 |
| alpha_terms | found | found | 2.827 | 2.178 | 3/2 | 0/0 | 100%/0% | constraints-z3 |

## Current metadata versus semantic constraints

| Case | Metadata result | Compiled result | Metadata ms | Compiled ms | Metadata/compiled time ratio | Construction ms | Z3 ms | Verification ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transpose_r2 | found | found | 13.265 | 21.659 | 0.61 | 8.060 | 3.311 | 2.442 |
| distribution_r2 | found | found | 18.484 | 34.801 | 0.53 | 13.910 | 5.375 | 3.580 |
| association_r2 | found | found | 69.409 | 107.014 | 0.65 | 60.946 | 15.887 | 4.570 |
| lora_r2 | found | found | 89.897 | 130.912 | 0.69 | 66.723 | 22.592 | 5.797 |
| transpose_r3 | found | found | 63.285 | 28.036 | 2.26 | 11.055 | 4.970 | 2.772 |
| distribution_r3 | found | found | 241.414 | 50.001 | 4.83 | 19.913 | 10.124 | 4.149 |
| association_r3 | unknown | found | 503.926 | 190.227 | — | 116.939 | 27.323 | 5.581 |
| lora_r3 | unknown | found | 504.709 | 270.489 | — | 151.796 | 45.569 | 7.419 |
| transpose_r5 | unknown | found | 502.806 | 71.258 | — | 31.198 | 13.587 | 5.749 |
| distribution_r5 | unknown | found | 504.355 | 134.498 | — | 58.298 | 28.110 | 8.683 |
| association_r5 | unknown | unknown | 505.566 | 644.659 | — | 506.514 | 0.091 | 0.000 |
| lora_r5 | unknown | unknown | 506.026 | 528.439 | — | 282.616 | 0.046 | 0.000 |
| distribution_extent0 | found | found | 11.363 | 25.513 | 0.45 | 11.398 | 4.866 | 0.138 |
| unused_extent0 | found | found | 1.916 | 1.959 | 0.98 | 0.320 | 0.051 | 0.012 |
| distribution_extent1 | found | found | 14.743 | 28.395 | 0.52 | 11.583 | 4.892 | 2.973 |
| unused_extent1 | found | found | 2.271 | 2.210 | 1.03 | 0.293 | 0.061 | 0.185 |
| distribution_extent2 | found | found | 52.459 | 28.789 | 1.82 | 11.976 | 5.053 | 2.999 |
| unused_extent2 | invalid | invalid | 2.277 | 2.393 | 0.95 | 0.299 | 0.066 | 0.330 |
| multi_axes | unknown | found | 503.521 | 93.926 | — | 37.524 | 21.709 | 6.786 |
| equal_extents | unknown | found | 502.859 | 54.042 | — | 22.056 | 10.987 | 4.691 |
| pinned_cycle | found | found | 12.829 | 26.636 | 0.48 | 10.297 | 4.610 | 2.941 |
| ordering_impossible | invalid | invalid | 9.434 | 22.792 | 0.41 | 10.691 | 4.429 | 0.000 |
| drop_impossible | invalid | invalid | 6.944 | 7.419 | 0.94 | 0.163 | 0.369 | 4.200 |
| repeated_symmetry | found | found | 3.825 | 5.255 | 0.73 | 1.357 | 0.235 | 1.379 |
| repeated_asymmetric | invalid | invalid | 13.503 | 15.477 | 0.87 | 2.102 | 0.366 | 9.851 |
| unit | found | found | 2.331 | 2.205 | 1.06 | 0.272 | 0.162 | 0.164 |
| dynamic_rhs_unit | found | found | 1.971 | 2.197 | 0.90 | 0.376 | 0.236 | 0.193 |
| alpha_terms | found | found | 2.103 | 2.178 | 0.97 | 0.344 | 0.083 | 0.234 |

## Production dispatch

| Case | Result | Method | Match ms | Ordered admission/construction ms | Fallback |
| --- | --- | --- | ---: | ---: | --- |
| transpose_r2 | found | ordered | 0.520 | 0.520 | 0% |
| distribution_r2 | found | constraints-z3 | 34.977 | 0.041 | 0% |
| association_r2 | found | ordered | 0.961 | 0.961 | 0% |
| lora_r2 | found | constraints-z3 | 142.667 | 0.396 | 0% |
| transpose_r3 | found | ordered | 0.380 | 0.380 | 0% |
| distribution_r3 | found | constraints-z3 | 49.958 | 0.034 | 0% |
| association_r3 | found | ordered | 0.688 | 0.688 | 0% |
| lora_r3 | found | constraints-z3 | 262.783 | 0.258 | 0% |
| transpose_r5 | found | ordered | 0.403 | 0.403 | 0% |
| distribution_r5 | found | constraints-z3 | 135.394 | 0.029 | 0% |
| association_r5 | found | ordered | 0.770 | 0.770 | 0% |
| lora_r5 | unknown | z3 | 586.754 | 0.290 | 100% |
| distribution_extent0 | found | constraints-z3 | 24.882 | 0.014 | 0% |
| unused_extent0 | found | z3 | 2.361 | 0.011 | 100% |
| distribution_extent1 | found | constraints-z3 | 29.967 | 0.016 | 0% |
| unused_extent1 | found | constraints-z3 | 2.126 | 0.009 | 0% |
| distribution_extent2 | found | constraints-z3 | 28.262 | 0.027 | 0% |
| unused_extent2 | invalid | z3 | 2.214 | 0.033 | 100% |
| multi_axes | found | constraints-z3 | 95.961 | 0.030 | 0% |
| equal_extents | found | constraints-z3 | 57.704 | 0.026 | 0% |
| pinned_cycle | found | constraints-z3 | 25.790 | 0.000 | 0% |
| ordering_impossible | invalid | z3 | 22.520 | 0.000 | 100% |
| drop_impossible | invalid | z3 | 7.599 | 0.309 | 100% |
| repeated_symmetry | found | constraints-z3 | 5.548 | 0.000 | 0% |
| repeated_asymmetric | invalid | z3 | 16.676 | 0.000 | 100% |
| unit | found | constraints-z3 | 2.085 | 0.018 | 0% |
| dynamic_rhs_unit | found | constraints-z3 | 2.187 | 0.000 | 0% |
| alpha_terms | found | constraints-z3 | 2.259 | 0.025 | 0% |
