# Native semantic solver benchmarks

[REPORT.md](REPORT.md) compares the initial compiled constraint implementation
with repeated-factor matching, singleton elimination, and term normalization.
The raw measurements are [before.csv](before.csv) and [after.csv](after.csv).
[before.json](before.json) and [after.json](after.json) identify the source files
used for those runs by SHA-256. The before measurements are a historical snapshot;
the current binary implements the extended path.

## Run the current comparison

```sh
bazel build -c opt //:semantic_benchmark
bazel-bin/semantic_benchmark --repeats 3 --warmup 1 --timeout-ms 500 > /tmp/current.csv
python3 tools/summarize_semantic_benchmark.py \
  examples/semantic/benchmarks/before.csv /tmp/current.csv > /tmp/report.md
```

Run without other builds or CPU-intensive jobs. The same CSV settings and input
metadata must be used for both sides of a before/after comparison. The summarizer
rejects mismatched settings for shared cases. To measure a future implementation
change, record a fresh baseline before editing and use it as the first argument.

`metadata` and `compiled` modes disable ordered inference and source-copy/identity
preferences. They use identical Z3 seeds, time/work budgets, and candidate limits.
`production` uses the normal entry point, including ordered construction and
preferences. Its seed stays at the production default of zero, and it has no
additional candidate-count limit. CSV rows record the actual settings.

`compiled` now enables axis-domain propagation. `unreduced` uses the same compiled
semantic matcher with propagation disabled. Both disable ordered construction and
preferences. Use these modes to isolate preprocessing rather than comparing it
against metadata-only search. [AXIS_PROPAGATION.md](AXIS_PROPAGATION.md) records
rank 10, 32, 100, and 200 measurements, including native propagation and residual
variable counters. The plan and runtime request format remain unchanged.

```sh
bazel-bin/semantic_benchmark --mode compiled --filter axis_ \
  --repeats 3 --warmup 1 --timeout-ms 2000 --budget 10000000 > /tmp/reduced.csv
bazel-bin/semantic_benchmark --mode unreduced --filter axis_ \
  --repeats 3 --warmup 1 --timeout-ms 2000 --budget 10000000 > /tmp/unreduced.csv
```

```sh
bazel-bin/semantic_benchmark --mode compiled --filter lora \
  --repeats 5 --warmup 2 --timeout-ms 2000 --budget 1000000 \
  --max-candidates 64 --seed 0 > /tmp/lora.csv
```

The measurements cover the native solver used by generated Rust. They exclude
graph search, descriptor construction, and ABI serialization. Parsing and
analysis run once per distinct rule; `compile_ms` includes both and is excluded
from matching time. Phase times overlap because metadata construction includes
source-legality Z3 calls. Candidate counters concern generic verification;
ordered construction reports its method and elapsed time instead.

Keep unknown results and fallback reasons when interpreting performance. A time
ratio is meaningful only for comparable conclusive results. The report exposes
both improvements and construction overhead. No correctness test asserts timing.
