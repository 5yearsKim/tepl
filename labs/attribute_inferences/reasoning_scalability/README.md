# Main scaling experiment

This is TEPL's main scalability experiment. It measures the production automatic metadata-DAG constructor
(`--core`, algebraic policy) after the match-time feasibility migration. All RHS
attribute descriptors are automatic. There are no explicit/partial comparison
curves and no guard-certification stage.

The final figure has **three horizontally aligned panels**, each with stacked
phase means and one black mean-total line. The line equals the stack height;
whiskers show the minimum and maximum repeated total. Axes start at zero. Sizes
are equally spaced categories (every series doubles between ticks).

| Panel | Independent variable | Fixed controls | Measured phases |
|---|---|---|---|
| (a) Dot-chain scaling | 1, 2, 4, 8, 16, 32, 64 dots per side | Symbolic rank; left-associated source, right-associated replacement | Preparation; checking/construction; lowering/validation; other (ms) |
| (b) Additive-term scaling | 1, 2, 4, 8, 16, 32, 64 independent terms | Symbolic rank; two dots per term; balanced addition trees; disjoint captures | Same compilation phases and colors as (a) |
| (c) Runtime rank scaling | Ranks 2, 4, 8, 16, 32, 64, 128, 256 | One compiled two-dot reassociation rule; one contraction axis and rank−2 batch axes; all extents 2 | Feasibility checking; attribute construction; other (µs) |

## Phase boundaries

Every compilation is isolated in a worker with a **hard 10-second wall-clock timeout**,
including parsing and checking. Ascending preflight stops each series at its first
failure; larger sizes in that panel are not attempted. A later failed repetition
also stops that case and larger sizes, excluding the whole affected cases from
the figure. Runtime rank scaling stops at its first metadata resource/feasibility
failure. Failed/stopped sizes and diagnostics remain in the raw data and report.

Compilation uses fresh production stages for every repetition:

- **Preparation:** parsing, import loading, resolution and frontend type/schema
  checking. Workload source text is preloaded; import reads are included.
- **Checking/construction:** `semantics::verify`, including structural/dtype
  admission, conditional soundness checks, constructor emission and materialization.
- **Lowering/validation:** `lowering::lower`, including executable validation and
  cached metadata-program creation.
- **Other:** the total interval minus those disjoint subintervals, including
  stage dispatch, moving outputs and intermediate representation teardown.

This measures compilation, including the frontend, rather than repeatedly
calling `proveRule` on a prepared plan. Timed construction does not prove C⇒q.
Compilation workloads use `f32[Batch...,Dleft,Dright]`, with an unbounded
shared batch prefix. Source batching lists equal `range(len(Batch))`; left/right
contraction positions are `[len(Batch)+1]` and `[len(Batch)]`. Positivity is
preserved for **every** batch and matrix extent, including the empty batch
prefix case. No concrete rank or batch-prefix length is written into these
rules. Each input has exactly two non-batch axes: this is a rank-polymorphic
batched-matrix family, not an experiment over unrestricted axis layouts.

For example, the two-dot source takes `A1:f32[Batch...,D0,D1]`,
`A2:f32[Batch...,D1,D2]`, `A3:f32[Batch...,D2,D3]`. Its replacement is
`dot A1 (dot A2 A3)` with fully automatic attributes. Panels (a)/(b) compile
this **single symbolic rank family**, rather than cloning a rule per rank.

Runtime compiles **exactly once for all ranks**, outside timing. The rank-independent
rule is `(dot (dot A B) C) => (dot A (dot B C))`, with C=true. Each execution calls
`executeSound` on the cached program. Its optional instrumentation surrounds:

- **Feasibility:** evaluating all q checks, after source legality replay.
- **Attribute construction:** evaluating f, building attributes/layouts and
  packaging the constructor results. This starts only after all q checks pass.
- **Other:** source legality replay, interpreter setup, surrounding allocations,
  output teardown, outer-loop bookkeeping and clock overhead. Allocations inside
  q/f remain charged to those phases.

The q and f intervals belong to **the same execution**, sharing the interpreter
cache. Separate predicate/build calls and subtraction of unrelated measurements
would double-count work and are not used. No loading, synthesis, proof search,
graph matching or tensor-value execution occurs inside runtime timing. Native
metadata limits are 1,000,000 work steps/items. This is the metadata VM cost,
not end-to-end XLA graph-rewrite latency.

Runtime samples average calibrated batches targeting ~10 ms (1–2,000 calls).
Worker startup, fork/IPC and runtime setup serialization/loading are outside
phase timing. A runtime rule is compiled once in its worker, loaded once and then
reused for every rank. Three warmup rounds and 20 measured rounds are the defaults. Matched unprofiled
batches are also saved to expose instrumentation overhead; no overhead is
subtracted. Whiskers describe variation between batch averages, not individual
call latency. After the ascending preflight, compilation and rank cases are shuffled serially each round with
seed 20261104. Do not run builds or other benchmarks concurrently with measurement.

## Correctness and selection

Independent dense integer NumPy checks use Python-integer object arrays to avoid
fixed-width overflow in long chains. They compare source/replacement chains and
additive sums with matrix-multiplication oracles. Outside runtime timing, plain
and profiled executions must agree on attributes, layouts and work counts; an
independent positional RHS replay checks legality and axis correspondence at
**every successful runtime rank**. Every successful compilation in (a)/(b) is
separately executed at ranks **2, 3, and 5**, using the same compiled program,
anisotropic matrix extents, and native guard matching/evaluation as well as
independent RHS replay. These checks are outside phase timing. Dense reference
checks additionally cover batch-prefix shapes `[]`, `[2]`, and `[2,3,2]`. No exponentially large tensors are allocated for high ranks.
Numerical reference checks validate the selected concrete workloads; they do not
establish arbitrary tensor equivalence.

Only cases whose warmups **and every measured repetition** succeed are plotted.
Failed compilation outcomes remain in the raw data and report. A mixed case is
excluded as a whole; slow successful repetitions are never discarded. Failed
correctness controls abort the run and preserve the previous result. Phase sums,
finite/nonnegative times, complete repetition counts and input/binary hashes are
validated before publishing results.

## Reproduction

From the repository root:

```sh
python3 -m venv /tmp/tepl-scaling-venv
/tmp/tepl-scaling-venv/bin/pip install -r labs/attribute_inferences/reasoning_scalability/requirements.txt
bazel build -c opt //labs/attribute_inferences/reasoning_scalability:inference_benchmark
bazel test -c opt //labs/attribute_inferences/reasoning_scalability:timing_test
/tmp/tepl-scaling-venv/bin/python -m unittest discover -s labs/attribute_inferences/reasoning_scalability/tests
/tmp/tepl-scaling-venv/bin/python labs/attribute_inferences/reasoning_scalability/scripts/benchmark.py
/tmp/tepl-scaling-venv/bin/python labs/attribute_inferences/reasoning_scalability/scripts/benchmark.py --check
```

`--prepare` generates inputs/reference checks only; `--report` rebuilds the figure
from validated results. `--warmups`, `--repetitions`, and `--binary` are configurable.
The optimized benchmark must be built before running. Schema v4 rejects the earlier rank-specialized and old
compilation-only/rank-proof data; it cannot supply runtime phase bars.

The current run's artifacts are Git-ignored under **`output/`**:

- `figures/scalability.png`, `.pdf`, `.svg`: the horizontal (a), (b), (c) figure.
- `performance.json`: all raw samples, calibration counts, outcomes and provenance.
- `performance_summary.json`: phase means, total means, ranges and exclusions.
- `performance.md`: protocol, outcomes, timings and observer-overhead comparison.
- `inputs/`: generated full-auto workloads and manifest.

There is one workload generator, one measurement driver, and one figure generator.
Running the experiment replaces the current artifacts after validation; copy
`output/` elsewhere before running if a separate snapshot is needed.
