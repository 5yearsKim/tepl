# Reasoning cost and scalability experiment plan

Measure how TEPL checking cost changes as tensor rewrite compositions become
deeper and normalized expressions become larger. Use two generated benchmark
families: batched dot-chain reassociation and products of sums. This is a
controlled scaling experiment, complementary to the
[XLA coverage evaluation](../../labs/xla_rewrite/README.md) and
[attribute inference evaluation](../../labs/attribute_inference/README.md).

This document specifies the experiment and result figures. The generator and
runner are in [scripts/benchmark.py](scripts/benchmark.py). Saved measurements
and their analysis are described in [reports/results.md](reports/results.md).

The later [axis normalization fix](reports/axis_normalization_fix.md) explains
why registered coordinate views prevented longer dot chains from matching.
Its [updated scalability figure](reports/canonical_scaling_axis_fix.md) uses
the corrected checker and preserves the earlier measurements separately.

## Questions and scope

1. How do checking time and memory grow with contraction-chain length?
2. How does asking for RHS attribute inference change the cost and success
   boundary compared with explicit RHS attributes?
3. Does varying physical axis order change those trends?
4. How does algebraic expansion affect cost, and where do fixed proof budgets
   prevent completion?

These workloads characterize regular contraction chains and polynomial
expansion. They do not establish scalability for arbitrary tensor graphs,
arbitrary overlapping axis groups, or all tensor ranks. In particular, varying
axis positions does not by itself create more complex logical overlap.

## Experiment A Dot chain reassociation

For `n` distinct input tensors, check this single composite rewrite:

```text
(((A1 dot A2) dot A3) dot ... dot An)
    =>
A1 dot (A2 dot (A3 dot ... dot An))
```

This checks the whole equation at once. It does not execute a sequence of
already verified local rewrites.

Each input has logical axes `[b, d(i-1), di]`:

```text
A1: [B, D0, D1]
A2: [B, D1, D2]
...
An: [B, D(n-1), Dn]
output: [B, D0, Dn]
```

Every dot preserves batch axis `b` and contracts the adjacent shared `di` axis.
Keep all tensors rank three, with symbolic positive extents. Use the same dtype
and the `algebraic` semantic policy throughout. The equation is interpreted
under that policy, not as a claim of bitwise floating-point equality.

Test `n = 3, 4, 6, 8, 12, 16, 24, 32`. Each side contains `n-1` dots and has
maximum dot nesting depth `n-1`. Each dot sums over one axis; the composed
computation sums over `n-1` distinct contraction axes. Increasing numerical
dimension sizes is unnecessary because tensor execution is not being measured.

### Axis layouts

Generate two deterministic layouts for every chain length:

- **Canonical:** each input has physical order `[b, left, right]`.
- **Permuted:** assign input `Ai` entry `(i-1) mod 6` from this ordered list:
  `[b,left,right]`, `[b,right,left]`, `[left,b,right]`,
  `[left,right,b]`, `[right,b,left]`, `[right,left,b]`.

Both layouts represent exactly the same logical contraction network. Do not
insert extra transpose operations. Generate dot attributes by locating logical
batch and contraction labels in each operand's physical axis list. Propagate
each intermediate's output order according to the dialect's dot definition;
do not assume intermediate axes inherit the input permutation. Both roots must
have ordered output axes `[b,d0,dn]`.

The permuted variant tests axis-position bookkeeping. Its pattern is fixed
before measurement and is not selected according to proof success.

### Attribute modes

For each chain length and layout, generate a matched pair:

| Mode | LHS attributes | RHS attributes |
| --- | --- | --- |
| Explicit | Fully specified | Fully specified by the label-based generator |
| Automatic | Identical to explicit mode | Every RHS dot descriptor is `auto` |

Use the same shapes, graph structures, applicability conditions, and policy
in both modes. Derive explicit attributes independently of TEPL's inference.
Any constraints needed to describe this chain family belong in both versions.
Do not add case-specific conditions after a failure to improve reported results.

Fixed ranks and layouts make this a bounded attribute-inference workload, not a
demonstration of arbitrary-rank constructor synthesis. Canonical attributes are
particularly repetitive; report the permuted results alongside them.

There are `8 lengths × 2 layouts × 2 modes = 32` cases.

## Experiment B Products of sums

For `k = 2, 3, ..., 10`, generate:

```text
(A1 + B1) * (A2 + B2) * ... * (Ak + Bk)
    =>
(Ak + Bk) * ... * (A2 + B2) * (A1 + B1)
```

Use distinct captures, identical rank-two symbolic shapes, and elementwise
addition and multiplication. Use left-associated multiplication trees on both
sides and reverse factor order on the RHS. Use the same dtype and algebraic
policy as Experiment A. There are no attributes to infer in this family.

Each side contains `k` additions and `k-1` multiplications. Both source trees
grow linearly, but full distributive normalization produces `2^k` distinct
terms. Keep both sides factored in the input; an explicitly expanded RHS would
confound internal expression growth with exponentially growing source size.

| Factors k | Terms after full expansion |
| ---: | ---: |
| 2 | 4 |
| 4 | 16 |
| 6 | 64 |
| 8 | 256 |
| 9 | 512 |
| 10 | 1024 |

The current [proof policy](../../src/core/semantics/proof/proof_policy.h)
sets `normalization_terms` to 256. A normalization-term failure is therefore
expected at nine factors if execution reaches that check. Another budget or
unsupported reasoning path may intervene earlier. Record the actual diagnostic;
do not substitute the predicted outcome for a measurement.

There are nine cases. This family isolates expression expansion rather than
axis-partition complexity.

## Measurement procedure

1. Freeze the generated cases and compiler build before measuring. Record the
   source revision, dirty-tree patch or source hashes, compiler binary hash,
   build configuration, dialect hash, machine, OS, and proof-limit values.
2. Validate generation independently: check contraction label routing, matching
   extents, root axis order, and expected source node counts. Use a small concrete
   reference evaluation to catch generator mistakes. This is a generator check,
   not a replacement for semantic verification.
3. Run each case once as a warm-up, then five measured times in fresh processes.
   Use one generated rule per file and invoke the prebuilt checker directly,
   without build time or graph execution. Run serially with a fixed recorded
   shuffle of case order for each repetition.
4. Set a 30-second wall-clock timeout per invocation. Keep all internal proof
   budgets unchanged across cases and modes. Retain failures at every size;
   do not stop the sweep at the first failure.
5. Measure elapsed wall-clock time using a monotonic clock and peak resident
   memory for that individual child process, for example with GNU `time`.
   Avoid cumulative child-process memory statistics. Capture stdout and stderr.
6. Keep profiling out of timed runs. The current CLI has no quiet semantic-report
   mode: use `--rule=CASE` consistently and redirect its report to a file.
   Report formatting and file output are included in end-to-end timing. Also
   obtain constructor details in separate diagnostic runs using the same input
   and compiler. Confirm that diagnostic and measured outcomes agree.

The measured workload is `41 cases × 5 repetitions = 205` invocations, plus
41 warm-ups and diagnostic runs. If every measured invocation times out,
the measured portion is bounded by approximately 103 minutes.

The primary timing is **end-to-end rule checking time**: parsing, checking,
construction, and verification. It is not isolated theorem-prover time. The
automatic/explicit timing ratio measures the difference between the two complete
checking paths; it does not identify pure inference time.

### Outcomes and raw records

Store one record per invocation with at least:

```text
case_id, family, size, layout, attribute_mode, repetition,
input_nodes, input_bytes, elapsed_seconds, peak_rss_mib,
outcome, exit_code, diagnostic, compiler_hash, input_hash
```

Use these distinct outcomes:

- `proved`: completed semantic verification; automatic cases also have no
  unresolved RHS attribute holes and no additional unestablished guard.
- `budget_exhausted`: an internal proof limit was reported; retain its kind.
- `unresolved`: another unsupported or unestablished obligation was reported.
- `timeout`: the external 30-second limit terminated the process.
- `error`: a crash, malformed generated case, or unexpected tool failure.

For Experiment B, record `expected_expanded_terms = 2^k` as an analytical input
property. It is not an observed allocation or a measured peak term count.

Record all five outcomes. A case is consistently proved only if all five runs
prove it. Mixed outcomes must be shown rather than reduced to a success median.
Fix generator errors in a versioned corpus and rerun affected comparisons;
compiler crashes remain reported results.

### Optional diagnostic counters

The existing `TEPL_INDEX_PROFILE=1` hook in
[context.cc](../../src/core/semantics/context.cc) reports context time, queries,
position lookups, bindings, cache hits, and other work counters. Preserve each
context separately: nested context times must not be summed as total runtime.

These counters can help interpret a timing curve. Bindings are not a count of
symbolic partitions, and the existing hook does not measure peak normalized
term count. If explaining representation growth becomes necessary, add explicit
per-phase counters for partitions and peak terms in a later instrumentation
step. The initial experiment does not require that extension.

## Results to derive

For each case, derive median time and peak RSS across repetitions, with the
minimum and maximum of each measurement. Peak RSS here means the per-process
peak, summarized across runs; it includes compiler baseline memory.

For each dot layout and mode, report:

- The measured time and memory trend with increasing `n`.
- The largest tested consistently proved case and the first tested failure,
  including its reason. Report later successes if outcomes are nonmonotonic;
  these are sampled boundaries, not exact maximum supported depths.
- The automatic/explicit median-time ratio where both cases consistently prove.
  Report unmatched outcomes separately instead of assigning a ratio.
- The permuted/canonical comparison at matching length and mode, without assuming
  that permutations must make checking more expensive.

For products of sums, report time and memory against `k`, the largest consistently
proved case, and the first observed normalization-budget failure or other limit.
Relate this boundary to `2^k` and the recorded term cap.

The intended result is an empirical account of cost growth and failure boundaries.
There is no required speedup or success threshold. A budget-exhausted case that
returns quickly is not evidence of successful scaling. Do not infer an
asymptotic complexity bound from these few data points.

## Figure specifications

### Figure 1 Dot chain checking cost

Use one figure with two aligned panels sharing the x-axis:

- **X-axis:** number of input tensors `n`, at the actual tested numeric positions.
  State in the caption that each side has `n-1` dot operations.
- **Panel A y-axis:** median end-to-end checking time in seconds, logarithmic scale.
- **Panel B y-axis:** median peak RSS in MiB, linear scale unless the measured
  range warrants a clearly labeled logarithmic scale.
- **Color:** explicit versus automatic attributes, consistent across panels.
- **Line style and marker:** canonical uses solid lines and circles; permuted
  uses dashed lines and squares. Include all four combinations in the legend.
- **Variability:** vertical whiskers show the minimum and maximum over five runs.

Draw successful-cost curves only through consistently proved cases. Break lines
at failures rather than connecting across missing sizes. Add a small outcome
strip below the panels, with a row for each combination and symbols for proved,
budget exhausted, unresolved, timeout, and error. Failure elapsed times and RSS
remain available in the accompanying table; they do not join the success curves.

The caption must state the policy, fixed rank, five repetitions, fixed budgets,
and timeout. It must also identify this as a regular contraction-chain workload.

### Figure 2 Expression expansion and the proof budget

Use two aligned panels with integer factor count `k` on the x-axis:

- **Panel A:** median successful checking time in seconds, logarithmic scale.
- **Panel B:** median peak RSS in MiB, with the same memory-scale convention as
  Figure 1.
- Add a top axis or aligned labels showing the analytical term count `2^k`.
- Mark the first factor count whose analytical expansion exceeds the recorded
  normalization cap. With a cap of 256, draw a vertical dashed line at `k=8.5`.
  Label this as the expected term-budget boundary, not a measured failure point.
- Use min/max whiskers and an outcome strip as in Figure 1. Actual failures may
  occur before the expected boundary and must remain visible.

Do not fit an exponential timing curve merely because term count is exponential.
Do not plot timeouts as completed 30-second proofs or budget failures as zero cost.

### Summary table

Include one row per dot layout/mode and one for products of sums:

| Family | Layout | Attribute mode | Consistently proved cases | Largest proved size | First failed size and reason | Median time at largest proved size | Median peak RSS there |
| --- | --- | --- | --- | --- | --- | --- | --- |

Report automatic/explicit time ratios at common successful sizes in a separate
small table if they clarify the comparison. No rendered figures are required
until measurements exist.

## Planned artifacts and reporting

Use `scripts/` for generation and execution, `inputs/` for frozen TEPL cases,
`results/` for raw records and summaries, and `reports/` for the written analysis
and figures. Keep this README as the experiment specification. Record any
change to the size grid, layouts, budgets, or conditions before comparing runs.

From the repository root, with NumPy available:

```sh
bazel build //:tepl
python3 -m unittest discover -s lab/reasoning_scalability/tests
python3 lab/reasoning_scalability/scripts/benchmark.py
python3 lab/reasoning_scalability/scripts/benchmark.py --check
```

The runner saves after every invocation. Use `--resume` for an interrupted run;
it checks compiler and dependency hashes before continuing. It refuses to replace
an existing run implicitly. `--prepare` generates and numerically validates the
41 inputs without invoking TEPL. The separate analysis script requires Matplotlib.

The report should answer the four questions using measured values and explain
where limits intervene. A suitable paper statement, completed only after running
the experiment, is:

> We measured checking time and memory for batched contraction reassociation
> across eight chain lengths, two axis layouts, and explicit versus inferred
> RHS attributes. A second workload varied compact products of sums to expose
> normalization growth and fixed-budget failures. The results characterize these
> controlled families; arbitrary overlapping axis systems remain future work.
