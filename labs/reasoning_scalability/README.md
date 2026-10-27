# Attribute inference cost and scalability

Current experiment: a bounded synthetic evaluation of TEPL attribute inference.
Dot-chain lengths are **1–10**. Tensor ranks are **2, 4, 8, 16, 32**.

## Required measurements

1. **Graph-size scaling:** left-to-right reassociation of a contraction chain,
   with 1–10 dot operations per side and 2–11 distinct input tensors. Tensor rank
   stays at three, with inputs ordered as batch, left dimension, right dimension.
   Each size has explicit, partial-auto and full-auto variants. Partial auto
   infers odd RHS nodes; the other fields come from the independent label-based generator.
   Size one is an identity-copy baseline; one descriptor has no proper partial mode.
2. **Rank scaling:** ranks 2, 4, 8, 16, 32 on a base-2 logarithmic x-axis,
   with three source transposes and one replacement
   transpose. Source permutations are independent symbolic attributes. Explicit
   composition and full auto use identical source metadata and conditions.
3. **Phase breakdown:** full-auto chain construction and verification time
   across all strategy attempts, including unsuccessful attempts and fallback. Dispatch/accounting
   time is shown separately. Phase bars use means; latency curves show medians
   with min/max whiskers.

All generated artifacts go in the Git-ignored `output/` directory:

- `output/figures/scalability.pdf`: publication figure with three horizontal
  panels: (a) Chain scaling, (b) Rank scaling, (c) Time breakdown. Arimo text uses
  25-point ticks and legends, 29-point axis labels, and 32-point panel titles.
  Line styles and markers distinguish modes.
- `output/figures/`: matching PNG and SVG versions of the combined figure.
- `output/performance.md`: report with outcomes and timing tables.
- `output/performance.json`: raw samples and provenance.
- `output/performance_summary.json`: per-case timing summaries.
- `output/inputs/`: generated rules, configuration and manifest.

## Measurement protocol

The C++ driver prepares each immutable Plan once, before timing. Every timed
`proveRule` call constructs fresh reasoning contexts. Initial parsing, import
loading, type checking and inspection are excluded. Returned proofs and complete
constructor counts are checked outside the timer. No diagnostics are printed
inside timed calls. Cases are shuffled each round with a fixed seed and run
serially. Only this benchmark replaces the global work-count cap with a
configurable wall-clock deadline: **10 seconds per inference call**, shared
across synthesis strategies, fallback and independent verification. Use
`--timeout-seconds 5` or `--timeout-seconds 20` to change it. Deadline checks are
cooperative, so small overruns are possible. Local search bounds remain active.
Ordinary compiler calls keep their existing 1,000,000-work budget per attempt
and have no added wall-clock deadline.

To keep this first run inexpensive, the default is **one warmup and five measured
repetitions** per case (234 calls for 39 cases). Repetitions can be increased
without changing inputs. All outcomes, including unresolved or time-limited
cases, remain in the raw data. These measure metadata/proof costs rather than
numerical tensor execution. The arithmetic policy is `algebraic`.
Mixed proof/timeout repetitions are retained and marked in the report. At the
default deadline, timed calls can take roughly 39 minutes altogether if every
call reaches its limit, plus preparation and cleanup.

The contraction generator supplies operand axes and explicit
RHS fields. A separate dense integer evaluator validates the generated
contractions against label-based matrix multiplication, and transpose
composition against physical array transposition. Rank reference checks use
small arrays with at most three non-singleton dimensions, moving those
dimensions across random axis positions, so rank 32 needs no huge allocation.
These checks do not use TEPL's normalization or inferred constructor choices. They supplement the
independent semantic verifier used in every measured call.

## Reproduction

Run from the repository root. Python needs NumPy for the independent reference
checks and Matplotlib for plotting; install them in an isolated environment:

```sh
python3 -m venv /tmp/tepl-inference-venv
. /tmp/tepl-inference-venv/bin/activate
python3 -m pip install -r labs/reasoning_scalability/requirements.txt
bazel build -c opt //labs/reasoning_scalability:inference_benchmark
python3 labs/reasoning_scalability/scripts/benchmark.py
python3 labs/reasoning_scalability/scripts/benchmark.py --check
python3 -m unittest discover -s labs/reasoning_scalability/tests -p 'test_*.py'
```

`--prepare` generates and validates inputs without timing. `--report` regenerates
figures from current validated measurements. Plot styling and documentation
changes do not invalidate measurements; executable measurement logic, inputs,
engine sources and the binary remain checked. `--warmups`, `--repetitions` and
`--timeout-seconds` are recorded in the result. The outer process limit scales
with the number of calls and their configured deadline. A failed/incomplete run
does not overwrite the previous validated result.

To remove generated artifacts, delete `output/`; the next run recreates it.
