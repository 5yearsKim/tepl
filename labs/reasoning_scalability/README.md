# Attribute inference cost and scalability

Current experiment: a bounded synthetic evaluation of TEPL attribute inference.
There are no XLA inputs. Dot-chain lengths and tensor ranks are exactly **1–6**.

## Required measurements

1. **Graph-size scaling:** left-to-right reassociation of a contraction chain,
   with 1–6 dot operations per side and 2–7 distinct input tensors. Tensor rank
   stays at three. Canonical and permuted physical layouts have matched
   explicit, partial-auto and full-auto variants. Partial auto infers odd RHS
   nodes; the other fields come from the independent label-based generator.
   Size one is an identity-copy baseline; one descriptor has no proper partial mode.
2. **Rank scaling:** ranks 1–6 with three source transposes and one replacement
   transpose. Source permutations are independent symbolic attributes. Explicit
   composition and full auto use identical source metadata and conditions.
3. **Phase breakdown:** construction and verification time across all strategy
   attempts, including unsuccessful attempts and fallback. Dispatch/accounting
   time is shown separately. Phase bars use means; latency curves show medians
   with min/max whiskers.

All generated artifacts go in the Git-ignored `output/` directory:

- `output/figures/`: chain scaling, rank scaling and phase breakdown (PNG, SVG, PDF).
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
serially. The default work budget remains 1,000,000 per strategy attempt.

To keep this first run inexpensive, the default is **one warmup and five measured
repetitions** per case (276 calls for 46 cases). Repetitions can be increased
without changing inputs. All outcomes, including unresolved or budget-exhausted
cases, remain in the raw data. These measure metadata/proof costs rather than
numerical tensor execution. The arithmetic policy is `algebraic`.

The contraction generator supplies both operand layouts and explicit
RHS fields. A separate dense integer evaluator validates the generated
contractions against label-based matrix multiplication, and transpose
composition against physical array transposition. These checks do not use
TEPL's normalization or inferred constructor choices. They supplement the
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
figures from current validated measurements. `--warmups`, `--repetitions` and
`--budget` are recorded in the result. A measurement run has a 300-second outer
limit; this is separate from the engine's work budget. A failed/incomplete run
does not overwrite the previous validated result.

To remove generated artifacts, delete `output/`; the next run recreates it.

The optimization evaluation also saves a frozen baseline binary and samples in
`output/baseline/`, with the before/after analysis in `output/comparison.md`.
It compares identical generated rules and budgets, reports latency ratios only
for cases proved by both versions, and lists newly proved cases separately.
