# TEPL evaluation artifact

The artifact bundles the paper's 123 modeled XLA tasks, including 38 axis
inference tasks, their scope audit, and reference experiment totals. Input
hashes are recorded in `fixtures/paper/manifest.json`. A sibling paper checkout
is optional.

[Recorded results](reports/current/report.md) include the new measurements and
their evidence hashes. Raw runs are retained locally under `output/release/`.

Use Python 3.10 or newer for coverage and Einsum. The scalability experiment
also needs NumPy and Matplotlib. `requirements-lock.txt` records the Python 3.14
environment used for this artifact's new runs.

From the repository root:

```sh
bazel build -c opt //:tepl \
  //labs/attribute_inferences/reasoning_scalability:inference_benchmark
python3 labs/artifact/run.py quick
python3 labs/artifact/run.py reproduce --out labs/artifact/output/run-1
```

To include scalability, install its dependencies in a virtual environment and
pass the benchmark binary. Run on an otherwise idle machine:

```sh
python3.14 -m venv /tmp/tepl-artifact
/tmp/tepl-artifact/bin/pip install -r labs/artifact/requirements-lock.txt
/tmp/tepl-artifact/bin/python labs/artifact/run.py reproduce \
  --out labs/artifact/output/run-2 \
  --scaling-binary bazel-bin/labs/attribute_inferences/reasoning_scalability/inference_benchmark
```

Each run requires a fresh output directory. Coverage retains the compiler
binary, checkout snapshot, migrated inputs, completed programs, per-command
logs, outcomes, and provenance. Migration removes retired separators and
`where` braces while preserving expressions. Acceptance requires semantic
proof under the written conditions, compilation, and portable validation. The
incorrect subtraction-to-addition control must be rejected.

Individual coverage runs and saved-result checks are available:

```sh
python3 labs/artifact/run.py check-inputs
python3 labs/artifact/run.py coverage --out labs/artifact/output/coverage-1
python3 labs/artifact/run.py check-run labs/artifact/output/coverage-1
python3 labs/artifact/run.py coverage --backend core --out labs/artifact/output/core-1
```

The Einsum report independently checks concrete replacement graphs and counts
axis entries. See [its protocol](../einsum_benchmark/README.md). Scalability
separately measures compilation and metadata execution; see
[its protocol](../attribute_inferences/reasoning_scalability/README.md).

The source-size runner recounts the saved explicit/inferred variants and the
reviewed XLA source spans. It reports the 30-rule matched comparison and the
31-rule TEPL total separately. Optional verification records each current
proof/compilation/validation outcome:

```sh
python3 labs/artifact/source_size.py --out labs/artifact/output/size-1 --verify
```

Collect a completed run's summaries and figures into a fresh report directory:

```sh
python3 labs/artifact/report.py --run labs/artifact/output/run-1 \
  --out labs/artifact/output/run-1-report
```

The [AlgebraicSimplifier adapter](../attribute_inferences/xla_algebraic_simplifier/README.md)
has a separate XLA build, 47 branch controls, and a strict four-workload CPU
pipeline runner. Its ports cover the 38 source variants observed in the
retained workloads. That runner measures actual committed TEPL applications
and checks native/integrated graphs and outputs. Four-workload reference
counts elsewhere in the bundled fixtures describe native XLA activity and
static semantic classification. Use the adapter's separately recorded results
for delegation evidence; `reproduce` runs compiler coverage, source size,
Einsum, and optional scalability.
The [recorded strict CPU comparison](../attribute_inferences/xla_algebraic_simplifier/reports/results.md)
delegates all 4,770 observed applications and passes the graph/output checks.

Retired transpose-folding experiments are stored in
`../archive/transpose-folding.tar.gz`. Its adjacent manifest records the
original revision and per-file hashes. They are outside the active Bazel
packages and reproduction commands.
