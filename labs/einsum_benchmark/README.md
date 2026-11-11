# 16-input Einsum rewrites and inferred axes

This lab evaluates one connected **16-input subnetwork** per original
[Einsum Benchmark instance](https://zenodo.org/records/11477304), using seed 42.
Each supplied contraction path has 15 binary contractions, plus any reductions
and a final permutation. The experiment measures whether both paths can be
represented in TEPL and whether TEPL can infer the omitted axis attributes when
rewriting between their operation trees. It does not execute tensor values or
measure numerical execution speedups.

The [recorded report](reports/local16/report.md) contains 168 sampled cases,
117 represented cases, and 226 independently verified transformations out of
228 nonidentical tasks. See the [category table](reports/local16/categories.csv)
and [per-rule axis counts](reports/local16/axis_counts.csv) for details.

## Run

From the repository root, build TEPL and run the included portable dataset
(Python 3.10+; the runner needs only the standard library):

```sh
bazel build //:tepl
python3 labs/einsum_benchmark/benchmark.py run --compress-artifacts
python3 labs/einsum_benchmark/axis_report.py
```

The defaults use `fixtures/local16.json`, write generated artifacts to
`results/local16`, and produce the report in `reports/local16`. The runner
requires exactly 16 inputs per case and an empty output directory. If a run
already exists, choose a fresh directory and pass it to both commands:

```sh
python3 labs/einsum_benchmark/benchmark.py run \
  --out labs/einsum_benchmark/results/local16-rerun --compress-artifacts
python3 labs/einsum_benchmark/axis_report.py \
  --run labs/einsum_benchmark/results/local16-rerun \
  --out labs/einsum_benchmark/reports/local16-rerun
```

The protocol uses core inference, generalized domain guards, algebraic
arithmetic, four parallel cases, and a 10-second timeout per subprocess.
`--jobs`, `--timeout`, and `--binary` can be overridden. A subprocess timeout
does not bound Python graph generation. Process timings include CLI startup,
compilation, loading, and matching; they do not isolate constructor latency.

Each run retains `results.json`, `results.csv`, `provenance.json`, a coverage
report, and per-case inputs, TEPL graphs, compiled programs, and subprocess logs.
`--compress-artifacts` compresses completed per-case graphs and logs. Raw
source downloads and generated runs are ignored by Git. Recorded reports retain
the hashes and provenance of their original run; a rerun records current code.

## Recreate the input selection

The included `fixtures/local16.json` contains equations, shapes, dtypes, and
both restricted paths, with no tensor values. Its adjacent manifest records
the original input digest and sampling policy. To recreate it from the pinned
official archive:

```sh
python3 -m pip install -r labs/einsum_benchmark/requirements.txt
python3 labs/einsum_benchmark/fetch_archive.py \
  --out labs/einsum_benchmark/data/instances.zip
python3 labs/einsum_benchmark/benchmark.py import-zip \
  labs/einsum_benchmark/data/instances.zip \
  --out labs/einsum_benchmark/data/all.json
python3 labs/einsum_benchmark/local_cases.py \
  --input labs/einsum_benchmark/data/all.json \
  --out labs/einsum_benchmark/data/local16.json --seed 42
```

Compare the generated JSON with `fixtures/local16.json`, or use it with
`--input` on both the runner and report command. NumPy is needed only to read
the official pickles. `import-zip` unpickles trusted official files; use it only
with the pinned archive. The downloader verifies the archive MD5 before
renaming the download. Dataset metadata is attributed to
[Zenodo DOI 10.5281/zenodo.11477304](https://doi.org/10.5281/zenodo.11477304)
under CC-BY-4.0.

Selection considers up to 64 seeded starting tensors and takes a connected
neighborhood. Original shapes and dtypes stay unchanged; indices needed outside
the selected region become local outputs. Both supplied paths are restricted
to the selected leaves rather than reoptimized. The 168 subnetworks are derived
from the originals and are not independent new benchmark instances or
necessarily contiguous subtrees of an original path.

## Verification and metrics

1. Validate equations, dimensions, and both supplied linear binary paths.
2. Lower each path to `dot_general`, `reduce_sum`, and a final `transpose`.
   Preserve shared indices as batch axes until their last use.
3. Compile both fully attributed graphs and verify them against the equation
   with an independent input-port partition oracle.
4. Generate both directed rewrites. The LHS captures descriptors at match time;
   the RHS keeps the target operation structure and omits every axis attribute.
   Rank, dtype, and shared extents remain declared; guards require extents >=2.
5. Compile each rule, commit one rewrite, export and recompile the concrete
   result, and independently verify its structure and tensor semantics.
6. Count inferred fields and positions from the verified result. The report
   reloads each concrete graph and checks its recorded counts independently.

An **axis-list field** is one completed RHS attribute, including an empty list.
An **axis position** is one integer entry in a completed field. Positions are
counted per occurrence across operations and operands, not as unique logical
axes. The report separates contraction, batching, reduction, and permutation
positions and records input ranks, ports, logical indices, and replacement
depth. Compare per-rule distributions alongside totals across categories.

Identical operation/capture structures are excluded from directed task counts.
Singleton-extent exclusions remain in the nonidentical task denominator.
Unsupported dtypes and trace/diagonal cases remain in the case denominator;
they are adapter exclusions. A compiled rule counts as successful inference
only after an actual rewrite and independent verification. A verified single
rewrite at the intentional step limit is accepted. The experiment does not
claim full-network coverage or arbitrary-rank generalization.

## Tests

```sh
python3 -m unittest discover -s labs/einsum_benchmark/tests -v
```

Small synthetic fixtures under `tests/fixtures` exercise lowering, the oracle,
and rule applicability. Integration tests use `bazel-bin/tepl` when available;
the batch test runs 16-input cases through the runner and axis report.
