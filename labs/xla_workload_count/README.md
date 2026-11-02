# Reusable XLA workload counting

The original **counts were automated**. The semantic support review of the 32
DLRM variants was manual, then encoded as data. This package separates those
steps so the same review is reused for other workloads.

| Step | Automation |
|---|---|
| Count successful rule applications and distinct variants | Automatic |
| Report per-rule counts and reviewed transformation families | Automatic |
| Reuse reviewed semantic representation categories | Automatic, pinned to XLA source and semantic-profile hashes |
| Assess a previously unseen transformation branch | Review once, then add a registry entry |
| Resolve data-dependent guard/reducer/dtype cases | Requires more metadata and/or a TEPL rule check; stays conditional |
| Prove a new TEPL rewrite or measure actual delegation | Not performed by this counting tool |

Scope: all AlgebraicSimplifier invocations during one normal XLA CPU
compilation. Other XLA passes require their own instrumentation. Counts are
successful source-branch variants, not an official or canonical semantic-rule
count. No numerical execution or timing repetitions are needed.

## Count another workload

The counter is already built and registered in this workspace. From TEPL:

```bash
python3 labs/xla_workload_count/run.py count --hlo /path/to/model.hlo --out /tmp/model-rewrite-count
```

This performs exactly one model compilation. It never builds XLA implicitly.
The output directory must be new or empty, preserving earlier results.

Outputs:

- `summary.json`: distinct variants, applications, semantic categories and totals.
- `rules.csv`: each rule, frequency and reused static semantic classification.
- `families.csv`: transformation-family counts.
- `review-needed.csv` and `review-context/`: unseen/stale variants, sorted by frequency.
- `conditional.csv`: previously reviewed variants requiring additional context.
- Raw counts, compiler stderr, final HLO, input and version/hash metadata.

To focus on another operation family, inspect `families.csv` and its rows in
`rules.csv`; the counter already observes all 322 instrumented sites rather
than just the two transpose-folding rules.

## Analyze saved counts without running XLA

```bash
python3 labs/xla_workload_count/run.py replay --raw /path/to/raw-results.json --provenance /path/to/counter-provenance.json --out /tmp/reviewed-counts
```

For older artifacts the provenance filename may be `provenance.json`.
Adjacent filenames are detected automatically. Saved raw data can be rejoined
with an updated review registry repeatedly, with no model recompilation.

## Review a new rule only once

1. Open the highest-frequency rows in `review-needed.csv` and inspect the original
   source and full applicability conditions; excerpts are navigation aids.
2. Check the required value/axis contracts and existing TEPL rule counterparts.
3. Add its ID to `support-registry.json`, recording `function`, `family`,
   `description`, `semantic_status`, `evidence`, and `reason`.
4. Replay saved counts. Future workloads reuse that entry automatically.

Statuses are `direct`, `unit_view_candidate`, `conditional`, `outside`, and
`unreviewed`. These are **static representation assessments**. They are not
measured successful TEPL proof checks or supported application counts.
Unknown rules are never guessed supported or silently counted unsupported.
Conditional sites remain conditional: a site can cover several actual scalar
operations or reducers, which the existing counter does not record individually.
The DLRM seed registry has 14 direct, six unit-view candidates, six conditional,
and six outside-profile variants. The combined
`semantic_model_coverage` field reports direct + unit-view candidates: 20/32
for DLRM. Keep that definition when using the combined number. The six unit-view
entries still need normalized ports and validation; the field does not report
successful TEPL execution or proof checks.

XLA revision and original/instrumented source hashes must match before counts
are interpreted. Changed semantic-profile files invalidate classifications and
queue review rather than reusing stale support claims. A new XLA revision needs
its rule IDs and normalization reviewed; source line numbers cannot be reused
blindly across revisions.

## One-time setup on another checkout

Source and instrumentation blueprints are under `counter/`. For the pinned XLA
revision, prepare without a build:

```bash
python3 labs/xla_workload_count/run.py prepare --xla-root /path/to/xla
```

Explicitly build and register the counter when needed:

```bash
python3 labs/xla_workload_count/run.py prepare --xla-root /path/to/xla --build
```

A fresh XLA dependency build can be expensive; this is separate from workload
counting and only happens with `--build`. Preparation checks source versions
and preserves conflicting local changes by stopping. The build uses the
recorded optimized Linux CPU configuration. Registration checks a repeated-rule
control (two applications, one variant) and a no-match control, then records the
binary hash. No model is compiled during setup.

## Local validation

```bash
python3 labs/xla_workload_count/test_reuse.py
```

Tests replay the saved DLRM fixture, introduce an unseen variant, invalidate a
source/profile version, and check bookkeeping and total-count failures. They
never execute XLA. Python 3.11 or newer is required.
