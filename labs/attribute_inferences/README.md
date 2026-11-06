# Attribute inference engineering

Engineering experiments and validation for TEPL attribute inference, its
scalability, and its use in XLA. Each experiment retains its own inputs, scripts,
tests, and recorded evidence.

| Experiment | Purpose |
| --- | --- |
| [reasoning_scalability](reasoning_scalability/README.md) | Compilation scaling and runtime metadata feasibility/construction costs |
| [xla_integration](xla_integration/) | XLA transpose-folding bridge and pinned upstream pass fixtures |
| [xla_rewrite](xla_rewrite/README.md) | XLA rewrite catalog, axis-inference coverage, and compilation validation |
| [xla_workflow](xla_workflow/README.md) | Integration of compiled TEPL rules into the XLA CPU pipeline |
| [xla_workload_count](xla_workload_count/README.md) | Workload rewrite counts and reviewed semantic applicability; does not measure inference or delegation |

Reproduction commands use paths relative to the repository root. Frozen results
retain their original paths, source/compiler hashes, and outcomes from before
this relocation. XLA evaluation tools resolve those recorded paths to the new
location while retaining integrity checks; changed source hashes still require
the recorded revision or a separately recorded fresh run.

Engineering validation checks implementation inputs. Research-document hashes
retained in older records do not invalidate implementation measurements.

Mathematical arguments and proofs live in
[research/attribute-synthesis](../../research/attribute-synthesis/README.md).
