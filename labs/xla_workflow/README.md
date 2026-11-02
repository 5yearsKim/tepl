# TEPL rules in the XLA CPU compilation pipeline

This experiment compares the ordinary CPU compiler with an opt-in adapter at
its existing `TransposeFolding` pass. The adapter exposes dot/transpose HLOs
and opaque tensor boundaries to TEPL's native pattern matcher, evaluates the
compiled rules and their guards, builds their completed replacement metadata,
and commits the result through HLO APIs. XLA's CPU backend callback remains
in force. The residual upstream pass handles unsupported cases. Counts refer
to folded operands in this selected pass, not all XLA optimizations.

The two original certified transpose/dot rules are unchanged. The adapter
imports only as much graph structure as their compiled patterns need and
preserves shared HLO identities. Its operation registry and HLO construction
are shared; it contains no handwritten transpose/dot pattern recognizer or
axis-remapping formula. This prototype runs one compilation at a time. CPU lowering keeps the parsed
XLA defaults, including Eigen eligibility; no optimizer passes are disabled.

## Workloads and scope

* Public Flax v0.12.0 TransformerLM: one layer, two heads, 32 embedding/QKV
  dimensions, 64 feed-forward dimensions, 128 vocabulary entries, length 16,
  FP32, deterministic evaluation, batches 1 and 4.
* Unmodified FP32 DLRM training-step and ResNet-50 HLO snapshots in the pinned
  XLA CPU benchmark directory.

The workload manifest is fixed before measuring pipeline rewrite counts.
Zero-opportunity workloads are included. Parameters and inputs are generated
by a fixed deterministic rule; no training or accuracy claim is made.

## Reproduction

Use a Python 3.12 environment with `jax[cpu]==0.7.2` and `flax==0.12.0`.
The vendored model retains its upstream license header and is fetched from
`https://raw.githubusercontent.com/google/flax/v0.12.0/examples/lm1b/models.py`.

1. Run `workloads.py OUTPUT_INPUTS` to export the two Flax configurations and
   copy the two original XLA HLOs. Source and input hashes are retained.
2. Build `//:tepl` and `//labs/xla_workflow:tepl_workflow.so` in TEPL with `-c opt`.
3. Compile `inferred.tepl` with `--semantic-policy algebraic --core`, then run
   portable validation. The resulting program is not specialized to a model.
4. Run `prepare_xla.py XLA_ROOT` to install the opt-in hook and copy this driver.
   Without a registered hook/observer the pass retains upstream behavior.
5. Build `//research/tepl_workflow:workflow` in XLA using `-c opt` and the pinned
   toolchain configuration recorded with the experiment.
6. Run the driver's `--self-test LIBRARY PROGRAM` and the Python bridge tests.
7. Run `run.py ARTIFACT_DIR` to freeze inputs, source, binaries and provenance,
   compile every workload with both paths, execute both, and save all results.

## Measurements

The pipeline hook records actual pass invocations and applications. Each model
uses a CPU PjRt client and the complete XLA CPU pipeline and backend. There is
one warmup per variant followed by ten paired rounds with rotating order.
The integrated compilation timer includes a fresh portable-program load.
Parsing, input cloning, graph comparison and numerical execution are excluded.
Diagnostic HLO verification and serialization inside the hook run only during
the warmup. There is no cross-match or output cache in this adapter.

The driver preserves pass-boundary and final optimized HLO, using canonical
instruction names and identifiers while retaining full constants, backend
configuration and control edges; debug metadata is ignored.
Baseline and integrated executables are compared for three deterministic input
sets using absolute and relative tolerances 1e-5; FP32 outputs must be finite.
The comparison is testing, not a proof of arbitrary whole-program equivalence.
Compilation overhead is reported as median paired percent change and min/max
across the ten rounds. No model execution speedup is claimed.

The earlier 25-case isolated-pass experiment remains separately frozen in the
paper's `evaluation/xla-integration/`; it is not overwritten by this workflow.
