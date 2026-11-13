# TEPL inside AlgebraicSimplifier

This adapter delegates replacement construction at XLA's existing
`AlgebraicSimplifier` branches to compiled TEPL programs. XLA retains traversal,
candidate selection, backend checks, and instruction ownership. TEPL selects
the replacement operations and computes their attributes; a typed C++ factory
materializes the returned graph. The adapter has ports for all 38 source-branch
variants observed in the four retained paper workloads. The
[port inventory](ports.json) identifies each original source site and TEPL rule.
The [recorded CPU comparison](reports/results.md) delegates all 4,770 observed
applications with equal final graphs and finite, matching outputs.

Two transpose rules use automatic compilation with the algebraic semantic
policy. The remaining ports use explicit structural programs in
[branch_rules.tepl](branch_rules.tepl). Their contracts validate graph metadata;
they are not additional semantic proof results. XLA's original branch guards
establish the conditions under which these programs are called. General shape
correspondence witnesses remain supplied by XLA; the TEPL programs compute the
replacement's dimensions from those witnesses. Literal extraction, literal
reciprocal evaluation, and typed HLO construction are host primitives.

The bridge loads portable executable programs. It performs no source parsing,
synthesis, or proof search during matching. Original commit APIs retain their
layout, sharding, control-dependency, and in-place-update behavior. A strict
callback fails on an unsupported proposal rather than silently using the
native replacement. Branches outside the port inventory remain native. The
standalone drivers compile serially with process-global hooks; these hooks are
not a concurrent compiler interface.

## Build

Use XLA revision `39c348196c834a7ee45a57f5b44688d53ae99a64`. Install the
[counting instrumentation](../xla_workload_count/README.md) first. Preparation
checks the pinned instrumented source and refuses conflicting local edits.
Repeated preparation checks the hashes of previously generated files.

From TEPL, with `XLA_ROOT` pointing to that checkout:

```sh
bazel build -c opt //:tepl \
  //labs/attribute_inferences/xla_algebraic_simplifier:tepl_algebraic.so
python3 labs/attribute_inferences/xla_algebraic_simplifier/prepare_xla.py "$XLA_ROOT"
```

In XLA:

```sh
bazel build --config=joint_shard -c opt --jobs=4 \
  //research/tepl_algebraic:integration //research/tepl_algebraic:pipeline
```

The local `joint_shard` configuration selects the recorded Linux sysroot. On a
clean checkout, add `build:joint_shard --repo_env=SYSROOT_DIST=linux_glibc_2_35`
to the local Bazel configuration before building.

## Branch controls

Back in TEPL:

```sh
mkdir -p labs/artifact/output/branch-programs
bazel-bin/tepl compile \
  labs/attribute_inferences/xla_algebraic_simplifier/rules.tepl \
  --core --semantic-policy algebraic \
  -o labs/artifact/output/branch-programs/automatic.json
bazel-bin/tepl compile \
  labs/attribute_inferences/xla_algebraic_simplifier/branch_rules.tepl \
  -o labs/artifact/output/branch-programs/branches.json
bazel-bin/tepl validate labs/artifact/output/branch-programs/automatic.json
bazel-bin/tepl validate labs/artifact/output/branch-programs/branches.json
python3 labs/attribute_inferences/xla_algebraic_simplifier/test_branch_bridge.py \
  bazel-bin/labs/attribute_inferences/xla_algebraic_simplifier/tepl_algebraic.so \
  labs/artifact/output/branch-programs/automatic.json
python3 labs/attribute_inferences/xla_algebraic_simplifier/test_branches.py \
  --driver "$XLA_ROOT/bazel-bin/research/tepl_algebraic/integration" \
  --library bazel-bin/labs/attribute_inferences/xla_algebraic_simplifier/tepl_algebraic.so \
  --automatic labs/artifact/output/branch-programs/automatic.json \
  --branches labs/artifact/output/branch-programs/branches.json \
  --out labs/artifact/output/branch-controls
```

The 47 controls cover every port, including a native guard rejection and a
no-match case. Positive cases require actual committed TEPL replacements.
Each case compares canonical native/integrated graphs and evaluates three
input sets against the original graph. These are generated branch controls,
separate from the paper's 28 related upstream tests. Their pass-level options
disable dot strength reduction and noncanonical dots to exercise the intended
branches. The workload driver uses the normal CPU pipeline instead.

## Four-workload comparison

```sh
python3 labs/attribute_inferences/xla_algebraic_simplifier/run_workloads.py \
  --xla-root "$XLA_ROOT" \
  --out labs/artifact/output/branch-workloads
```

This runner bundles the exact retained DLRM, GPT-2, ViT, and ResNet HLOs,
compiles the two programs, and snapshots the binaries, adapter sources, and
provenance. See [workload provenance](workloads/README.md). Each workload is
compiled twice through the CPU backend, first with native replacements and
then with TEPL hooks. All `AlgebraicSimplifier` invocations are counted. The
comparison includes full constant values, backend configuration, layouts, and
control edges in canonical final HLO. Three deterministic input sets must
produce finite, matching outputs at absolute/relative tolerance `1e-5`.
This is an integration/equivalence experiment, not a timing benchmark.

The retained dumps contain elided literal payloads: one in DLRM and 201 in
ViT. By default, the runner initializes their floating values deterministically
in `[0.0001, 0.0097]` and initializes integer placeholders as iotas along their
last axis before cloning the shared input module. The integer sequences cover
DLRM's interaction indices and ViT's 197 position indices. Explicit literals
and predicate placeholders retain XLA's parsing behavior. The original
HLO bytes are preserved. `--shortform-constants original` selects XLA's default
synthetic bit patterns; numerical validation still rejects non-finite outputs.
Each run records the `finite_inputs_v2` policy and separate floating/integer
literal counts. This prevents synthetic out-of-range position indices from
selecting the ViT HLO's NaN error branch.

Strict mode requires every observed non-bookkeeping application to be
delegated, checks successful commits against native source-site counters, and
rejects native fallback. `--allow-native` is an explicitly labeled hybrid
diagnostic mode. It cannot certify complete delegation. Failed runs preserve
raw logs and intermediate compilation evidence. Use a fresh output directory for each run; final graphs with full
constants can be large. Successful comparisons archive them losslessly as
`.final.hlo.gz`, retaining the uncompressed graph hashes.

The paper's retained native counts are reference values in the input manifest.
Actual counters from this runner are the integration result. The 38-port
workload cohort does not represent all source branches in
`AlgebraicSimplifier`. Static semantic classification and standalone attribute
inference results remain separate experiments.
