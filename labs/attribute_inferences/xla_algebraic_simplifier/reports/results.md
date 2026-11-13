# XLA branch integration results

All observed applications were delegated during paired normal CPU compilations. The native and integrated compilations produced identical canonical final HLO and identical per-source-site application counts. Both outputs were finite and agreed at absolute/relative tolerance `1e-5` for three deterministic input sets per workload.

| Workload | Source variants | Native applications | TEPL applications | Final graph | Output sets |
|---|---:|---:|---:|---|---:|
| dlrm | 32 | 201 | 201 | Equal | 3 |
| gpt2 | 14 | 1693 | 1693 | Equal | 3 |
| vit | 16 | 1849 | 1849 | Equal | 3 |
| resnet | 10 | 1027 | 1027 | Equal | 3 |

The four workloads exercise 37 distinct source sites and 4,770 applications. The adapter has 38 ports for the retained paper workload cohort. All 38 ports also passed the 47 generated branch controls (141 numerical input sets). The versioned JSON bridge passed 546 independent permutation cases across 13 dtypes and six protocol boundary checks. Six preparation/evidence tests and the two runtime test targets passed.

Inputs use `finite_inputs_v2`: floating placeholders receive bounded finite values; integer placeholders receive iotas along their last axis. ViT position IDs are `0..196`. Explicit literals, including the HLO error-path NaNs, are preserved. The original input HLO bytes and their hashes are unchanged. Runtime arguments are deterministic synthetic values; this experiment does not measure pretrained model accuracy.

The retained ViT reference has 17 variants and 1,850 applications. The valid-index replay has 16 and 1,849: only the uniform-NaN literal rewrite AS-2311 disappears. The floating-only diagnostic matched all 1,850 applications and final graphs but failed numerical validation because its original synthetic position indices selected the NaN error path. That diagnostic is excluded from the successful results above.

Two transpose ports use automatically compiled algebraic rules. The other ports use explicit structural TEPL programs; this deployment evidence is separate from semantic proof and standalone attribute-inference measurements. Source branches outside the port inventory remain native. Strict mode rejects an observed native application without a committed TEPL replacement.

Machine: Intel Core Ultra 7 155H. XLA revision: `39c348196c834a7ee45a57f5b44688d53ae99a64`. TEPL base revision: `301ac2d894163331599b5f3677d319a403a23470`, with the recorded checkout patch and source snapshots. This is an equivalence/integration comparison, not a timing result.

Evidence: [workloads](workloads.json), [branch controls](branch-controls.json), [bridge checks](bridge.json), [runtime tests](runtime-tests.log), and [floating-only ViT diagnostic](vit-floating-only-diagnostic.json). The workload JSON retains binary, program, input, source, and final-graph hashes. Full binaries, complete compressed final graphs, source snapshots, and raw logs are retained in [the local run](../../../artifact/output/branch-integration/workloads-2/). Reproduction commands are in the [adapter README](../README.md).
