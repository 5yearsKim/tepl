# Retained workload inputs

These HLO files are byte-for-byte copies of the inputs retained with the
submitted paper, revision `130052b1e92c23dfe390e244b4fcdffbde8bd0a1`.
[manifest.json](manifest.json) records the input SHA-256 hashes, original
evidence paths relative to the paper's `evaluation` directory, native reference
counts, and pinned XLA revision. The experiment checks these hashes before
running. It does not require access to the paper checkout.

GPT-2 export details are preserved in
[gpt2-export-provenance.json](gpt2-export-provenance.json): the GPT-2 Small
configuration has 124,439,808 parameters and 148 parameter tensors, with batch
size 1 and sequence length 16. The model inputs include runtime parameter
tensors. The runner supplies deterministic values rather than downloading
pretrained weights. Floating arguments are small positive values; integer
arguments are ones and predicates are false. Three seeds vary the floating
arguments. DLRM contains one elided literal and ViT contains 201. The default
runner initializes elided floating literals to bounded finite values using
the recorded `finite_inputs_v2` policy. Elided integer literals become iotas
along their final axis: DLRM gets indices `0..14` and ViT gets position IDs
`0..196`. The latter are within the position embedding table's bounds. Explicit
literals and predicate placeholders are preserved.
`--shortform-constants original` preserves the parser's original synthetic bit
patterns instead. Neither policy supplies pretrained weights. This tests
compiler integration and numerical agreement, not model accuracy.

The reference `modeled` counts are retained static classifications. They are
not counts of semantic proofs performed by the integration runner. The runner
records actual native and committed TEPL applications separately.
