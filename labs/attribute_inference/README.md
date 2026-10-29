# Historical attribute-inference evaluation

This lab preserves the original full-auto/partial-auto XLA cohorts and their
recorded rechecks. Those experiments included coordinate and extent inference
that production TEPL no longer performs. For the current field-level axis-only
protocol, start with [XLA coverage](../xla_rewrite/coverage/README.md).
For current explicit/inferred size measurements, use the
[specification-size experiment](../xla_rewrite/specification_size/README.md).

## Recorded evidence

| Record | Scope |
| --- | --- |
| [Pilot](reports/pilot.md) | Initial cases, assisted variants, negative control |
| [Batch 01](reports/batch_01.md) | First expanded batch |
| [Corpus classification](reports/corpus_classification.md) | Frozen available-semantic cohort |
| [Complete cohort](reports/corpus.md) | Primary 38 axis/extent cases and four supplementary scalar-setting cases |
| [Reasoning fixes](reports/reasoning_fixes.md) | General reasoning recheck |
| [Reverse inference](reports/reverse_inference.md) | Same-frame reverse recheck |
| [Slice frame inference](reports/slice_frame_inference.md) | Shared-frame slice recheck |

Reports retain their original outcomes and limitations. Raw measurements and
source/compiler hashes are in `results/`; case definitions are in `data/` and
`tepl/`. The [XLA catalog](../xla_rewrite/catalog/README.md) defines the source
corpus. The [scalability lab](../reasoning_scalability/README.md) measures a
separate synthetic workload.

## Reproduction

Use the revision and compiler recorded by each result. These commands describe
the original workflow; successful checks against a newer checkout are not
promised by retaining the scripts:

```sh
python3 labs/attribute_inference/scripts/classify_corpus.py --check
python3 labs/attribute_inference/scripts/run_corpus.py --check
python3 labs/attribute_inference/scripts/run_reasoning_fixes.py \
  --stage slice_frame_inference --check
python3 -m unittest discover -s labs/attribute_inference/tests -p 'test_*.py'
```

`--check` validates recorded artifacts and hashes without new TEPL measurements.
On a compatible historical checkout, `run_corpus.py --prepare` and then
`run_corpus.py` rebuild the full cohort; `run_reasoning_fixes.py` runs a selected
recheck stage. Preserve the existing result files when investigating current
behavior. Never replace historical hashes merely to make an integrity check pass.
