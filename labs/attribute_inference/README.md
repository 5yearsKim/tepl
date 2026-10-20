# Automatic attribute inference evaluation

This experiment measures whether TEPL can construct replacement attributes,
how much explicit attribute code remains, and the cost of checking the rule.
The [XLA evaluation](../xla_rewrite/README.md) remains the source corpus.

## Plan and checkpoints

1. **Small pilot (completed).** Test the five cases in
   [data/pilot_cases.json](data/pilot_cases.json), with explicit and full-auto
   versions, two assisted variants, and one incorrect-constructor control.
   Review [reports/pilot.md](reports/pilot.md) with the user before expanding.
2. **Corpus evaluation (completed).** Classify all XLA cases by
   operation-semantic support, presence of RHS attributes, and target attribute
   kind. Attempt eligible cases, including cases whose explicit versions are
   unproved. Report overall and explicit-baseline-conditioned coverage.
3. **Generality and validation (after review).** Add general symbolic transpose,
   reduction, broadcast, guarded dot, and slice rules. Independently evaluate
   generated attributes and tensor results on legal fixed-seed metadata. Include
   empty groups, interleaved selections, and supported zero extents. Track
   semantic-definition cost once per dialect.
4. **Performance (after review).** Generate depth/overlap/expression-size
   workloads. Measure repeated checking time, memory, constructor DAG size and
   work counters. Measure concrete-rank constructor execution separately.

After reviewing the staged pilots, the user authorized completing all remaining
cases. The [complete cohort report](reports/corpus.md) now measures all 38
axis/extent cases with operation semantics, plus the four scalar-setting cases
as a supplementary cohort. Generality, independent numerical validation and
scaling remain proposed work; this completes the XLA attribute-cohort experiment.

## Current results after shared-frame slice inference

The [general reasoning recheck](reports/reasoning_fixes.md) and
[reverse-inference recheck](reports/reverse_inference.md), followed by the
[shared-frame slice recheck](reports/slice_frame_inference.md), repeat all 42
explicit and full-auto cases, plus the incorrect-constructor control, using
the frozen inputs. Each earlier result is preserved.

| Measurement | Original | General reasoning | Reverse inference | Slice frame |
|---|---:|---:|---:|---:|
| Primary explicit proofs | 20/38 | 20/38 | 20/38 | 20/38 |
| Primary full-auto proofs | 17/38 | 19/38 | 21/38 | 22/38 |
| Full auto among the 20 explicitly verified cases | 15/20 (75%) | 17/20 (85%) | 19/20 (95%) | 20/20 (100%) |
| Full-auto proofs excluding the detected identity witness | 16/38 | 18/38 | 20/38 | 21/38 |

The two previously assisted cases, `scalar_broadcast_transpose` and
`slice_compose`, the two reverse-composition cases, and
`slice_compose_zero_starts` now prove with full auto.
No previously proved explicit/full-auto variant was lost. The four supplementary
scalar-setting cases still have no full-auto proof. Assistance was not rerun in
these comparisons; the incorrect permutation control was rejected each time.

The implementation uses general transfer rules for uniform-vector arithmetic,
legal selections and concatenation, length-compatible unit scaling and zero
offsets, and zero-cardinality coordinate sequences. It does not inspect operation
or case names. Focused regressions include symbolic-length slice composition,
scalar broadcast with symbolic permutations, concrete tensor comparisons,
empty groups, and rejection of invalid metadata. A dimension with extent zero
remains an axis; only a sequence with zero axes becomes an empty group.
Reverse inference matches coordinate identities, then binds an unknown direct
selector to the source's normalized reversal parity. When the reversals cancel,
it binds the empty selector. Fresh verification checks the completed constructor.
Tests cover arbitrary selector overlap, rank zero through four, cancellation,
concrete tensor results, and rejection of a different coordinate frame.
For single non-reduced output domains, inference now preserves output axes
that overlap an input. This lets the existing affine slice normalization infer
zero-start slice composition without a rule-specific constructor.

```sh
bazel build //:tepl
python3 labs/attribute_inference/scripts/run_reasoning_fixes.py --stage slice_frame_inference
python3 labs/attribute_inference/scripts/run_reasoning_fixes.py --stage slice_frame_inference --check
bazel test //tests:all
python3 -m unittest discover -s labs/attribute_inference/tests -p 'test_*.py'
```

Raw results and source/compiler hashes are saved separately in
[results/reasoning_fixes.json](results/reasoning_fixes.json),
[results/reverse_inference.json](results/reverse_inference.json), and
[results/slice_frame_inference.json](results/slice_frame_inference.json). Earlier
stages' `--check` commands require their matching source and compiler hashes;
use the shared-frame slice recheck for the current compiler.

## Baseline results before reasoning fixes

[Corpus classification](reports/corpus_classification.md) identifies 111 cases
with RHS attributes: 42 have semantics for all required operations, and 38 of
those request only axis/extent fields. The other four request scalar comparison
settings. Missing semantics and previously unproved baselines are recorded, not
silently removed. Classification is a static inventory; the complete run checks
every member of the available-semantic cohort rather than sampling it.

| Measurement | Complete primary cohort |
|---|---:|
| Axis/extent cases attempted | 38 |
| Explicit versions proved | 20/38 |
| Full-auto versions proved | 17/38 |
| Full auto among explicitly verified cases | 15/20 (75%) |
| Additional cases proved with field assistance | 2 |
| Full or assisted inference | 19/38 |
| Full or assisted among explicitly verified cases | 17/20 (85%) |
| Full-auto successes with unproved explicit baselines | 2 |

The two assisted recoveries pin `dimensions=[]` for scalar broadcast through
transpose, and `strides=[1]` for the rank-one unit-stride slice-composition case.
Operation-level assistance recovers no additional cases under the tested policy.
The four supplementary comparison-setting cases have no full-auto success.

The complete report distinguishes reuse, constant construction and computed
attributes. In the 15 pairs where both explicit and inferred versions verify,
attribute/derive tokens fall from 341 to 60 (82.4%). Guards and shared semantic
definitions remain unchanged. The two extra automatic proofs use alternative
metadata programs where the explicit comprehensions remain unproved.

One of those extra proofs, `catalog_indexing_slice_singleton_stride`, simply
copies the original slice attributes. It is a valid identity witness and does
not implement the intended stride simplification. The report labels it and
gives coverage with that detected identity excluded (16 full-auto cases).
Semantic-witness coverage does not establish optimization profitability.

The [first pilot](reports/pilot.md) and [batch 01](reports/batch_01.md) retain
their original selected-sample results. The complete run measures each case
afresh with a single compiler and fixed assistance policy, avoiding duplicate
counting across those earlier stages.

## Scope and comparisons

Target attributes include axis lists, permutations, contraction/batch dimensions,
extents, and supported calculated-read coefficients/offsets. Current semantics
include some calculated indices; they do not support tensor-valued indices or
coordinate-dependent branching required for dynamic slicing and general padding.
Other semantic limitations must be classified separately in the corpus sweep.

Each pair uses the same dialect, operation structure, tensor restrictions,
guards and arithmetic policy. Changing a condition creates a new benchmark.
Different attribute programs may be valid witnesses of the same equation;
identity witnesses and the intended optimization must be distinguished.

- **Explicit:** the original rule and its attribute construction.
- **Full auto:** all RHS descriptors inferred; explicit derivations removed.
- **Assisted operation:** one RHS descriptor stays explicit, another is inferred.
- **Assisted field:** some fields are pinned; the other fields are inferred.

Report assisted successes separately, including how many additional cases they
recover beyond full auto. Supplying every field does not count as inference.
Separate reuse, fixed-constant construction, and metadata-dependent computation.
Cases without RHS attributes have no inference work, not an inference success.

## Pilot selection

The five cases are deliberately chosen examples, not a random sample or a corpus
coverage estimate. Prior exploratory measurements informed their selection.
They exercise permutation composition, slice arithmetic, two RHS descriptors,
reversal matching, and broadcast legality. Two assisted variants demonstrate
the different forms of author guidance. Some rules have fixed-rank restrictions;
the pilot does not establish arbitrary-rank support for every family.

Acceptance requires exit code zero, exactly one `status=proved`,
`additional-guard: true`, no unresolved constructor holes, and a reported
constructor for every requested RHS descriptor. A failed proof is unproved,
not evidence that every possible replacement attribute assignment is invalid.
The incorrect-constructor control must be rejected by semantic checking.

## Reproduce only the pilot

From the repository root:

```sh
bazel build //:tepl
python3 labs/attribute_inference/scripts/pilot.py
python3 labs/attribute_inference/scripts/pilot.py --check
python3 -m unittest discover -s labs/attribute_inference/tests -p 'test_*.py'
```

The runner executes only manifest-listed variants and one control, serially.
It supports `--compiler PATH` and `--timeout SECONDS`. Generated rule inputs
are retained under `tepl/`; original XLA files and semantics are reused without
modification. Raw evidence, source/compiler hashes, environment, and diagnostics
are saved in [results/pilot.json](results/pilot.json).

Attribute-code tokens count RHS descriptor annotations and `derive` contents,
excluding comments, whitespace and descriptor spelling differences. They are
an author-code proxy, not a usability measurement; whole-rule tokens are also
recorded. Guards are unchanged and their token counts are reported separately.
Shared semantics are reused, not eliminated. Code-size savings are interpreted
only for variants that verify. Single-run wall times include CLI startup,
parsing, checking and report printing; they are not performance results.

This pilot does not run a full corpus sweep, numerical validation, scalability
workloads, or compiler ablations. Algebraic checking uses the declared local
model and does not establish complete XLA/IEEE equivalence.

## Reproduce classification and batch 01

```sh
python3 labs/attribute_inference/scripts/classify_corpus.py
python3 labs/attribute_inference/scripts/classify_corpus.py --check
python3 labs/attribute_inference/scripts/run_batch.py
python3 labs/attribute_inference/scripts/run_batch.py --check
```

Classification reads all 209 non-control cases and writes a static inventory;
it does not execute TEPL. The five-case batch is selected before checking from
broadcast, reduction, dot, indexing and convolution strata. The reduction and
indexing selections deliberately challenge previously unproved explicit cases.
Selection order, original case scope and source hashes are recorded in
[data/batch_01.json](data/batch_01.json). The two mixed-operation cases additionally
request broadcast-only inference, keeping the other descriptors explicit.

The batch runner allows an explicit baseline to remain unproved and reports it
alongside its inferred counterpart. It executes only those five new cases and
their listed variants, with no changes to original rules, guards or semantics.
Its [raw results](results/batch_01.json) and [report](reports/batch_01.md) are
separate from the first pilot's evidence. Check commands validate hashes and
reports without running TEPL again.

## Reproduce the complete cohort

```sh
bazel build //:tepl
python3 labs/attribute_inference/scripts/classify_corpus.py --check
python3 labs/attribute_inference/scripts/run_corpus.py --prepare
python3 labs/attribute_inference/scripts/run_corpus.py
python3 labs/attribute_inference/scripts/run_corpus.py --check
python3 -m unittest discover -s labs/attribute_inference/tests -p 'test_*.py'
```

[data/corpus_cases.json](data/corpus_cases.json) freezes all 42 available-semantic
attribute cases and every assisted variant before execution. The primary
cohort is 38 axis/extent cases; the supplementary cohort is four comparison
settings. These groups are identified by field type rather than proof outcome.

After a full-auto failure, assistance tries every nonempty proper subset of
RHS descriptors, retaining required derivation dependencies. Single-descriptor
cases also try available literal field pins taken from original constructors,
spreads or direct guard equalities. Pins must leave at least one field inferred.
The policy combines neither assistance modes nor invented values, and makes no
claim of completeness over arbitrary user hints. All failures remain recorded.

The run prepares 161 variants/control, executes 122 checks, and skips 39 assisted
variants whose full-auto versions already verify. Of 37 attempted assisted
variants, two verify. Variant counts are separate from rule-case coverage.
The negative control is executed again and rejected. Original pilot/batch
evidence is preserved, and the consolidated evidence is in
[results/corpus.json](results/corpus.json), with a readable
[report](reports/corpus.md).
