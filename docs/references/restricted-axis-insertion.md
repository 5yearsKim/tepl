# Restricted axis insertion in the constructive core

The native `--core` backend recognizes insertion from the declared `place`
axis contract and operand-read value semantics. Operation names and rewrite
names do not select this path. The existing canonical two-field insertion
contract has placement first and fresh extents second; arbitrary extra
assertions or different contracts are not implicitly admitted.

Compile with:

```sh
bazel-bin/tepl compile rule.tepl --core --semantic-policy algebraic -o rule.json
bazel-bin/tepl validate rule.json
```

## Boundary and construction

Both sides must use the same independent captures, once per component, with
independent automatic RHS core fields. This extension admits one component.
Known LHS insertion nodes are normalized bottom up:

- Compose nested placements and order fresh extents by final position.
- Move transpose and reversal through insertion.
- Move sums over original axes below insertion.
- Eliminate created left contraction axes by summing their paired right axes,
  checking every Appendix A.6 condition: all created axes contracted, original
  batch axes, paired/disjoint legal selectors, and preserved free output order.

The normalized LHS must be core with at most one outer insertion. The original
RHS must be core with at most one outer insertion. Internal RHS insertion is
outside this path. RHS normalization and removal of original holes are not used.

For outer RHS insertion, the compiled check compares the ordered, supplied fresh
extents. The existing core emitter solves the inner RHS with a fresh output
permutation. Its placement builder sets `d[pi[j]] = a[j]`, absorbs the auxiliary
permutation, and returns exactly the original RHS descriptors. A bare RHS
requires no remaining fresh extents and retains the known LHS permutation.
An unwrapped LHS has identity placement and empty fresh extents.

The implementation is in `src/inference/insertion.cc`. Compilation emits two
reusable blocks: known-source metadata normalization and the existing
constructive core DAG. Matching evaluates those blocks; it does not enumerate
axes, invoke the general synthesizer, or select a specialized rewrite schema.
Reversal uses signed source replay and a union-find orientation check over the
core layouts, with deterministic reversal builders and no orientation search.

Original supplied extents, inner unknowns, helper-expression definedness, and
user guards survive compilation. The supported supplied metadata language is
finite nonnegative integers, lists, source fields/shapes, length, range,
addition, gather/index, ordered exclusion, concatenation, and the existing
recognized inverse/scatter comprehensions. Unsupported expressions are not
silently replaced with normalized-source extents.

The exact-real theorem domain requires all original input extents at least two
and supplied fresh extents positive; fresh singleton axes and scalar inputs are
allowed. `execute` reports `OutsideDomain` for failed normalization/domain
conditions and `Impossible` for incompatible ordered fresh extents or an
infeasible inner core problem. Native rewrite execution skips these matches;
it preserves the stored user guards. Work and axis limits remain enforced.

## Regression evidence

```sh
bazel test //tests:insertion_test --test_output=all
python3 labs/attribute_inferences/xla_rewrite/coverage/scripts/restricted_insertion.py \
  --paper /path/to/TEPL-paper --output /tmp/restricted-insertion.json
```

The test validates all eight `main_theorem_reduction` inputs, including the
recorded normalized leading-unit insertion and the broadcast/transpose
companion. Fixtures retain the original guards and supplied expressions;
`tests/fixtures/restricted_insertion/provenance.json` records source hashes.
Only old syntax and dialect qualification are migrated. The companion's
existing metadata comprehensions are preserved.

Each fixture must have the generic-path marker and two compiled blocks, compile
to a portable executable, and survive serialization/loading. An independent
physical-coordinate interpreter compares ordered shapes and every multilinear
basis coefficient for the selected bounded metadata. Additional tests exercise
extent mismatches, scalars, fresh singleton axes, original zero/singleton axes,
unsupported internal RHS insertion, unused undefined helper expressions,
nontrivial embeddings, reuse at rank 128, auxiliary permutations, and reversal transfer through
contractions. End-to-end rewriting tests check user guards and domain skipping.

## Remaining limits

This is the restricted insertion reduction, not arbitrary broadcast
completeness. Internal RHS insertions, uneliminated source insertion under
contraction, reductions over fresh axes, multiple additive components sharing
an insertion, and supplied inner RHS axis restrictions are outside this generic
path. Existing sound schemas may still handle some such rules separately.

The path is available through the explicit native `--core` backend. The default
symbolic inference pipeline and C++/Rust generated adapters are not migrated;
core DAG artifacts continue to require the native portable-program runtime.
The paper and its frozen audit remain unchanged. These engineering tests do
not establish unrestricted completeness or constitute a machine-checked proof.

For the wider XLA audit, see the [current core classification](../../labs/attribute_inferences/xla_rewrite/coverage/reports/core_classification.md).
The 8 generic insertion reductions join 13 direct theorem forms. Actual native
constructive dispatch currently covers 18 of those 21 forms: three existing pure
reversal forms still select specialized schemas. The signed core machinery in
this extension does not yet change that standalone dispatch.
