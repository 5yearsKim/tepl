# Core constructor and certified backend

`tepl compile rule.tepl --semantic-policy algebraic --core -o rule.json`
produces a portable program with capability `tepl.core-dag.v1`. `validate`
rechecks it; `rewrite` executes it in the native runtime. `check --core`
uses the same sufficient guard checker. `core-emit` remains inspection only.
Generated C++/Rust exports reject this capability until those runtimes
implement it.

The complete constructor admits only the ordered axis/value templates and independent
automatic target descriptors, without supplied derive helpers. It additionally
checks canonical dtype transport: identity for permutations, numeric type
preservation for sums, and equal numeric input types with preservation for
contractions/addition. Target numeric operations require numeric source legality;
custom dtype behavior and newly introduced numeric premises are rejected.
It emits the
mathematical check/build DAG, then proves its predicate from the *original*
source constraints and guard using `inference::certify`:

- Sum movement: lift dot lists through the sum survivor injection, reduce
  the newly retained free axes at the root.
- Unary permutation/sum chains: a target sum can remove the complement and
  a target permutation can order survivors. Require the respective target
  operation whenever it occurs on the source side.
- Equal expanded component trees: copied source attributes witness feasibility.
- Batched two-dot reassociation: require the original outer left batch list
  to equal the prefix determined by the inner left batch list length.
- Otherwise require exact ranks and *all* source axis lists from positive
  guard conjunctions/capture constraints. Predicate interpretation at extent
  two is an exact abstraction of that topology: port classes and order do
  not depend on extent values on legal inputs. No tensor values are sampled.

Only these cases are certified; arbitrary guard implication is unresolved.
The universal structural cases do not impose rank bounds. The fixed-layout
abstraction caps rank at 256 and total input ports at 4096, with bounded
interpretation work. Source inequalities remain runtime conditions.

Emission completeness holds in the original extent-at-least-two domain.
Builder *soundness* also holds for legal zero/singleton extents. Native
`executeSound` supports that larger domain but does not label a false
canonical predicate as impossibility.

The loader validates the canonical DAG by regeneration, binds the program
schedule to actual operation contracts/fields, checks descriptor assignment,
and reconstructs the guard facts independently of the compiler plan. It
caches the emission once. Native graph application relies on this immutable,
validated snapshot and validates the caller-owned graph, rather than repeating
whole-program guard certification. Native rewriting replays captured metadata and
interprets the DAG under normal resource budgets; it does not synthesize per
match or add a feasibility guard. Declaration/backend agreement is trusted,
as with existing portable programs; this is not authentication of arbitrary
operation value semantics.

`//labs/accepted_core:acceptance_test` covers all six symbolic families and
5–10-dot chains, serialization, numeric integer reference evaluation of
actual rewrites (including zero/singleton/scalar cases), corrupted DAGs,
missing capabilities, arithmetic policy, missing guards, and disjunctions.
`//labs/accepted_core:benchmark` separately measures compile, load/validation,
and actual native graph rewriting, with reference evaluation outside timing.

## Sound schema extensions

`--certified` includes the canonical `--core` constructor and additionally admits
recognized view/reindexing graph pairs through
`src/inference/extensions.{h,cc}`. Their programs carry the extra capability
`tepl.core-schemas.v1`. They do not enlarge the completeness theorem's domain:

- Any reverse chain to one reverse, or to identity when parity cancels.
- Any negation chain by parity; negation through summation.
- Broadcast composition when the outer insertion appends its dimensions.
- Distribution of identical broadcasts through pointwise addition/product.
- Reverse through a broadcast, discarding only inserted-axis reversals.
- Transpose of any insertion-only broadcast; insertion-free broadcast to transpose.
- Reverse over addition with a scalar broadcast summand.
- Reduction through broadcast when only original dimensions are reduced.
- Moving reversals of contracted dimensions between dot operands.
- Full aligned dot to reduced product; pure outer product to broadcasts/product.
- Scalar-factor extraction from either operand of a generalized dot.
- Moving a broadcast across dot by summing paired created contraction axes,
  checking retained free-axis order and remapping remaining dot lists.
- Scalar-factor movement between nested products and broadcast embeddings.
- Leading unit-axis removal with reinsertion before appended broadcast extents.
- Identical operation trees with supported descriptors copied.

Templates are recognized from declarations, not operation spellings. Shape-only
contracts in portable programs omit the frontend's `axis_fields` annotations;
the loader binds fields by their recognized semantic roles and retained extent
usage. Arithmetic scalar meanings remain part of the declaration/backend TCB.
Paired pointwise operations must use the same operation identity.

The compiler verifies every supplied field against the generated DAG expression,
including inserted extents, and checks that original derived metadata expressions
are accounted for. Unsupported helpers and mixed fields not justified by this
check reject. Guard facts come only from positive equalities/capture constraints;
a conservative, finite congruence checker cannot add applicability guards.
Unrecognized graph pairs reject before guard checking. At load, exact structural
regeneration, schedule binding, canonical dtype checks, and guard recertification
run again. Execution uses the metadata DAG, never the default symbolic backend.
Zero and singleton extents are included in the soundness statement. Failures of
schema applicability are `Unsupported`, not an infeasibility decision.

The paper's certified-schema lemma proves the index and finite-sum laws; proofs
are handwritten, and implementation correspondence is trusted. Schema acceptance
is sound but is not a complete decision procedure for broadcast or reverse.

`bazel test -c opt //labs/accepted_core:extensions_test` exercises public
compile/serialize/load/rewrite, an independent dense tensor reference, ranks
0–4 and 128, arbitrary/unsorted selectors, empty/singleton domains, guarded
cancellation, different broadcast embeddings, batched dot reversal, reduction,
corrupted schedules/DAGs/capabilities/guards, wrong semantics/dtypes, mismatched
supplied fields, and resource limits. The original `acceptance_test` retains
its previous whole-axis families and chains.

The two CLI flags are mutually exclusive. `--core` does not admit any schema
extension or mixed/supplied constructor; `--certified` is the combined sound
backend. Internal portable fields/capability names retain `core` for wire
compatibility; these do not broaden the completeness theorem. All exports of
this DAG path remain native portable-runtime only.

General broadcast/transpose uses a checked linear metadata primitive. The
compiler recognizes the exact supplied finite inverse/scatter comprehensions,
including both loop bounds, indices and zero branches. It proves their values
and definedness from source permutation legality and matches every helper to a
safe constructor subexpression before erasing helpers. Altered bounds, scatter
values and unused undefined helpers reject. Unit-axis removal is recognized
only with the complete valid-selector/positive/unit-upper-bound contract and
identity dtype transport. The numerical suite additionally covers anisotropic
and zero extents, all small output permutations, nested broadcasts with a
nonunit scalar, batched scalar extraction, created-plus-original contracting
axes, and a second singleton axis whose supplied selector must be preserved.
