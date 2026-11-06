# Bounded early rejection with sparse inputs

> Historical research material. Implementation references and reproduction
> instructions describe the earlier approach. See the
> [current synthesis design](../../../design/attribute-synthesis.md) and
> [research index](../README.md) for present scope.

> Host-function APIs shown here describe earlier generated artifacts. The
> current frontend uses pure builtins and descriptor constructors; the extended
> rule backend and application integrations await migration.


The explicit-attribute checker is implemented. It derives a polynomial degree
bound from composed semantic IR, chooses legal small LHS configurations, and
searches sparse inputs for a failure. It never recognizes operation names.

The guarantee comes from the [sparse witness theorem](sparse_counterexamples.md):
for fixed legal metadata, a nonzero difference polynomial of degree at most `D`
has a witness with at most `D` nonzero input entries. Nonzero values come from
`{1, ..., D}`. Degree-zero differences use the all-zero input. Repeated captures
share entries.

This is complete value coverage for a checked metadata configuration, under
ideal rational/real arithmetic. It does not establish a sufficient rank or shape
cutoff. The configuration portfolio and attribute samples are bounded search.

## Run inside the compiler

```sh
bazel run //:tepl -- check tests/fixtures/semantic/rules/explicit.tepl \
    --semantic-policy algebraic --verify-small
```

`--verify-small` also works with `generate`. For explicit rules without guards,
shape declarations, or derivations, it tests uniform input shapes `[]`, `[2]`,
`[2, 2]`, and `[2, 3]`, with up to eight legal source attribute assignments per
shape tuple. Each query uses at most 100 ms and 100,000 work units. The configured
solver timeout can reduce the query timeout further. Only confirmed failures
produce errors. The checked configuration count is retained in the inference
proof notes; successful checking is not an arbitrary-rank certificate.

Rules with declarations, guards, or derivations defer this check to the generated
Rust entry point. The compiler cannot execute application-provided host functions.
Rules without supported semantics retain ordinary TEPL behavior.

## Run with application host functions

Each generated rule module exports:

```rust
pub fn verify_small<F: HostFunctions>(
    functions: &F,
    options: &VerificationOptions,
) -> VerificationReport;
```

Call it in an application test or a build validation executable. Supply the same
host implementation used by the rewrite matcher:

```rust
use crate::ir::builtins::verification::VerificationOptions;
use crate::ir::rules::explicit::rule_transpose_host_copy;

let report = rule_transpose_host_copy::verify_small(
    &functions,
    &VerificationOptions::default(),
);
report.reject_counterexample().expect("invalid TEPL rewrite");
println!("{:?}: {}", report.status, report.reason);
```

Build and expose the native runtime before running the validation executable:

```sh
bazel build //src/core:libtepl_semantic_runtime.so
export TEPL_SEMANTIC_RUNTIME="$PWD/bazel-bin/src/core/libtepl_semantic_runtime.so"
cargo test --manifest-path path/to/application/Cargo.toml
```

The host functions execute on concrete metadata. False guards and fallible calls
returning `None` skip the configuration, following matcher behavior. Derivations
run in source order and can consume earlier derived descriptors. Functions should
have the same deterministic behavior in validation and matching. No function body
needs translation to Z3. The harness checks the returned RHS attributes exactly;
it does not search for replacement attributes.

The harness also runs generated source metadata programs and declaration checks
before guards. RHS metadata errors, shape differences, and dtype differences are
reported separately from value counterexamples.

## Configure the search

Defaults are:

| Option | Default |
| --- | --- |
| Maximum portfolio rank | 2 |
| Dimension choices, in priority order | `[2, 1, 0, 3]` |
| Shape tuples | At most 64 |
| Source attribute assignments per shape tuple | At most 8 |
| Total native/harness time budget | 2,000 ms |
| Per-query timeout | 100 ms |
| Per-query work budget | 100,000 |
| Polynomial-degree limit | 8 |
| Dtype for undeclared inputs | `F32` |

The portfolio first tries equal shapes at each small rank, then varies individual
input shapes. It is not the full Cartesian product. A declared exact dtype takes
precedence over the default dtype. Dtype choices are not exhaustively enumerated.
The automatic portfolio supports rank limits up to eight and at most sixteen
dimension choices. Use explicit configuration tuples for other shapes.

For a specific matrix example:

```rust
let options = VerificationOptions {
    configurations: vec![vec![vec![2, 3], vec![3, 4]]],
    query_timeout_ms: 1_000,
    ..Default::default()
};
```

Tuples follow the original-input capture order in the composed semantic plan,
which is the first LHS traversal order. Bound intermediates are computed from
those inputs rather than supplied independently. Empty dimensions are included
when legal. Skip counts make vacuous searches visible.

## Result meanings

| Status | Meaning |
| --- | --- |
| `Counterexample` | A legal applicable source configuration fails RHS metadata checks or has an independently confirmed value witness. |
| `NoCounterexampleFound` | At least one applicable configuration was verified, and no query was inconclusive. Untested configurations remain unverified. |
| `Unknown` | No applicable configuration was verified, or some query was unsupported or exhausted its budget. |

`reject_counterexample()` fails only for a confirmed counterexample. It does not
require that every rule be proved valid. Inspect the status and coverage counts
if the application needs a stricter policy.

Value witnesses include input shapes, concrete attributes, nonzero entries,
output coordinates, and unequal output values. The existing polynomial evaluator
independently confirms each witness at its distinguishing output coordinate.
Metadata-only failures have no nonzero entries or output values.

## Implementation and limits

1. Derive degree bounds from primitive IR after composing tensor reads.
2. Instantiate source legality in Z3, sharing repeated attribute descriptors.
3. Block exact sampled source assignments. RHS legality never filters sampling.
4. Execute generated Rust declarations, guards, and host derivations.
5. Check RHS metadata using the concrete returned attributes.
6. Try the existing exact indexed normalizer. A successful proof avoids SMT
   search; failed normalization does not establish inequality.
7. Allocate `D` symbolic sparse slots with distinct enabled entries.
8. Expand bounded output and reduction domains, then query inequality in Z3.
9. Confirm a SAT witness with the independent polynomial evaluator.

Supported arithmetic is constants, reads, addition, subtraction, multiplication,
and finite sums, with metadata-dependent branches and indexing. Tensor-value
branches, division, arbitrary nonpolynomial operations, and IEEE floating-point
proofs are outside this checker. Explicit index-list attributes are supported;
sampled source fields need an `is_valid_axis_list` bound. Boolean inputs are not
analyzed as arbitrary real-valued tensors. `auto`, opaque attribute
fields, and rules that cannot lower to the current semantic plan return `Unknown`.

Sparse storage does not eliminate reduction iteration. This implementation
unrolls small reductions and output spaces; large examples may return `Unknown`.
Arithmetic normalization can expand expressions, and witness confirmation retains
the existing polynomial term limit. SMT calls obey their remaining deadline;
individual host calls and native expression operations are not preemptible.

The existing verifier already uses polynomial expressions. Sparse checking is
not automatically faster. Measure construction time, solver time, and failures
found under equal budgets before making a performance claim.
