# Binding and rule evaluation

The Rust and C++ runtimes match operation nodes and retain each node's attributes
with its tensor bindings. One e-class may contain several matching nodes with
different attributes. Those values cannot be recovered from an arbitrary class
representative.

For example, matching `(p: transpose X)` against two equivalent nodes can yield
`X=A, p.attrs=[1,0]` and `X=B, p.attrs=[0,1]`. Each pair remains a separate witness.
Repeated tensor captures compare canonical e-class IDs. Repeated attribute
captures compare complete attribute values.

## Application order

For each concrete witness, the runtime:

1. Binds declared shape and dtype restrictions.
2. Evaluates source-only `where` bindings and conditions in source order.
3. Evaluates derivations, including synthesized `auto` expressions.
4. Evaluates `where` bindings and conditions that depend on derived descriptors.
5. Prepares and validates the complete replacement.
6. Inserts replacement nodes and unions the roots.

Local bindings remain available to later conditions in the same block. `derive`
has a separate local scope and cannot read `where` locals. Boolean operators and
conditionals evaluate only the required branch. Comprehensions restore their
enclosing local scope after evaluation.

The egg and egg-c adapters rematch using current canonical bindings. Each tensor
binding stays paired with its attribute witness. Metadata checks run against
current facts.

Conditions currently run after structural matching. Early pruning is an
optimization for future work; it must preserve statement order, scope, and
short-circuit behavior.

Missing required metadata skips a candidate. Evaluation errors reject the
candidate and record rule/source diagnostics in `analysis.rejections()`.
Conflicting metadata and resource limits stop the checked runner. A failed
replacement does not insert partial RHS nodes.
