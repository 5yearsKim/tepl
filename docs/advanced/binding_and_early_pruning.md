# Binding and rule evaluation

The Rust runtime matches operation nodes and retains each node's attributes with
its tensor bindings. One e-class may contain several matching nodes with
different attributes. Those values cannot be recovered from an arbitrary class
representative.

For example, matching `(p: transpose X)` against two equivalent nodes can yield
`X=A, p.attrs=[1,0]` and `X=B, p.attrs=[0,1]`. Each pair remains a separate witness.
Repeated tensor captures compare canonical e-class IDs. Repeated attribute
captures compare complete attribute values.

## Application order

For each concrete witness, the runtime:

1. Binds declared shape and dtype restrictions.
2. Evaluates `where` statements in source order.
3. Evaluates derivations, including synthesized `auto` expressions.
4. Prepares and validates the complete replacement.
5. Inserts replacement nodes and unions the roots.

Local bindings remain available to later conditions and derivations. Boolean
operators and conditionals evaluate only the required branch. Comprehensions
restore their enclosing local scope after evaluation.

The egg adapter rematches during application using current canonical bindings.
This recovers attribute witnesses without storing them in egg's tensor-only
substitutions. Metadata checks run against current facts.

Conditions currently run after structural matching. Early pruning is an
optimization for future work; it must preserve statement order, scope, and
short-circuit behavior.

Missing required metadata skips a candidate. Evaluation errors report a
diagnostic through the checked runner. A failed replacement does not insert
partial RHS nodes.
