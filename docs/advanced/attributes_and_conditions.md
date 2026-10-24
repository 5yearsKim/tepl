# Nodes, attributes, conditions, and derivations

Name an operation occurrence with `(name: operation operands...)`. Names are
optional on both sides of a rule. They identify nodes, without capturing extra
tensors or changing the operation's semantics. A name must be unique within the
rule and must not conflict with a tensor, dimension, dtype, or local binding.

## Match and reuse attributes

Every LHS operation captures its attributes implicitly. Use a node's `attrs`
property when a condition or replacement needs them:

```tepl
rule transpose_negate {
    (source: t.transpose (t.negate X))
    => (t.negate (t.transpose[source.attrs] X))
}
```

`source.attrs` is the complete descriptor, and `source.attrs.permutation` reads
one field. An inline `[source.attrs]` copies the descriptor to an operation with
the same schema. Two unnamed LHS operations have independent attribute captures.
Reusing `[source.attrs]` on another LHS operation requires equal descriptors.

## Infer or construct replacement attributes

Omitting RHS attributes requests inference from operation semantics. No `auto`
annotation or node name is needed:

```tepl
rule compose_transposes {
    (t.transpose (t.transpose X)) => (t.transpose X)
}
```

To specify attributes instead, name the RHS node and assign its whole descriptor
in `derive`:

```tepl
rule compose_transposes_explicit {
    (outer: t.transpose (inner: t.transpose X))
    => (result: t.transpose X)
    derive {
        let permutation = gather(inner.attrs.permutation, outer.attrs.permutation);
        result.attrs = { ..outer.attrs, permutation: permutation };
    }
}
```

`gather(p, q)` computes `[p[q[i]] for i in range(len(q))]`. A complete copy can
also be written `result.attrs = outer.attrs;`. An explicit definition replaces
default inference for that node; competing definitions are errors. LHS
attributes are read-only. Derivations can read source attributes and earlier
derived descriptors, but cannot read themselves, later derivations, or pending
inferred attributes.

The RHS operation supplies the expected descriptor schema. A constructor permits
one initial spread of the same schema; explicit fields override copied fields.
Duplicate explicit fields are errors. Without a spread, defaults apply and
omitted optional fields become `none`; missing required fields are errors.
Spreading preserves every field, including precision and algorithm settings.
Opaque fields may be copied but cannot be constructed arbitrarily.

## Semantic axis properties

Group names come from the operation's axis declaration. For the example tensor
dialect, a dot has batch group `B`, left-free `I`, right-free `J`, and reduced `K`.

| Property | Meaning | Type |
| --- | --- | --- |
| `node.axes.B.members` | Logical identities in group `B`, without order | `LogicalAxisSet` |
| `node.axes.B.order` | Identities in the group's declared order | `LogicalAxis[]` |
| `node.axes.B.positions` | Those identities' positions in the node's output | `index[]` |
| `node.axes.order` | Complete actual output order | `LogicalAxis[]` |
| `A.axes.order` | Input capture `A`'s physical order, in logical identities | `LogicalAxis[]` |
| `node.attrs.field` | Actual operation attribute | Declared field type |

For a dot output ordered `[batch0, batch1, left0, right0]`, `node.axes.B.order`
is `[batch0, batch1]` and `node.axes.B.positions` is `[0, 1]`. These logical
identities are not integer positions or extents. Distinct equal-sized axes
remain distinct; source batch/contraction attributes establish shared identities.
A reduced group has no output positions, so `node.axes.K.positions` is rejected.
A group's order need not equal its order in the output, for example after a
permutation.

Logical membership and group-order properties currently recognize source
contraction/permutation groups. Source group positions also use the general
symbolic mapper, including operations without attributes. RHS attributes must be constructed and verified from source metadata.
Runtime RHS feasibility queries are not supported; use source restrictions
that the compiler can prove sufficient.

## Verified restriction suggestions

Run `check --semantic-policy=algebraic --suggest-where --rule NAME` to search
for a concise sufficient source restriction. For example, reassociation can
be proved under:

```tepl
where {
    outer.attrs.lhs_batching_dimensions == inner.axes.B.positions
}
```

The compiler verifies the whole rule under each proposed restriction using a
fresh proof context. It also finds a concrete legal metadata witness to avoid
vacuous recommendations. It tries individual predicates and small conjunctions,
then removes clauses when verification still succeeds. The search is bounded;
a recommendation is verified sufficient, not necessarily weakest or shortest.
An unproved rule remains unproved until the user adds a restriction and the
normal compiler verifies it.

Only written conditions control applicability. `axis_domain()` and runtime
RHS feasibility queries have been removed. Source group order, membership,
positions, and input axis order lower to ordinary metadata expressions.
Singleton and zero extents remain valid where operation semantics permit them.
Explicit RHS descriptors may be combined with automatic descriptors; final
verification checks all definitions and generated constructors together.

## Guards and local expressions

`where` contains ordered local bindings followed by one Boolean expression.
Only bindings end with `;`. Combine predicates with `&&` or `||`. Omit the block
for unconditional rules; inherited rules require both guards. Locals are visible
after their declaration within their own block; `where` locals are unavailable
in `derive`. Comprehensions have their own variable scopes.

Expressions support arithmetic, lists, indexing, comprehensions, conditionals,
and [builtins](../references/built-ins.md). `++` means `concat`. Indices use
unsigned 64-bit integers, signed integers use `i64`, and arithmetic is checked.
Invalid indexing, overflow, or failed evaluation rejects a match before
insertion. Boolean operators and conditionals short-circuit. Local initializers
execute in their phase even when unused. Conditions depending on explicit RHS
descriptors execute after derivation; independent source conditions execute
first. Cached axis analysis is shared by feasibility and attribute construction.

Both inferred and explicit attributes must satisfy legality and semantic checks
under the selected arithmetic policy. Native C++ execution supports the new
conditions; Rust/code-generation backend migration remains deferred.

Legacy `@` descriptor bindings, `->` group projections, `[auto]`, and partial
attribute pins have been removed. Name nodes, use `.attrs` or `.axes`, and
provide either a complete descriptor in `derive` or leave it for inference.
