# Attributes, conditions, and derivations

Operation attributes describe configuration such as transpose permutations and
contraction axes. A descriptor holds the operation's complete attributes,
separately from operands and tensor metadata.

## Capture and reuse

Capture a descriptor on the LHS with `@name`, then reuse it on the RHS:

```tepl
rule transpose_negate {
    (t.transpose[@p] (t.negate X))
    => (t.negate (t.transpose[@p] X))
}
```

These examples assume the tensor dialect is imported as `t`. Repeated captures
of one descriptor require matching attributes. Every RHS operation must accept
the descriptor's schema.

## Construct and update descriptors

Use a constructor in `derive` to calculate attributes directly:

```tepl
rule compose_transposes {
    (t.transpose[@outer] (t.transpose[@inner] X))
    => (t.transpose[@combined] X)

    derive {
        let permutation = gather(@inner.permutation, @outer.permutation);
        @combined = { ..@outer, permutation: permutation };
    }
}
```

`gather(p, q)` computes `[p[q[i]] for i in range(len(q))]`. No application
callback is required.

- The RHS operation determines the expected schema. A spread can also supply it.
- One spread is allowed, before explicit fields, and must use the same schema.
- Explicit fields override copied values. Duplicate explicit fields are errors.
- Without a spread, declared defaults apply and omitted optional fields become
  `none`. Missing required fields are errors.
- Descriptors are immutable. Derivations can read captures and earlier derived
  descriptors, but cannot reference themselves or later derivations.
- Spreading preserves every field, including precision and algorithm settings.
  The rule author must ensure preserving those settings is valid.

A constructor without a spread uses the same rules:

```tepl
derive {
    @combined = {
        permutation: gather(@inner.permutation, @outer.permutation)
    };
}
```

Schema inference can follow chains of descriptor copies. A constructor with
neither a schema-constraining use nor a spread is rejected as ambiguous.
Opaque values such as regions may be copied from existing descriptors; there
is no constructor for arbitrary region bodies or tensor payloads.

## Guards and local expressions

`where` contains Boolean guards and ordered local bindings:

```tepl
where {
    let rank = len(Dims);
    len(@p.permutation) == rank;
    all([@p.permutation[i] == i for i in range(rank)]);
}
```

Rule expressions support arithmetic, Boolean operators, homogeneous nested
lists, indexing, conditionals, comprehensions, and
[builtins](../references/built-ins.md). Comprehensions iterate over lists and
their variables are scoped to the body. `++` is shorthand for `concat`.

Locals are visible after their declaration within their own block. `where`
locals are unavailable in `derive`. A local cannot redefine an existing local,
tensor capture, dimension, or dtype binding. Comprehension variables may shadow
names within their body. Optional fields can be copied, assigned `none`, or
compared for equality; numeric/list operations require nonoptional values.

Integer arithmetic stays checked: indices use unsigned 64-bit values and signed
integers use `i64`, without implicit signed/unsigned conversion. Invalid indexing,
overflow, or failed builtin evaluation rejects a candidate before insertion.
`&&`, `||`, and conditionals preserve short-circuit evaluation. Every executed
local initializer is evaluated in source order, including unused locals.

## Explicit construction and auto

An explicit constructor specifies how attributes are computed. `[auto]` asks
the compiler to synthesize supported attributes from operation semantics.
These remain separate forms. Both must satisfy operation legality and the
applicable semantic checks. Matching output shape and dtype alone does not
prove numerical equivalence; distribution and reassociation require an explicit
arithmetic policy.

The frontend and symbolic checker support explicit constructors. The symbolic
checker accepts its supported metadata fragment and diagnoses expressions it
cannot lower or prove. Rule constructors and extended rule expressions do not
yet have Rust/C++ code generation or an execution interpreter; generation
reports a diagnostic before writing files. `$` host calls and host `fn`
parameters are no longer part of TEPL.
