# Shape definitions

> Implementation status: shape blocks, expressions, and builtin call syntax are
> parsed into a structured AST and checked by core. Parameters, local names,
> attribute fields, builtin signatures, and expression types are resolved. Shape
> programs are emitted as checked Rust operation evaluators. The compiler
> also supports the nested/opaque attribute types and `$` host-call syntax here.
> Existing rule-level shape declarations remain match-time constraints.

An operation's shape definition computes its output shape from operand shapes
and attributes. TEPL compiles that definition into a shared evaluator used
by e-class analysis and by RHS validation before a rewrite inserts nodes.
Every parameter is an operand shape or a variadic list of operand shapes, not
a tensor.
Operation attributes are available separately through `attrs`.

Rule declarations and operation definitions have different roles:

```tepl
// Existing rule syntax: restrict which tensors a rule can match.
X: [Batch..., M, K]
```

A rule-level declaration is shorthand for shape conditions in a `where` clause:
when searching the e-graph, inspect the captured tensor's shape, check its rank
and dimensions, and bind the named dimensions. Repeated names must have equal
sizes across captures. `Batch...` binds the prefix (possibly empty), and `M` and
`K` bind the final two sizes. A mismatch rejects the candidate match; it does not
change a tensor's shape or assert new facts into an e-class. Missing metadata
cannot establish the match condition.

Operation shape blocks provide the reusable operation-level counterpart:
`assert` checks input shape conditions, and `yield` computes the result shape.
They do not replace the rule's shape-pattern checks.

```tepl
// Operation syntax: describe how an output shape is computed.
op transpose(input: tensor) -> tensor {
    attrs {
        permutation: index[];
    }

    shape(s) {
        assert is_valid_axis_list(attrs.permutation, len(s));
        assert len(attrs.permutation) == len(s);
        yield gather(s, attrs.permutation);
    }
}
```

The operation examples below belong inside a `dialect` declaration. Their shape
blocks define shape behavior only. Optional [dtype policies](dtype_guide.md)
describe output dtypes separately; numerical rewrite legality remains a separate
responsibility.

Shape parameters bind to operand shapes in declaration order: `shape(s)` receives
the shape of the sole operand, and `shape(l, r)` receives the shapes of the first
and second operands. The generated caller supplies these lists automatically.
Parameter names are local to the block and need not match operand names. The
fixed parameters bind one-to-one to fixed operands; a zero-operand operation
uses `shape()`. A final variadic parameter binds a list of shapes as described
below.

## Variadic operands

Mirror the operation's trailing variadic operand with a trailing `...` parameter.
The parameter is a list of shapes, not a flattened shape and not a tensor tuple:

```text
op concatenate(inputs: tensor...) -> tensor
shape(inputs...)

Three operand shapes: inputs = [[2, 3], [2, 5], [2, 1]]
len(inputs) = 3                  // Number of operands
inputs[1] = [2, 5]               // Second operand's shape
inputs[1][1] = 5                 // Its dimension at axis 1
```

There may be at most one variadic parameter, and it must be last. A declaration
with fixed operands followed by a variadic tail uses `shape(a, b, rest...)`;
`a` and `b` are shapes, and `rest` is a possibly empty list of shapes. The shape
signature must match the fixed/variadic structure of the operation signature.
The `...` marks a parameter binding; it is not an expression spread operator.

Variadic parameters can be empty. An operation requiring at least one input
must assert `len(inputs) > 0` before using `inputs[0]`. Lists and comprehensions
support homogeneous nested lists; `concat([a, b], rest)` builds a list of shapes
without flattening their dimension lists. Nested comprehensions and `all` can
check conditions across operands and dimensions.

## StableHLO scope and metadata mapping

[`dialects/tensor.tepl`](dialects/tensor.tepl) contains a selected subset of
[StableHLO operations](https://openxla.org/stablehlo/spec), rather than a complete
StableHLO importer or verifier. Only operation names in that specification are
declared. `exp`, `mul`, and `dot` are TEPL aliases for `exponential`, `multiply`,
and `dot_general`. The `Scalar` dialect specializes `add` and `negate` to scalars.

StableHLO binary elementwise operations require matching operand shapes;
broadcasting is explicit through `broadcast_in_dim`. The `broadcast_shape`
helper remains available for dialects whose operations include implicit
broadcasting. It is not used for StableHLO `add`.

The examples use these explicit adapter conventions:

| TEPL representation | StableHLO information preserved |
| --- | --- |
| `attrs.shape` on `reshape` and `broadcast_in_dim` | Shape from the declared result type; this is adapter metadata, not a native StableHLO attribute |
| Separate dot/convolution dimension fields | Fields of `dot_dimension_numbers` / `dimension_numbers` |
| `precision[]` | Precision enum values `DEFAULT`, `HIGH`, or `HIGHEST` |
| `dot_algorithm?` | Optional structured dot algorithm attribute |
| `i64[][]` padding | Signed low/high padding pairs, one pair per spatial axis |
| `replica_groups` | Typed opaque explicit replica-ID groups or a mesh-axis group specification |
| `region` | Opaque reducer region, including its computation and identity; not a string such as `kind = "sum"` |
| `elements` | Typed constant payload, including its element type and shape |

These types extend the proposed attribute grammar with nested lists (`T[][]`),
optional values (`T?`), and the named opaque types above. Opaque metadata is
preserved for matching and construction; it cannot be inspected by arbitrary
shape expressions. Its host adapter must preserve semantic identity and perform
validation beyond the shape block. Materialize omitted stride/dilation/reversal
and precision defaults when constructing these descriptors; these examples use
fully populated lists, with two entries for a precision configuration.

`dot_algorithm` preserves `lhs_precision_type`, `rhs_precision_type`,
`accumulation_type`, `lhs_component_count`, `rhs_component_count`,
`num_primitive_operations`, and `allow_imprecise_accumulation`. These fields affect
numerical behavior, not output dimensions. The host validates their constraints.

`reduce`, `all_gather`, `all_reduce`, and `all_to_all` are single-result
specializations of StableHLO operations that also support variadic results.
`reduce` explicitly includes its scalar initialization operand. Reducer regions
are stored in adapter fields `body` or `computation`; in StableHLO they are
regions, not ordinary attributes. Multiple results and executable region syntax
remain outside this proposal. Shape checks do not validate reducer bodies,
element types, quantization, or numerical rewrite equivalence.

`all_gather` and `reduce_scatter` intentionally have no generated shape block:
the actual process-group size can depend on the process grid and partition count,
not just the replica-ID table width. Mesh-based groups also require context.
The host must supply the resulting metadata. `all_reduce` preserves shape;
`all_to_all` can compute dimensions from its explicit split count, while group
membership and collective configuration validation remain host responsibilities.

The shape/attribute sources are the StableHLO sections for
[dot_general](https://openxla.org/stablehlo/spec#dot_general),
[convolution](https://openxla.org/stablehlo/spec#convolution),
[reduce](https://openxla.org/stablehlo/spec#reduce),
[collectives](https://openxla.org/stablehlo/spec#all_gather), and
[constant](https://openxla.org/stablehlo/spec#constant).

## Builtin and host function names

Use unprefixed names for compiler-supported builtins and `$` for host functions:

| Form | Meaning | Example |
| --- | --- | --- |
| `name(...)` | Builtin | `gather(s, axes)` |
| `$name(...)` | Host-function call | `$infer_dot(X, W, @outer)` |
| `@name` | Descriptor reference | `@outer` |

A host function is implemented by the consuming application. Its signature may
still be inferred from use; `$` identifies the function category, not its argument
or return types. It does not request access to tensor contents. For example, the
existing Rust host interface passes tensor captures as `TensorInfo` metadata.
Host functions remain fallible: failure rejects a rewrite candidate.

Shape blocks allow only builtins in this initial proposal. Rule `where` and
`derive` expressions can call builtins and host functions. Builtins have fixed
signatures and compiler-defined semantics. An unprefixed call must resolve to a
builtin or an explicitly declared `fn` template parameter; unknown names are
errors and do not implicitly declare host functions.
A `$` name always refers to a host function, independently of builtins.

For example, the host portion of `rules/lora.tepl` uses:

```tepl
where {
    $is_broadcastable(Batch, WeightBatch);
    $is_reassociable(X, A, B, @outer, @inner);
}

derive {
    @xw  = $infer_dot(X, W, @outer);
    @xa  = $infer_dot(X, A, @outer);
    @out = $infer_lora_out(X, A, B, @outer, @inner);
}
```

`$is_broadcastable` retains its host-defined behavior; the prefix does not turn it
into a compiler builtin. The abstract-rule examples in `rules/abstract.tepl`
parameterize only operations and have no `where` clauses or host-function
parameters. Function-parameter syntax is outside this guide's scope.
`$` is part of TEPL call syntax, not part of the generated host
method's name. `%` remains the remainder operator and `&&` remains logical AND.

## Values and expressions

A shape is an ordered list of nonnegative dimension sizes. An axis list contains
positions in a shape, starting at zero:

```text
shape = [2, 3, 4]
axes  = [0, 1, 2]
shape[1] = 3
```

Shape helpers manipulate metadata, not tensor elements. For example, the
`gather` helper selects entries from a shape list; it does not construct a tensor
gather operation.

The shape parser supports:

- Shape parameters: `shape(s)`, `shape(l, r)`, or `shape(inputs...)`.
- Immutable local bindings: `let axes = range(len(s));`.
- Attribute fields: `attrs.dimension`.
- Integer and Boolean literals, list literals, and indexing: `[]`, `[2, 3]`, `s[i]`.
- Integer arithmetic, comparisons, list equality, and Boolean expressions.
- Conditional expressions: `if condition then a else b`.
- Bounded list comprehensions: `[expression for i in range(n)]`.
- Calls to a fixed library of pure shape helpers.
- Validity assertions and a final result: `assert condition;`, `yield shape;`.

There are no implicit elementwise arithmetic operations on lists. Use a
comprehension to calculate each element. `&&`, `||`, and conditional expressions
evaluate only the branches needed. A comprehension evaluates its iterable once,
binds a local variable for each element, and preserves iteration order.

Shape expressions use a signed 128-bit integer domain for dimensions, indices,
and intermediate arithmetic. It represents every unsigned 64-bit input dimension
and signed 64-bit attribute without narrowing. Integer literals must fit this
domain. Future evaluators must use checked 128-bit arithmetic and convert yielded
dimensions to unsigned 64-bit values, rejecting negative or out-of-range results.
Rule expressions retain their separate `Index`/`I64` types.

Scalar arithmetic permits signed intermediate values, such as padding offsets;
final dimensions must be nonnegative. Indexing requires a nonnegative, in-range
index. Arithmetic must be checked for overflow and division by zero. `/` is
integer division truncated toward zero; use `floor_div` or `ceil_div` when that
rounding is required. `%` uses the corresponding truncating remainder.

## Builtin helpers

These are directly available library functions in this proposal. Users do not
need `map`, `filter`, or `fold`; comprehensions and these helpers provide the
initial composition mechanisms. An implementation may share code between helpers
without exposing that implementation in the language.

Descriptive Boolean predicates use `is_`, such as `is_valid_axis_list` and
`is_disjoint`. The same convention applies to host predicates such as
`$is_broadcastable` and `$is_same_dtype`. Quantifiers `all` and `any`, the
membership predicate `contains`, and Boolean variables such as `has_empty_axis`
retain their conventional names.

| Helper | Meaning | Example |
| --- | --- | --- |
| `len(xs)` | List length; shape length is rank | `len([2, 3, 4]) == 3` |
| `range(n)` | Integers from zero to `n`, excluding `n` | `range(3) == [0, 1, 2]` |
| `concat(a, b, ...)` | Concatenate two or more lists | `concat([2], [3, 4]) == [2, 3, 4]` |
| `gather(xs, indices)` | Select positions in the supplied order | `gather([2, 3, 4], [2, 0]) == [4, 2]` |
| `exclude(xs, removed)` | Remove matching values, preserving order | `exclude([0, 1, 2], [1]) == [0, 2]` |
| `slice(xs, start, end)` | Contiguous section, end exclusive | `slice([2, 3, 4], 0, 2) == [2, 3]` |
| `replace(xs, axis, value)` | Copy a list with one entry changed | `replace([2, 3], 1, 5) == [2, 5]` |
| `sum(xs)` | Sum integers; empty list gives `0` | `sum([3, 5]) == 8` |
| `product(xs)` | Multiply integers; empty list gives `1` | `product([2, 3, 4]) == 24` |
| `all(bs)` | All Boolean entries are true; empty list gives `true` | `all([true, false]) == false` |
| `any(bs)` | At least one Boolean entry is true; empty list gives `false` | `any([true, false]) == true` |
| `contains(xs, x)` | Test membership | `contains([1, 3], 3) == true` |
| `is_valid_axis_list(ax, rank)` | Axes are unique and in `[0, rank)` | `is_valid_axis_list([1, 1], 3) == false` |
| `is_disjoint(a, b)` | Lists share no values | `is_disjoint([0], [1, 2]) == true` |
| `broadcast_shape(a, b)` | Compute a right-aligned broadcast shape, or fail | `broadcast_shape([2, 1, 4], [3, 4]) == [2, 3, 4]` |
| `min(a, b)`, `max(a, b)` | Scalar minimum and maximum | `max(0, -2) == 0` |
| `floor_div(a, b)`, `ceil_div(a, b)` | Integer quotient rounded down or up; `b > 0` | `ceil_div(7, 3) == 3` |

Use `all` for a condition that must hold for every entry, and `any` to detect
whether at least one entry satisfies a condition:

```tepl
let has_empty_axis = any([d == 0 for d in s]);
let has_singleton_axis = any([d == 1 for d in s]);
```

For the scalar shape `s = []`, both expressions are false. The identities are
`all([]) == true` and `any([]) == false`. These helpers consume Boolean lists;
they do not make list comprehensions lazy or suppress errors while constructing
the lists.

`gather` permits repeated indices; operations that prohibit them must separately
use `is_valid_axis_list`. `exclude` removes values, not positions: normally use it on
axis lists, then use `gather` to select the remaining dimensions. `slice` requires
`0 <= start <= end <= len(xs)` and does not clamp invalid bounds.

Broadcasting accepts equal dimensions or a dimension of size `1`, treating
missing leading dimensions as `1`. In particular, broadcasting `0` with `1`
produces `0`; it is not implemented as the maximum of the dimensions.

## Common transformations

### Elementwise operations

StableHLO elementwise addition requires identical operand shapes. Broadcasting
must be expressed separately with `broadcast_in_dim`.

```tepl
op add(lhs: tensor, rhs: tensor) -> tensor {
    shape(l, r) {
        assert l == r;
        yield l;
    }
}
```

### Reshape to an explicit target

The target comes from the adapter's `shape` field, populated from the StableHLO
result type; it cannot be deduced from the input shape alone. This version does
not support inferred-dimension markers such as `-1`.
An empty target list denotes a scalar with one element.

```tepl
op reshape(input: tensor) -> tensor {
    attrs {
        shape: index[];
    }

    shape(s) {
        assert product(s) == product(attrs.shape);
        yield attrs.shape;
    }
}
```

For example, `[2, 3, 4]` can reshape to `[6, 4]` because both contain 24 elements.

### Concatenation: a variadic list of shapes

All inputs must have the same rank and equal sizes outside the concatenation
axis. For `inputs = [[2, 3], [2, 5], [2, 1]]` and `dimension = 1`, the output is
`[2, 9]`. The nonempty assertion precedes the first list access.

```tepl
op concatenate(inputs: tensor...) -> tensor {
    attrs {
        dimension: index;
    }
    shape(inputs...) {
        assert len(inputs) > 0;
        let first = inputs[0];
        let axis = attrs.dimension;
        assert is_valid_axis_list([axis], len(first));
        assert all([len(s) == len(first) for s in inputs]);
        let other_axes = exclude(range(len(first)), [axis]);
        assert all([gather(s, other_axes) == gather(first, other_axes)
                    for s in inputs]);
        yield replace(first, axis, sum([s[axis] for s in inputs]));
    }
}
```

### Reduction

This example removes reduced axes. `[2, 3, 4, 5]` reduced over `[1, 3]` produces
`[2, 4]`.

```tepl
op reduce(input: tensor, init_value: tensor) -> tensor {
    attrs {
        dimensions: index[];
        body: region;
    }

    shape(s, init) {
        assert init == [];
        assert is_valid_axis_list(attrs.dimensions, len(s));
        yield gather(s, exclude(range(len(s)), attrs.dimensions));
    }
}
```

For another dialect's reduction variant that retains reduced dimensions with
size `1`, the yield could be the following. StableHLO `reduce` itself removes
the dimensions:

```tepl
yield [if contains(attrs.dimensions, i) then 1 else s[i]
       for i in range(len(s))];
```

### Slice: composing arithmetic across dimensions

This is why list helpers alone are insufficient without a composition mechanism.
Each output dimension needs its own arithmetic calculation.

```tepl
op slice(input: tensor) -> tensor {
    attrs {
        start_indices: index[];
        limit_indices: index[];
        strides: index[];
    }

    shape(s) {
        let axes = range(len(s));
        assert len(attrs.start_indices) == len(s);
        assert len(attrs.limit_indices) == len(s);
        assert len(attrs.strides) == len(s);
        assert all([attrs.start_indices[i] <= attrs.limit_indices[i]
                     && attrs.limit_indices[i] <= s[i]
                     && attrs.strides[i] > 0
                     for i in axes]);
        yield [ceil_div(attrs.limit_indices[i] - attrs.start_indices[i], attrs.strides[i])
               for i in axes];
    }
}
```

For input `[10, 8]`, start `[1, 0]`, limit `[9, 8]`, and strides `[2, 4]`, the
output is `[4, 2]`.

### General contraction

Contracting dimensions may be nonadjacent and there may be several. Batch and
contracting lists pair by position; free axes retain their original order.
Batch sizes must match, with no implicit broadcasting.

```tepl
op dot_general(lhs: tensor, rhs: tensor) -> tensor {
    alias: dot;
    attrs {
        lhs_contracting_dimensions: index[];
        rhs_contracting_dimensions: index[];
        lhs_batching_dimensions: index[];
        rhs_batching_dimensions: index[];
        precision_config: precision[];
        algorithm: dot_algorithm?;
    }

    shape(l, r) {
        let lc = attrs.lhs_contracting_dimensions;
        let rc = attrs.rhs_contracting_dimensions;
        let lb = attrs.lhs_batching_dimensions;
        let rb = attrs.rhs_batching_dimensions;

        assert is_valid_axis_list(lc, len(l));
        assert is_valid_axis_list(rc, len(r));
        assert is_valid_axis_list(lb, len(l));
        assert is_valid_axis_list(rb, len(r));
        assert is_disjoint(lb, lc);
        assert is_disjoint(rb, rc);
        assert len(lb) == len(rb);
        assert len(lc) == len(rc);
        assert gather(l, lb) == gather(r, rb);
        assert gather(l, lc) == gather(r, rc);

        let lf = exclude(range(len(l)), concat(lb, lc));
        let rf = exclude(range(len(r)), concat(rb, rc));
        yield concat(gather(l, lb), gather(l, lf), gather(r, rf));
    }
}
```

Example with two contracting axes:

```text
lhs shape: [2, 3, 5, 7]     lhs batch: [0]     lhs contracting: [1, 3]
rhs shape: [7, 11, 2, 3]    rhs batch: [2]     rhs contracting: [3, 0]

paired batch sizes:       [2] == [2]
paired contracting sizes: [3, 7] == [3, 7]
lhs free sizes:           [5]
rhs free sizes:           [11]
result:                  [2, 5, 11]
```

## Grammar and parser support

The EBNF below summarizes the shape syntax implemented in `grammar/Tepl.g4`.
An operation may have at most one alias, attribute schema, and shape block.
Alias and attribute properties may appear in either order; the shape block is
last. A shape block has exactly one final `yield`.

`assert`, `yield`, `if`, `then`, `else`, `for`, and `in` are reserved keywords.
`shape` is contextual, allowing fields such as `attrs.shape`. Builtin names
remain ordinary identifiers and are stored as unresolved AST calls. The parser
checks syntax only; names, types, operand/parameter correspondence, and builtin
signatures are checked by core after parsing.

Host calls extend rule expressions separately. The existing `constraintExpr`
production supplies their argument expressions; all other existing expression
forms remain available:

```ebnf
ruleCall       = [ "$" ], ID, "(", [ arguments ], ")" ;
arguments      = constraintExpr, { ",", constraintExpr } ;
```

The `$` prefix selects a host call; an unprefixed call selects a builtin.
In abstract rules, unprefixed calls may also reference explicitly declared
`fn` parameters, whose host bindings use `parameter = $function`.
Native builtin resolution is future work; the current checker rejects unknown
unprefixed calls in `where` and `derive` rather than treating them as host calls.
Abstract-rule operation bindings continue to use the existing grammar.

Shape blocks use the expression grammar below, which excludes `$` calls:

```ebnf
opProperties = aliasProperty, [ attrsProperty ], [ shapeBlock ]
             | attrsProperty, [ aliasProperty ], [ shapeBlock ]
             | shapeBlock ;

shapeBlock   = "shape", "(", [ shapeParameters ], ")",
               "{", { shapeStatement }, "yield", expr, ";", "}" ;
shapeParameters = ID, { ",", ID }, [ "..." ] ;
attrType     = ID, { "[", "]" }, [ "?" ] ;
shapeStatement = "let", ID, "=", expr, ";"
               | "assert", expr, ";" ;

expr         = "if", expr, "then", expr, "else", expr | logicalOr ;
logicalOr    = logicalAnd, { "||", logicalAnd } ;
logicalAnd   = equality, { "&&", equality } ;
equality     = comparison, [ ("==" | "!="), comparison ] ;
comparison   = additive, [ ("<" | "<=" | ">" | ">="), additive ] ;
additive     = multiplicative, { ("+" | "-"), multiplicative } ;
multiplicative = unary, { ("*" | "/" | "%"), unary } ;
unary        = ("!" | "+" | "-"), unary | postfix ;
postfix      = primary, { ".", ID | "[", expr, "]" } ;
primary      = ID
             | "attrs"
             | INT | "true" | "false"
             | ID, "(", [ expr, { ",", expr } ], ")"
             | "(", expr, ")"
             | "[", "]"
             | "[", expr, { ",", expr }, "]"
             | "[", expr, "for", ID, "in", expr, "]" ;
```

Shape parameters, `attrs`, earlier local bindings, and comprehension variables
form the expression scope. Tensor operand bindings are not available inside a
shape block; only their shapes or lists of shapes are passed in. Shape parameter names must be
distinct and cannot use the reserved name `attrs`. `attrs` is available only when
the operation has an attribute schema. Attribute fields are checked against that
schema. Helpers have checked signatures; there are no implicit conversions
between an integer, a Boolean, and a list. List equality compares lengths and
corresponding elements. `assert` expects a Boolean; `yield` expects a shape.

This proposal does not introduce user-defined functions, lambdas, unrestricted
loops, or recursion. These can be considered later if builtins and bounded
comprehensions prove insufficient.

## Core checking

Each operation's checked IR holds an optional shape program. Fixed parameters
have type `List<Integer>`; a variadic parameter has type `List<List<Integer>>`.
Parameters match the operation's fixed/variadic operand structure in declaration
order, independently of their names. All operation shape blocks are checked,
including those unused by any rule. An absent shape program remains absent.

Earlier `let` bindings are visible to later statements. Parameters and `let`
bindings cannot duplicate a name. A comprehension may shadow an outer binding;
its variable is visible only in the element expression, and its iterable is
checked in the outer scope. Calls resolve only to the 19 documented builtins.
`attrs.field` resolves a non-optional `index`, `i64`, or `bool` field, including
nested lists. Strings, enum values, optional fields, and opaque payloads cannot
be inspected in shape expressions.

Lists are homogeneous. Generic list helpers preserve the element type,
including Boolean and nested lists. `assert` requires `Bool`; `yield` requires
`List<Integer>`; a conditional requires `Bool` and matching branch types. List
indexing requires an integer, and comparisons/arithmetic follow the operator
rules above. Unknown names, wrong builtin arity/types, and mismatched shape
signatures produce source diagnostics.

Empty list element types are inferred from use: `yield []` is an integer list,
`all([])` is a Boolean list, and `concat([s], [])` is a list of shapes. A later
use can establish a local binding's type. An ambiguous empty list, such as an
unused `let xs = [];` or `len([])`, is rejected; no default element type is chosen.

Core preserves statements, assertions, and short-circuit branches without
executing them. Value-dependent failures such as negative dimensions, bad axis
values, out-of-range indexing, and arithmetic overflow remain runtime checks.
Successful checking does not prove shape equality or numerical rewrite legality.
`check` output includes resolved builtin IDs, local bindings, and expression
types. Shape IR owns its data and remains valid after the AST is destroyed.

## Evaluation and analysis boundary

The generated evaluator receives operand shapes, variadic lists of shapes, and
operation attributes.
It does not receive tensor values or read tensor contents. E-class analysis
obtains those shapes from child e-classes; RHS validation obtains them from
matched captures and previously inferred RHS operands.

Statements execute in source order. Assertions are always enabled, including in
release builds. A false `assert` returns a shape-validation error rather than
crashing the optimizer. Invalid helper calls and arithmetic errors also stop
evaluation with an error. Successful evaluation returns a concrete
shape. Unavailable metadata returns `Unknown`, distinct from a proven error;
the first version need not evaluate partially known dimension lists.

Generated RHS validation should reject a match when required shape checks cannot
be established, before inserting or unioning nodes. E-class analysis can reuse
the evaluator, but its merge policy must be specified separately: shape blocks
do not define how facts from equivalent alternatives combine.

An absent shape definition provides no generated inference. The host may supply
metadata; otherwise the result is `Unknown`, not an invalid operation. Graph
inputs are supplied by the host; `symbol` is not a StableHLO operation and is not
declared in this dialect. Constants obtain their shape from typed payload
metadata. Data-dependent output shapes may remain unknown. Partial dimensions,
multiple results, and region-dependent inference are possible future extensions.
