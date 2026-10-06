# Tensor shapes

A **shape** is an ordered list of tensor dimension sizes. Its length is the
tensor's **rank**, and axis positions start at zero.

| Shape | Meaning |
| --- | --- |
| `[]` | A scalar: rank zero, one element. |
| `[5]` | A vector with five elements. |
| `[2, 3]` | A matrix with two rows and three columns. |
| `[0, 3]` | A rank-two tensor with no elements. |

Shapes describe tensor dimensions, independently of the [dtype](dtype.md) of each
element. For example, `f32[2, 3]` has six floating-point elements.

## Operation shape programs

An operation's `shape(...)` block checks operand shapes and computes its output
shape. Parameters receive dimension lists in operand order, not tensor values.
Their names can differ from the operand names.

This matrix multiplication example belongs inside a `dialect` declaration:

```javascript
op matmul(lhs: tensor, rhs: tensor) -> tensor {
    shape(l, r) {
        assert len(l) == 2 && len(r) == 2;
        assert l[1] == r[0];
        yield [l[0], r[1]];
    }
}
```

For `l = [2, 3]` and `r = [3, 4]`, the output is `[2, 4]`. The first assertion
checks rank before indexing; the second checks the contracting dimensions.
This operation defines shape inference only. A separate dtype block can define
its element-type requirements.

Statements execute in order. `let` creates an immutable local value, `assert`
requires a Boolean condition, and one final `yield` returns the dimension list.
Assertions remain active in optimized builds.

### Attributes can determine the result

A reshape operation can take its target shape from an attribute:

```javascript
op reshape(input: tensor) -> tensor {
    attrs {
        shape: index[];
    }
    shape(s) {
        let target = attrs.shape;
        assert product(s) == product(target);
        yield target;
    }
}
```

Input `[2, 3, 4]` can reshape to `[6, 4]`: both contain 24 elements.
`attrs.shape` is declared metadata supplied when constructing the operation.
This example uses explicit dimensions; it does not infer a dimension from `-1`.

### Fixed and variadic operands

The shape parameters must mirror the operation's operand structure:

| Operand signature | Shape signature | Parameter values |
| --- | --- | --- |
| No operands | `shape()` | No operand shapes. |
| Two fixed operands | `shape(a, b)` | Two dimension lists. |
| `inputs: tensor...` | `shape(inputs...)` | A list of dimension lists. |
| Fixed operand followed by `rest: tensor...` | `shape(first, rest...)` | One shape and a list of shapes. |

For three variadic operands shaped `[2, 3]`, `[2, 5]`, and `[2, 1]`, the parameter
is `[[2, 3], [2, 5], [2, 1]]`. `len(inputs)` is the operand count;
`inputs[1][1]` is the second operand's dimension at axis one. A variadic tail
can be empty, so check its length before accessing its first element.

## Shape constraints in rules

A rule declaration restricts what a pattern can match:

```javascript
X: [Batch..., M, K]
Y: [Batch..., K, N]
```

`M`, `K`, and `N` bind dimension sizes. Repeated names require equal values.
`Batch...` binds a dimension sequence, possibly empty; repeating it requires the
same batch dimensions. Here, `X` and `Y` must each have at least two dimensions.

`X: []` requires a scalar. `X: [...]` accepts any rank. Adding a dtype prefix,
such as `X: f32[M, K]`, also constrains the element type.

These declarations check matched metadata. They do not reshape tensors or add
new shape facts. Missing metadata cannot establish a required constraint.

## Expressions and inference

Shape programs support arithmetic, comparisons, list literals and indexing,
`attrs.field`, conditional expressions, and comprehensions. For example,
`[d * 2 for d in s]` doubles each dimension size. List arithmetic is explicit;
`s * 2` is not an elementwise operation.

Intermediate integers use checked signed 128-bit arithmetic. Yielded dimensions
must be nonnegative and fit `u64`. Only required branches of `&&`, `||`, and
conditional expressions are evaluated. Function calls must resolve to supported
builtins; `$` host calls are unavailable inside shape and dtype programs.
See [built-in functions](built-ins.md) for signatures and failure behavior.

Generated Rust and C++ evaluators reuse these programs for e-class analysis and
replacement validation. A result can be **known**, **unknown** when required
metadata or inference is unavailable, or **invalid** when a check fails.
Omitting a shape block supplies no generated shape inference; the host can
provide it. Checking source validates types and names, while concrete dimensions
are checked during evaluation.

Shape inference does not evaluate tensor contents or prove a rewrite's
mathematical equivalence. See the [tensor dialect](../../examples/sample/dialects/tensor.tepl)
for more operation definitions and the [core concepts](../2_core_concept_tepl.md)
for how dialects, rules, and graphs work together.
