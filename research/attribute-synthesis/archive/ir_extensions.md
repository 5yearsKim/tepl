I would extend your current design in four steps. These cover useful XLA rewrites while preserving your existing graph and normalization approach.

> Historical research material. Implementation references and reproduction
> instructions describe the earlier approach. See the
> [current synthesis design](../../../design/attribute-synthesis.md) and
> [research index](../README.md) for present scope.

| Priority | Extension | What it enables | Effort |
|---|---|---|---|
| 1 | Strengthen existing axis reasoning | Merge reductions, move transpose through reduction/dot, simplify inferred guards | Small–medium |
| 2 | Elementwise multiply | Factoring, distribution, multiplication reassociation | Small: mostly an operation definition |
| 3 | Scalar constants, negate, subtract | A+0, A*1, A-A, cancellation, constant folding | Medium |
| 4 | Explicit broadcast | Compose broadcasts, absorb transpose into broadcast, move broadcast across arithmetic and some reductions | Medium; one meaningful axis-model extension |

These target families present in XLA’s arithmetic, broadcast, and reduction simplifications.

1. Finish the existing mechanism first.

Improve sequence identities, permutation composition, position lookup, and reduction-axis mapping. Some rules already produce constructors but leave guards that follow from ordinary legality. Discharging these guards gives more practical coverage without adding operations.

2. Add elementwise multiply through the dialect.

Its semantics is simply:

axes:      both inputs share S; output S
semantics: Product(lhs, rhs)

Your existing algebraic machinery can then handle:

A*C + B*C  →  (A+B)*C

3. Extend the scalar expression layer.

Add Const, and represent:

Negate(x)      = Product(Const(-1), x)
Subtract(x,y)  = Add(x, Negate(y))

Extend normalization to combine coefficients and repeated terms. Currently, repeated tensor factors and ambiguous term correspondence are rejected.

Also preserve the tensor’s output interface when simplification removes all reads—for example, A-A → 0 must retain A’s shape.

4. Add restricted broadcasting.

Start with explicit broadcast_in_dim: existing input axes map into output positions; additional output axes have declared extents.

The essential change is:

> A tensor can have output axes that its value expression does not depend on.

This supports constants and broadcasting naturally. Initially support broadcast composition and transpose/broadcast rewrites. Reducing a broadcasted axis additionally needs the law:

n-1
 ∑  e = n e  when e is independent of k.
k=0

I would postpone general reshape, slice, pad, gather, and convolution. They require coordinate arithmetic or piecewise indexing. The four steps above keep your core centered on axis mappings plus scalar algebra, with no need to introduce an e-graph or a new general-purpose solver.