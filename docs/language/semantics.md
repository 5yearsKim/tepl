# Semantic verification

A replacement can have the right shape and dtype while computing the wrong
values. Transposing a square matrix is a simple example:

```text
X = [1 2]       transpose(X) = [1 3]
    [3 4]                      [2 4]
```

Both tensors have shape `[2, 2]`. Removing the transpose would still change the
computation. Semantic verification checks more than metadata compatibility.

## Describe axes and values

A shape records dimension sizes. An `axes` block also tracks which logical
dimensions an operation preserves, reorders, introduces, or reduces. Two
same-sized dimensions can still be different logical axes.

A `semantics` block describes the value at an output coordinate. TEPL combines
these two descriptions to reason about the operation's meaning.

## Declare what an operation means

The [tutorial dialect](../../examples/projects/transpose/dialects/tensor.tepl) defines:

```tepl
dialect Tensor {
    op transpose(input: tensor) -> tensor {
        attrs { permutation: axis[] }
        dtype(t) { yield t }
        axes {
            bind S { input: all_axes(input) }
            assert is_valid_axis_list(attrs.permutation, rank(input))
            assert len(attrs.permutation) == rank(input)
            yield permute(S, attrs.permutation)
        }
        semantics { yield input }
    }
}
```

Read the three contracts together:

- `dtype` preserves the input dtype.
- `axes` binds the input axes, validates the permutation, and orders the output axes.
- `semantics` reads the input value at the logical coordinates established by those axes.

`yield input` does not make transpose an identity operation: the axis ordering is
part of its meaning. Logical axes remain distinct even when their sizes agree.

## Verify a rule

The tutorial's identity rule says:

```tepl
rule remove_identity {
    X: [Dims...]
    (node: t.transpose X) => X
    where node.attrs.permutation == range(len(Dims))
}
```

The condition requires the identity permutation. Under that condition, input and
output have the same ordered axes and read the same elements.

Run:

```sh
bazel-bin/tepl check examples/projects/transpose --rule remove_identity
```

The report includes `policy=strict status=proved`. Remove the condition and
check again in a copy of the rule: an arbitrary transpose cannot be proved equal
to its input, even if you constrain that input to a square shape.

## Understand the guarantees

| Check | Establishes |
| --- | --- |
| Language checking | Names, signatures, types, and attribute expressions are well formed |
| Runtime metadata checking | Concrete operations are legal under their contracts and known root shape/dtype are preserved |
| Semantic verification | Supported value, coordinate, and axis relationships preserve the declared meaning under the selected policy |

Rules without semantic contracts do not acquire a value-equivalence proof from
shape checking. Unsupported proofs are reported as unproved; that does not by
itself establish that a transformation is mathematically false.

## Choose an arithmetic policy

The default `strict` policy preserves arithmetic structure while allowing
supported coordinate reasoning such as transpose composition. The `algebraic`
policy permits implemented ideal-arithmetic laws, including supported
reassociation and distribution:

```sh
bazel-bin/tepl check examples/advanced/semantic_verification/rules/contraction_algebra.tepl \
  --semantic-policy algebraic --rule associate
```

An algebraic proof does not establish IEEE floating-point rounding, NaN,
signed-zero, or overflow equivalence. Choose a policy consistent with the host's
numerical requirements. TEPL checks declared semantics; the host must implement
them faithfully.

Next: [infer replacement axis attributes](axis-inference.md).
For details, see [scalar semantics](../references/scalar-semantics.md) and
[metadata reasoning](../references/metadata-reasoning.md).
