#!/usr/bin/env python3
"""Small exact oracle for convolution_attribute_synthesis_proof.md.

Run directly with Python 3; this is not a default test-suite target.
Input and kernel entries are distinct formal variables. At each output,
the sorted (input coordinate, kernel coordinate) pairs are the bilinear
monomials, each with coefficient one. No floating-point sampling is used.
"""

from itertools import product


def construct(input_shape, kernel_shape, output_shape, offset, output_map, kernel_map):
    """The proposed metadata checker, on legal positive-domain source metadata."""
    rank = len(input_shape)
    if any(offset):
        return None
    for row in range(rank):
        for column in range(rank):
            if row == column:
                continue
            if output_shape[column] > 1 and output_map[row][column] != 0:
                return None
            if kernel_shape[column] > 1 and kernel_map[row][column] != 0:
                return None

    dilation = tuple(
        kernel_map[j][j] if kernel_shape[j] > 1 else 1 for j in range(rank)
    )
    if min(dilation) <= 0:
        return None
    residual = tuple(
        input_shape[j] - 1 - dilation[j] * (kernel_shape[j] - 1)
        for j in range(rank)
    )
    if min(residual) < 0:
        return None
    stride = tuple(
        output_map[j][j] if output_shape[j] > 1 else residual[j] + 1
        for j in range(rank)
    )
    if min(stride) <= 0:
        return None
    if tuple(residual[j] // stride[j] + 1 for j in range(rank)) != output_shape:
        return None
    return stride, dilation


def diagonal(values):
    return tuple(
        tuple(value if i == j else 0 for i in range(len(values)))
        for j, value in enumerate(values)
    )


def signature(input_shape, kernel_shape, output_shape, offset, output_map, kernel_map):
    """Expand exact monomials; return None if any source read is out of bounds."""
    rank = len(input_shape)
    outputs = []
    for output in product(*(range(n) for n in output_shape)):
        monomials = []
        for kernel in product(*(range(n) for n in kernel_shape)):
            access = tuple(
                offset[j]
                + sum(
                    output_map[j][i] * output[i] + kernel_map[j][i] * kernel[i]
                    for i in range(rank)
                )
                for j in range(rank)
            )
            if any(x < 0 or x >= n for x, n in zip(access, input_shape)):
                return None
            monomials.append((access, kernel))
        outputs.append(tuple(sorted(monomials)))
    return output_shape, tuple(outputs)


def enumerate_rhs(input_shape, kernel_shape):
    """Enumerate legal RHS descriptors without using the proposed constructor.

    Bounds 1..I_j cover every behavior: larger strides give at most one
    output and can be replaced by I_j; a dilation larger than I_j is either
    illegal (K_j > 1) or irrelevant (K_j = 1), when dilation 1 suffices.
    """
    rank = len(input_shape)
    meanings = set()
    for stride in product(*(range(1, n + 1) for n in input_shape)):
        for dilation in product(*(range(1, n + 1) for n in input_shape)):
            residual = tuple(
                input_shape[j] - 1 - dilation[j] * (kernel_shape[j] - 1)
                for j in range(rank)
            )
            if min(residual) < 0:
                continue
            shape = tuple(residual[j] // stride[j] + 1 for j in range(rank))
            meaning = signature(
                input_shape, kernel_shape, shape, (0,) * rank,
                diagonal(stride), diagonal(dilation),
            )
            assert meaning is not None
            meanings.add(meaning)
    return meanings


def compare(source, oracle):
    meaning = signature(*source)
    if meaning is None:
        return None
    actual = construct(*source)
    assert (actual is not None) == (meaning in oracle), (source, actual)
    if actual is not None:
        stride, dilation = actual
        inputs, kernels, outputs, *_ = source
        rebuilt = signature(
            inputs, kernels, outputs, (0,) * len(inputs),
            diagonal(stride), diagonal(dilation),
        )
        assert rebuilt == meaning, (source, actual)
    return actual is not None


def main():
    checked = accepted = 0
    for length, kernel in product(range(1, 8), range(1, 5)):
        inputs, kernels = (length,), (kernel,)
        oracle = enumerate_rhs(inputs, kernels)
        for size, offset, a, b in product(
            range(1, 5), range(length), range(-3, 5), range(-3, 5)
        ):
            result = compare(
                (inputs, kernels, (size,), (offset,), ((a,),), ((b,),)),
                oracle,
            )
            if result is not None:
                checked += 1
                accepted += result
    assert (checked, accepted) == (7580, 872)
    print(f"1D: {checked} legal affine sources; {accepted} feasible; no discrepancies.")

    inputs = (3, 3)
    checked = accepted = 0
    for kernels, outputs in product(product((1, 2), repeat=2), repeat=2):
        oracle = enumerate_rhs(inputs, kernels)
        for coefficients in product((0, 1), repeat=8):
            output_map = (coefficients[:2], coefficients[2:4])
            kernel_map = (coefficients[4:6], coefficients[6:])
            result = compare(
                (inputs, kernels, outputs, (0, 0), output_map, kernel_map),
                oracle,
            )
            if result is not None:
                checked += 1
                accepted += result
    assert (checked, accepted) == (3721, 441)
    print(f"2D: {checked} legal affine sources; {accepted} feasible; no discrepancies.")

    examples = (
        (3, 0, 2, ((2,), (1,))),
        (2, 0, 1, None),
        (3, 1, 2, None),
        (1, 0, 1, ((6,), (1,))),
    )
    for size, offset, coefficient, expected in examples:
        assert construct(
            (8,), (3,), (size,), (offset,), ((coefficient,),), ((1,),)
        ) == expected
    print("All four worked examples agree with the constructor.")


if __name__ == "__main__":
    main()

