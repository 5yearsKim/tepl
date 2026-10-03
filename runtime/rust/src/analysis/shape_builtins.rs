//! Pure, checked primitives used by operation shape evaluators.
//!
//! These helpers do not depend on dialects, host semantics, or e-graphs.
//! Implements every builtin listed in examples/shape_guide.md, plus `ensure`.
//! Shapes use u64 dimensions; arithmetic also supports signed integer types.
//! Index helpers check conversion to Rust's usize. `len` returns usize and
//! `range` returns u64 axes. The generator chooses an integer type and performs
//! explicit checked conversions when mixing signed values and dimensions.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeError {
    Assertion(&'static str),
    Overflow,
    IndexOutOfBounds,
    InvalidRange,
    NonPositiveDivisor,
    IncompatibleBroadcast,
    CapacityExceeded,
    DivisionByZero,
    InvalidDimension,
}

impl ShapeError {
    pub fn message(self) -> &'static str {
        match self {
            Self::Assertion(message) => message,
            Self::Overflow => "shape arithmetic overflow",
            Self::IndexOutOfBounds => "shape index out of bounds",
            Self::InvalidRange => "range bound must be nonnegative and fit usize",
            Self::NonPositiveDivisor => "shape division requires a positive divisor",
            Self::IncompatibleBroadcast => "incompatible broadcast dimensions",
            Self::CapacityExceeded => "shape list capacity exceeded",
            Self::DivisionByZero => "shape division by zero",
            Self::InvalidDimension => "output dimension must be nonnegative and fit u64",
        }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for ShapeError {}

pub type ShapeResult<T> = Result<T, ShapeError>;

/// An always-enabled TEPL assertion. A failed condition returns an error.
pub fn ensure(condition: bool, message: &'static str) -> ShapeResult<()> {
    if condition {
        Ok(())
    } else {
        Err(ShapeError::Assertion(message))
    }
}

/// Widen external dimensions/indices to the shape language's integer domain.
pub fn integers(values: &[u64]) -> Vec<i128> {
    values.iter().copied().map(i128::from).collect()
}

/// Validate a yielded shape before publishing it as tensor metadata.
pub fn dimensions(values: &[i128]) -> ShapeResult<Vec<u64>> {
    values
        .iter()
        .map(|&value| u64::try_from(value).map_err(|_| ShapeError::InvalidDimension))
        .collect()
}

/// Fallible indexing preserves TEPL errors rather than panicking.
pub fn index<T>(values: &[T], position: impl TryInto<usize>) -> ShapeResult<&T> {
    let position = position
        .try_into()
        .map_err(|_| ShapeError::IndexOutOfBounds)?;
    values.get(position).ok_or(ShapeError::IndexOutOfBounds)
}

pub fn add(lhs: i128, rhs: i128) -> ShapeResult<i128> {
    lhs.checked_add(rhs).ok_or(ShapeError::Overflow)
}

pub fn sub(lhs: i128, rhs: i128) -> ShapeResult<i128> {
    lhs.checked_sub(rhs).ok_or(ShapeError::Overflow)
}

pub fn mul(lhs: i128, rhs: i128) -> ShapeResult<i128> {
    lhs.checked_mul(rhs).ok_or(ShapeError::Overflow)
}

/// TEPL's / truncates toward zero; rounded division uses floor_div/ceil_div.
pub fn div(lhs: i128, rhs: i128) -> ShapeResult<i128> {
    if rhs == 0 {
        return Err(ShapeError::DivisionByZero);
    }
    lhs.checked_div(rhs).ok_or(ShapeError::Overflow)
}

pub fn rem(lhs: i128, rhs: i128) -> ShapeResult<i128> {
    if rhs == 0 {
        return Err(ShapeError::DivisionByZero);
    }
    lhs.checked_rem(rhs).ok_or(ShapeError::Overflow)
}

/// Integer operations used by the checked arithmetic builtins.
/// Sealed to Rust's primitive integer types so overflow behavior is consistent.
pub trait ShapeInteger: sealed::Sealed + Copy + Ord {
    const ZERO: Self;
    const ONE: Self;
    fn checked_add(self, other: Self) -> Option<Self>;
    fn checked_sub(self, other: Self) -> Option<Self>;
    fn checked_mul(self, other: Self) -> Option<Self>;
    fn checked_div(self, other: Self) -> Option<Self>;
    fn checked_rem(self, other: Self) -> Option<Self>;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! shape_integers {
    ($($ty:ty),+ $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl ShapeInteger for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            fn checked_add(self, other: Self) -> Option<Self> { <$ty>::checked_add(self, other) }
            fn checked_sub(self, other: Self) -> Option<Self> { <$ty>::checked_sub(self, other) }
            fn checked_mul(self, other: Self) -> Option<Self> { <$ty>::checked_mul(self, other) }
            fn checked_div(self, other: Self) -> Option<Self> { <$ty>::checked_div(self, other) }
            fn checked_rem(self, other: Self) -> Option<Self> { <$ty>::checked_rem(self, other) }
        }
    )+};
}

shape_integers!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

pub fn len<T>(values: &[T]) -> usize {
    values.len()
}

/// Eager integer list [0, end). Rejects negative or unrepresentable bounds.
pub fn range(end: impl TryInto<usize>) -> ShapeResult<Vec<u64>> {
    let end = end.try_into().map_err(|_| ShapeError::InvalidRange)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(end)
        .map_err(|_| ShapeError::CapacityExceeded)?;
    values.extend((0..end).map(|index| index as u64));
    Ok(values)
}

/// Concatenate two or more lists. Nested lists retain their element boundaries.
/// TEPL concat(a, b, c) lowers to concat(&[&a, &b, &c]).
pub fn concat<T: Clone>(lists: &[&[T]]) -> ShapeResult<Vec<T>> {
    ensure(lists.len() >= 2, "concat requires at least two lists")?;
    let size = lists.iter().try_fold(0_usize, |size, list| {
        size.checked_add(list.len()).ok_or(ShapeError::Overflow)
    })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| ShapeError::CapacityExceeded)?;
    for list in lists {
        output.extend_from_slice(list);
    }
    Ok(output)
}

/// Remove matching values (not positions), preserving order and duplicates
/// among the remaining entries.
pub fn exclude<T: Clone + PartialEq>(values: &[T], removed: &[T]) -> Vec<T> {
    values
        .iter()
        .filter(|value| !removed.contains(value))
        .cloned()
        .collect()
}

/// Copy [start, end); invalid bounds are errors and are never clamped.
pub fn slice<T: Clone>(
    values: &[T],
    start: impl TryInto<usize>,
    end: impl TryInto<usize>,
) -> ShapeResult<Vec<T>> {
    let start = start.try_into().map_err(|_| ShapeError::IndexOutOfBounds)?;
    let end = end.try_into().map_err(|_| ShapeError::IndexOutOfBounds)?;
    values
        .get(start..end)
        .map(<[T]>::to_vec)
        .ok_or(ShapeError::IndexOutOfBounds)
}

/// Copy a list with exactly one entry replaced.
pub fn replace<T: Clone>(values: &[T], axis: impl TryInto<usize>, value: T) -> ShapeResult<Vec<T>> {
    let axis = axis.try_into().map_err(|_| ShapeError::IndexOutOfBounds)?;
    values.get(axis).ok_or(ShapeError::IndexOutOfBounds)?;
    let mut output = values.to_vec();
    output[axis] = value;
    Ok(output)
}

/// Checked sum in list order; the empty sum is zero.
pub fn sum<T: ShapeInteger>(values: &[T]) -> ShapeResult<T> {
    values.iter().try_fold(T::ZERO, |sum, &value| {
        sum.checked_add(value).ok_or(ShapeError::Overflow)
    })
}

/// Checked product in list order; the empty product is one.
/// Any zero makes the product zero, even if a preceding prefix would overflow.
pub fn product<T: ShapeInteger>(values: &[T]) -> ShapeResult<T> {
    if values.contains(&T::ZERO) {
        return Ok(T::ZERO);
    }
    values.iter().try_fold(T::ONE, |product, &value| {
        product.checked_mul(value).ok_or(ShapeError::Overflow)
    })
}

/// Consumes an already evaluated Boolean list; all([]) is true.
pub fn all(values: &[bool]) -> bool {
    values.iter().all(|&value| value)
}

/// Consumes an already evaluated Boolean list; any([]) is false.
pub fn any(values: &[bool]) -> bool {
    values.iter().any(|&value| value)
}

pub fn contains<T: PartialEq>(values: &[T], value: &T) -> bool {
    values.contains(value)
}

/// Whether axes are unique and all lie in [0, rank). Negative axes are invalid.
pub fn is_valid_axis_list<I: Copy + PartialEq + TryInto<usize>>(axes: &[I], rank: usize) -> bool {
    axes.iter().enumerate().all(|(i, &axis)| {
        axis.try_into().is_ok_and(|axis| axis < rank) && !axes[..i].contains(&axis)
    })
}

pub fn is_disjoint<T: PartialEq>(lhs: &[T], rhs: &[T]) -> bool {
    !lhs.iter().any(|value| rhs.contains(value))
}

/// Select entries in the supplied order. Repeated indices are allowed.
/// Bounds are checked here, independently of any earlier axis assertions.
pub fn gather<T: Clone, I: Copy + TryInto<usize>>(
    values: &[T],
    indices: &[I],
) -> ShapeResult<Vec<T>> {
    indices
        .iter()
        .map(|&index| {
            let index = index.try_into().map_err(|_| ShapeError::IndexOutOfBounds)?;
            values
                .get(index)
                .cloned()
                .ok_or(ShapeError::IndexOutOfBounds)
        })
        .collect()
}

/// Right-aligned broadcasting with missing leading dimensions treated as one.
/// A zero broadcasts with one to zero, but is incompatible with other sizes.
pub fn broadcast_shape(lhs: &[u64], rhs: &[u64]) -> ShapeResult<Vec<u64>> {
    let rank = std::cmp::max(lhs.len(), rhs.len());
    let mut output = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left = lhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let right = rhs.iter().rev().nth(offset).copied().unwrap_or(1);
        output.push(if left == right || right == 1 {
            left
        } else if left == 1 {
            right
        } else {
            return Err(ShapeError::IncompatibleBroadcast);
        });
    }
    output.reverse();
    Ok(output)
}

pub fn min<T: ShapeInteger>(lhs: T, rhs: T) -> T {
    std::cmp::min(lhs, rhs)
}

pub fn max<T: ShapeInteger>(lhs: T, rhs: T) -> T {
    std::cmp::max(lhs, rhs)
}

/// Integer division rounded down. The divisor must be strictly positive.
pub fn floor_div<T: ShapeInteger>(lhs: T, rhs: T) -> ShapeResult<T> {
    if rhs <= T::ZERO {
        return Err(ShapeError::NonPositiveDivisor);
    }
    let quotient = lhs.checked_div(rhs).ok_or(ShapeError::Overflow)?;
    let remainder = lhs.checked_rem(rhs).ok_or(ShapeError::Overflow)?;
    if remainder < T::ZERO {
        quotient.checked_sub(T::ONE).ok_or(ShapeError::Overflow)
    } else {
        Ok(quotient)
    }
}

/// Integer division rounded up. Avoids overflow from adding rhs - 1 to lhs.
pub fn ceil_div<T: ShapeInteger>(lhs: T, rhs: T) -> ShapeResult<T> {
    if rhs <= T::ZERO {
        return Err(ShapeError::NonPositiveDivisor);
    }
    let quotient = lhs.checked_div(rhs).ok_or(ShapeError::Overflow)?;
    let remainder = lhs.checked_rem(rhs).ok_or(ShapeError::Overflow)?;
    if remainder > T::ZERO {
        quotient.checked_add(T::ONE).ok_or(ShapeError::Overflow)
    } else {
        Ok(quotient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_boundaries_and_indices_are_checked() {
        assert_eq!(integers(&[0, u64::MAX]), vec![0, i128::from(u64::MAX)]);
        assert_eq!(
            dimensions(&[0, i128::from(u64::MAX)]),
            Ok(vec![0, u64::MAX])
        );
        assert_eq!(dimensions(&[-1]), Err(ShapeError::InvalidDimension));
        assert_eq!(
            dimensions(&[i128::from(u64::MAX) + 1]),
            Err(ShapeError::InvalidDimension)
        );
        assert_eq!(index(&[7], 0_i128), Ok(&7));
        assert_eq!(index(&[7], -1_i128), Err(ShapeError::IndexOutOfBounds));
        assert_eq!(index(&[7], i128::MAX), Err(ShapeError::IndexOutOfBounds));
        assert_eq!(index(&[7], 1_i128), Err(ShapeError::IndexOutOfBounds));
    }

    #[test]
    fn scalar_shape_arithmetic_checks_overflow_and_zero_divisors() {
        assert_eq!(add(i128::from(u64::MAX), -2), Ok(i128::from(u64::MAX) - 2));
        assert_eq!(add(i128::MAX, 1), Err(ShapeError::Overflow));
        assert_eq!(sub(i128::MIN, 1), Err(ShapeError::Overflow));
        assert_eq!(mul(i128::MAX, 2), Err(ShapeError::Overflow));
        assert_eq!(div(-7, 3), Ok(-2));
        assert_eq!(rem(-7, 3), Ok(-1));
        assert_eq!(div(1, 0), Err(ShapeError::DivisionByZero));
        assert_eq!(rem(1, 0), Err(ShapeError::DivisionByZero));
        assert_eq!(div(i128::MIN, -1), Err(ShapeError::Overflow));
        assert_eq!(rem(i128::MIN, -1), Err(ShapeError::Overflow));
    }

    #[test]
    fn products_preserve_scalar_and_zero_sized_shapes() {
        assert_eq!(product(&[]), Ok(1));
        assert_eq!(product(&[2, 3, 4]), Ok(24));
        assert_eq!(product(&[u64::MAX, 1]), Ok(u64::MAX));
        assert_eq!(product(&[u64::MAX, 2]), Err(ShapeError::Overflow));
        for values in [[0, u64::MAX, 2], [u64::MAX, 0, 2], [u64::MAX, 2, 0]] {
            assert_eq!(product(&values), Ok(0));
        }
    }

    #[test]
    fn axis_lists_check_uniqueness_and_bounds() {
        assert!(is_valid_axis_list(&[] as &[u64], 0));
        assert!(is_valid_axis_list(&[2, 0], 3));
        assert!(!is_valid_axis_list(&[1, 1], 3));
        assert!(!is_valid_axis_list(&[0], 0));
        assert!(!is_valid_axis_list(&[3], 3));
        assert!(!is_valid_axis_list(&[u64::MAX], 3));
    }

    #[test]
    fn gather_preserves_order_and_duplicates_and_checks_each_index() {
        assert_eq!(gather(&[2, 3, 4], &[2, 0, 2]), Ok(vec![4, 2, 4]));
        assert_eq!(gather::<u64, u64>(&[], &[]), Ok(vec![]));
        assert_eq!(
            gather::<u64, u64>(&[], &[0]),
            Err(ShapeError::IndexOutOfBounds)
        );
        assert_eq!(gather(&[2, 3], &[0, 2]), Err(ShapeError::IndexOutOfBounds));
        assert_eq!(gather(&[2], &[u64::MAX]), Err(ShapeError::IndexOutOfBounds));
        assert_eq!(gather(&[vec![2, 3], vec![4]], &[1]), Ok(vec![vec![4]]));
    }

    #[test]
    fn disjointness_compares_values() {
        assert!(is_disjoint::<u64>(&[], &[]));
        assert!(is_disjoint(&[0, 0], &[1, 2]));
        assert!(!is_disjoint(&[0, 2], &[1, 2]));
    }

    #[test]
    fn assertions_return_errors_in_all_build_modes() {
        assert_eq!(ensure(true, "valid"), Ok(()));
        let error = ensure(false, "invalid permutation").unwrap_err();
        assert_eq!(error, ShapeError::Assertion("invalid permutation"));
        assert_eq!(error.message(), "invalid permutation");
        assert_eq!(error.to_string(), "invalid permutation");
    }

    #[test]
    fn lengths_and_ranges_cover_empty_negative_and_unrepresentable_bounds() {
        assert_eq!(len::<u64>(&[]), 0);
        assert_eq!(len(&[vec![2, 3], vec![4]]), 2);
        assert_eq!(range(0), Ok(vec![]));
        assert_eq!(range(3_u64), Ok(vec![0, 1, 2]));
        assert_eq!(range(len(&[2, 3, 4])), Ok(vec![0, 1, 2]));
        assert_eq!(range(-1_i64), Err(ShapeError::InvalidRange));
        assert_eq!(range(u128::MAX), Err(ShapeError::InvalidRange));
        assert_eq!(range(usize::MAX), Err(ShapeError::CapacityExceeded));
    }

    #[test]
    fn concatenation_preserves_nested_lists_and_requires_two_arguments() {
        assert_eq!(concat(&[&[2], &[3, 4], &[]]), Ok(vec![2, 3, 4]));
        assert_eq!(concat::<u64>(&[&[], &[]]), Ok(vec![]));
        for lists in [vec![], vec![&[] as &[u64]]] {
            assert_eq!(
                concat(&lists),
                Err(ShapeError::Assertion("concat requires at least two lists")),
            );
        }
        let first = vec![vec![2, 3]];
        let rest = vec![vec![4], vec![]];
        assert_eq!(
            concat(&[&first, &rest]),
            Ok(vec![vec![2, 3], vec![4], vec![]])
        );
        assert_eq!(first, vec![vec![2, 3]]);
    }

    #[test]
    fn membership_and_exclusion_compare_values_and_preserve_order() {
        assert!(contains(&[1, 3], &3));
        assert!(!contains(&[1, 3], &2));
        assert!(!contains::<i64>(&[], &0));
        assert_eq!(exclude(&[7, 1, 7, 3, 1], &[1, 1, 99]), vec![7, 7, 3]);
        assert_eq!(exclude(&[7, 1, 7], &[]), vec![7, 1, 7]);
        assert_eq!(exclude::<u64>(&[], &[0]), vec![]);
        assert_eq!(exclude(&[1, 1], &[1]), Vec::<i32>::new());
        assert!(contains(&[vec![2, 3], vec![4]], &vec![2, 3]));
    }

    #[test]
    fn slicing_and_replacement_check_bounds_without_changing_inputs() {
        let input = [2, 3, 4];
        assert_eq!(slice(&input, 0, 2), Ok(vec![2, 3]));
        assert_eq!(slice(&input, 0, 3), Ok(input.to_vec()));
        assert_eq!(slice(&input, 3, 3), Ok(vec![]));
        assert_eq!(slice::<u64>(&[], 0, 0), Ok(vec![]));
        for (start, end) in [(-1_i64, 2), (0, -1), (2, 1), (0, 4), (4, 4)] {
            assert_eq!(slice(&input, start, end), Err(ShapeError::IndexOutOfBounds));
        }
        assert_eq!(
            slice(&input, 0, u128::MAX),
            Err(ShapeError::IndexOutOfBounds)
        );
        assert_eq!(replace(&input, 1, 5), Ok(vec![2, 5, 4]));
        assert_eq!(replace(&input, 0_u64, 7), Ok(vec![7, 3, 4]));
        assert_eq!(
            replace(&input, -1_i64, 5),
            Err(ShapeError::IndexOutOfBounds)
        );
        assert_eq!(replace(&input, 3, 5), Err(ShapeError::IndexOutOfBounds));
        assert_eq!(
            replace(&input, u128::MAX, 5),
            Err(ShapeError::IndexOutOfBounds)
        );
        assert_eq!(replace::<u64>(&[], 0, 5), Err(ShapeError::IndexOutOfBounds));
        assert_eq!(input, [2, 3, 4]);
    }

    #[test]
    fn signed_and_unrepresentable_axes_are_checked_without_panicking() {
        assert_eq!(
            gather(&[2, 3], &[-1_i64]),
            Err(ShapeError::IndexOutOfBounds)
        );
        assert_eq!(
            gather(&[2, 3], &[u128::MAX]),
            Err(ShapeError::IndexOutOfBounds)
        );
        assert!(!is_valid_axis_list(&[-1_i64, 0], 2));
        assert!(!is_valid_axis_list(&[u128::MAX], 2));
        assert!(is_valid_axis_list(&[1_i64, 0], 2));
    }

    #[test]
    fn arithmetic_reductions_accept_signed_values_and_check_overflow() {
        assert_eq!(sum::<u64>(&[]), Ok(0));
        assert_eq!(sum(&[3_u64, 5]), Ok(8));
        assert_eq!(sum(&[-3_i64, 5]), Ok(2));
        assert_eq!(sum(&[u64::MAX, 1]), Err(ShapeError::Overflow));
        assert_eq!(sum(&[i64::MIN, -1]), Err(ShapeError::Overflow));
        assert_eq!(sum(&[i64::MAX, 1]), Err(ShapeError::Overflow));
        assert_eq!(product::<i64>(&[]), Ok(1));
        assert_eq!(product(&[-2_i64, 3, -4]), Ok(24));
        assert_eq!(product(&[i64::MIN, -1]), Err(ShapeError::Overflow));
        assert_eq!(product(&[i64::MIN, -1, 0]), Ok(0));
        // A wider signed representation can combine u64 dimensions and padding.
        assert_eq!(
            sum(&[i128::from(u64::MAX), -2]),
            Ok(i128::from(u64::MAX) - 2)
        );
    }

    #[test]
    fn boolean_reductions_have_the_specified_empty_identities() {
        assert!(all(&[]));
        assert!(!any(&[]));
        assert!(all(&[true, true]));
        assert!(!all(&[true, false]));
        assert!(any(&[false, true]));
        assert!(!any(&[false, false]));
    }

    #[test]
    fn broadcasting_aligns_right_and_handles_zero_dimensions() {
        for (lhs, rhs, expected) in [
            (vec![], vec![], vec![]),
            (vec![], vec![2, 3], vec![2, 3]),
            (vec![2, 1, 4], vec![3, 4], vec![2, 3, 4]),
            (vec![2, 0, 4], vec![1, 4], vec![2, 0, 4]),
            (vec![0], vec![0], vec![0]),
            (vec![u64::MAX], vec![1], vec![u64::MAX]),
        ] {
            assert_eq!(broadcast_shape(&lhs, &rhs), Ok(expected.clone()));
            assert_eq!(broadcast_shape(&rhs, &lhs), Ok(expected));
        }
        for (lhs, rhs) in [
            (vec![0], vec![2]),
            (vec![2, 3], vec![2]),
            (vec![2], vec![3]),
        ] {
            assert_eq!(
                broadcast_shape(&lhs, &rhs),
                Err(ShapeError::IncompatibleBroadcast)
            );
            assert_eq!(
                broadcast_shape(&rhs, &lhs),
                Err(ShapeError::IncompatibleBroadcast)
            );
        }
    }

    #[test]
    fn scalar_extrema_support_signed_values_and_full_width_dimensions() {
        assert_eq!(min(-2_i64, 0), -2);
        assert_eq!(max(0_i64, -2), 0);
        assert_eq!(min(i64::MIN, i64::MAX), i64::MIN);
        assert_eq!(max(0_u64, u64::MAX), u64::MAX);
        assert_eq!(min(3, 3), 3);
    }

    #[test]
    fn rounded_division_obeys_mathematical_bounds_for_both_signs() {
        for numerator in -31_i64..=31 {
            for divisor in 1..=9 {
                let floor = floor_div(numerator, divisor).unwrap();
                let ceil = ceil_div(numerator, divisor).unwrap();
                assert!(floor * divisor <= numerator && numerator < (floor + 1) * divisor);
                assert!((ceil - 1) * divisor < numerator && numerator <= ceil * divisor);
            }
        }
        assert_eq!(floor_div(-7_i64, 3), Ok(-3));
        assert_eq!(ceil_div(-7_i64, 3), Ok(-2));
        assert_eq!(floor_div(i64::MIN, 1), Ok(i64::MIN));
        assert_eq!(ceil_div(i64::MAX, 1), Ok(i64::MAX));
        assert_eq!(floor_div(i64::MIN, 3), Ok(-3074457345618258603));
        assert_eq!(ceil_div(i64::MIN, 3), Ok(-3074457345618258602));
        assert_eq!(ceil_div(u64::MAX, 2), Ok(1_u64 << 63));
        assert_eq!(floor_div(u64::MAX, 2), Ok(u64::MAX / 2));
        for divisor in [0_i64, -1, i64::MIN] {
            assert_eq!(
                floor_div(i64::MIN, divisor),
                Err(ShapeError::NonPositiveDivisor)
            );
            assert_eq!(
                ceil_div(i64::MIN, divisor),
                Err(ShapeError::NonPositiveDivisor)
            );
        }
    }
}
