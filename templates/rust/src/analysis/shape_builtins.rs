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
pub fn broadcast_shape<T: ShapeInteger>(lhs: &[T], rhs: &[T]) -> ShapeResult<Vec<T>> {
    if lhs.iter().chain(rhs).any(|&dimension| dimension < T::ZERO) {
        return Err(ShapeError::InvalidDimension);
    }
    let rank = std::cmp::max(lhs.len(), rhs.len());
    let mut output = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left = lhs.iter().rev().nth(offset).copied().unwrap_or(T::ONE);
        let right = rhs.iter().rev().nth(offset).copied().unwrap_or(T::ONE);
        output.push(if left == right || right == T::ONE {
            left
        } else if left == T::ONE {
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
