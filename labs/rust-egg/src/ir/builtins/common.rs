//! Shared checked list and integer primitives for shapes and rules.
use super::{BuiltinError, BuiltinResult};

/// An always-enabled TEPL assertion. A failed condition returns an error.
pub fn ensure(condition: bool, message: &'static str) -> BuiltinResult<()> {
    if condition {
        Ok(())
    } else {
        Err(BuiltinError::Assertion(message))
    }
}

/// Fallible indexing preserves TEPL errors rather than panicking.
pub fn index<T>(values: &[T], position: impl TryInto<usize>) -> BuiltinResult<&T> {
    let position = position
        .try_into()
        .map_err(|_| BuiltinError::IndexOutOfBounds)?;
    values.get(position).ok_or(BuiltinError::IndexOutOfBounds)
}

/// Integer operations used by the checked arithmetic builtins.
/// Sealed to Rust's primitive integer types so overflow behavior is consistent.
pub trait Integer: sealed::Sealed + Copy + Ord {
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

macro_rules! integer_types {
    ($($ty:ty),+ $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl Integer for $ty {
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

integer_types!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

pub fn add<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    lhs.checked_add(rhs).ok_or(BuiltinError::Overflow)
}
pub fn sub<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    lhs.checked_sub(rhs).ok_or(BuiltinError::Overflow)
}
pub fn mul<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    lhs.checked_mul(rhs).ok_or(BuiltinError::Overflow)
}
/// / truncates toward zero; rounded division uses floor_div/ceil_div.
pub fn div<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    if rhs == T::ZERO {
        return Err(BuiltinError::DivisionByZero);
    }
    lhs.checked_div(rhs).ok_or(BuiltinError::Overflow)
}
pub fn rem<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    if rhs == T::ZERO {
        return Err(BuiltinError::DivisionByZero);
    }
    lhs.checked_rem(rhs).ok_or(BuiltinError::Overflow)
}
pub fn neg<T: Integer>(value: T) -> BuiltinResult<T> {
    sub(T::ZERO, value)
}
/// Reject nonfinite host values independently of build mode.
pub fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
/// Convert list lengths into the rule language's index domain without truncation.
pub fn index_len<T>(values: &[T]) -> BuiltinResult<u64> {
    u64::try_from(len(values)).map_err(|_| BuiltinError::Overflow)
}
pub fn len<T>(values: &[T]) -> usize {
    values.len()
}

/// Eager integer list [0, end). Rejects negative or unrepresentable bounds.
pub fn range(end: impl TryInto<usize>) -> BuiltinResult<Vec<u64>> {
    let end = end.try_into().map_err(|_| BuiltinError::InvalidRange)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(end)
        .map_err(|_| BuiltinError::CapacityExceeded)?;
    values.extend((0..end).map(|index| index as u64));
    Ok(values)
}

/// Concatenate two or more lists. Nested lists retain their element boundaries.
/// TEPL concat(a, b, c) lowers to concat(&[&a, &b, &c]).
pub fn concat<T: Clone>(lists: &[&[T]]) -> BuiltinResult<Vec<T>> {
    ensure(lists.len() >= 2, "concat requires at least two lists")?;
    let size = lists.iter().try_fold(0_usize, |size, list| {
        size.checked_add(list.len()).ok_or(BuiltinError::Overflow)
    })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| BuiltinError::CapacityExceeded)?;
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
) -> BuiltinResult<Vec<T>> {
    let start = start
        .try_into()
        .map_err(|_| BuiltinError::IndexOutOfBounds)?;
    let end = end.try_into().map_err(|_| BuiltinError::IndexOutOfBounds)?;
    values
        .get(start..end)
        .map(<[T]>::to_vec)
        .ok_or(BuiltinError::IndexOutOfBounds)
}

/// Copy a list with exactly one entry replaced.
pub fn replace<T: Clone>(
    values: &[T],
    axis: impl TryInto<usize>,
    value: T,
) -> BuiltinResult<Vec<T>> {
    let axis = axis
        .try_into()
        .map_err(|_| BuiltinError::IndexOutOfBounds)?;
    values.get(axis).ok_or(BuiltinError::IndexOutOfBounds)?;
    let mut output = values.to_vec();
    output[axis] = value;
    Ok(output)
}

/// Checked sum in list order; the empty sum is zero.
pub fn sum<T: Integer>(values: &[T]) -> BuiltinResult<T> {
    values.iter().try_fold(T::ZERO, |sum, &value| {
        sum.checked_add(value).ok_or(BuiltinError::Overflow)
    })
}

/// Checked product in list order; the empty product is one.
/// Any zero makes the product zero, even if a preceding prefix would overflow.
pub fn product<T: Integer>(values: &[T]) -> BuiltinResult<T> {
    if values.contains(&T::ZERO) {
        return Ok(T::ZERO);
    }
    values.iter().try_fold(T::ONE, |product, &value| {
        product.checked_mul(value).ok_or(BuiltinError::Overflow)
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

pub fn is_disjoint<T: PartialEq>(lhs: &[T], rhs: &[T]) -> bool {
    !lhs.iter().any(|value| rhs.contains(value))
}

/// Select entries in the supplied order. Repeated indices are allowed.
/// Bounds are checked here, independently of any earlier axis assertions.
pub fn gather<T: Clone, I: Copy + TryInto<usize>>(
    values: &[T],
    indices: &[I],
) -> BuiltinResult<Vec<T>> {
    indices
        .iter()
        .map(|&index| {
            let index = index
                .try_into()
                .map_err(|_| BuiltinError::IndexOutOfBounds)?;
            values
                .get(index)
                .cloned()
                .ok_or(BuiltinError::IndexOutOfBounds)
        })
        .collect()
}

pub fn min<T: Integer>(lhs: T, rhs: T) -> T {
    std::cmp::min(lhs, rhs)
}

pub fn max<T: Integer>(lhs: T, rhs: T) -> T {
    std::cmp::max(lhs, rhs)
}

/// Integer division rounded down. The divisor must be strictly positive.
pub fn floor_div<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    if rhs <= T::ZERO {
        return Err(BuiltinError::NonPositiveDivisor);
    }
    let quotient = lhs.checked_div(rhs).ok_or(BuiltinError::Overflow)?;
    let remainder = lhs.checked_rem(rhs).ok_or(BuiltinError::Overflow)?;
    if remainder < T::ZERO {
        quotient.checked_sub(T::ONE).ok_or(BuiltinError::Overflow)
    } else {
        Ok(quotient)
    }
}

/// Integer division rounded up. Avoids overflow from adding rhs - 1 to lhs.
pub fn ceil_div<T: Integer>(lhs: T, rhs: T) -> BuiltinResult<T> {
    if rhs <= T::ZERO {
        return Err(BuiltinError::NonPositiveDivisor);
    }
    let quotient = lhs.checked_div(rhs).ok_or(BuiltinError::Overflow)?;
    let remainder = lhs.checked_rem(rhs).ok_or(BuiltinError::Overflow)?;
    if remainder > T::ZERO {
        quotient.checked_add(T::ONE).ok_or(BuiltinError::Overflow)
    } else {
        Ok(quotient)
    }
}
