//! Shape-specific semantics, also available to rule expressions.
use super::common::Integer;
use super::{BuiltinError, BuiltinResult};

/// Widen external dimensions/indices to the shape language's integer domain.
pub fn integers(values: &[u64]) -> Vec<i128> {
    values.iter().copied().map(i128::from).collect()
}

/// Validate a yielded shape before publishing it as tensor metadata.
pub fn dimensions(values: &[i128]) -> BuiltinResult<Vec<u64>> {
    values
        .iter()
        .map(|&value| u64::try_from(value).map_err(|_| BuiltinError::InvalidDimension))
        .collect()
}

/// Whether axes are unique and all lie in [0, rank). Negative axes are invalid.
pub fn is_valid_axis_list<I: Copy + PartialEq + TryInto<usize>>(
    axes: &[I],
    rank: impl TryInto<usize>,
) -> bool {
    let Ok(rank) = rank.try_into() else {
        return false;
    };
    axes.iter().enumerate().all(|(i, &axis)| {
        axis.try_into().is_ok_and(|axis| axis < rank) && !axes[..i].contains(&axis)
    })
}

/// Right-aligned broadcasting with missing leading dimensions treated as one.
/// A zero broadcasts with one to zero, but is incompatible with other sizes.
pub fn broadcast_shape<T: Integer>(lhs: &[T], rhs: &[T]) -> BuiltinResult<Vec<T>> {
    if lhs.iter().chain(rhs).any(|&dimension| dimension < T::ZERO) {
        return Err(BuiltinError::InvalidDimension);
    }
    let rank = ::std::cmp::max(lhs.len(), rhs.len());
    let mut output = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left = lhs.iter().rev().nth(offset).copied().unwrap_or(T::ONE);
        let right = rhs.iter().rev().nth(offset).copied().unwrap_or(T::ONE);
        output.push(if left == right || right == T::ONE {
            left
        } else if left == T::ONE {
            right
        } else {
            return Err(BuiltinError::IncompatibleBroadcast);
        });
    }
    output.reverse();
    Ok(output)
}
