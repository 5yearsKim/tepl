// Runtime tests for copied shape builtins.
use rust_egg::ir::builtins::{BuiltinError, common::*, shape::*};

#[test]
fn dimension_boundaries_and_indices_are_checked() {
    assert_eq!(integers(&[0, u64::MAX]), vec![0, i128::from(u64::MAX)]);
    assert_eq!(
        dimensions(&[0, i128::from(u64::MAX)]),
        Ok(vec![0, u64::MAX])
    );
    assert_eq!(dimensions(&[-1]), Err(BuiltinError::InvalidDimension));
    assert_eq!(
        dimensions(&[i128::from(u64::MAX) + 1]),
        Err(BuiltinError::InvalidDimension)
    );
    assert_eq!(index(&[7], 0_i128), Ok(&7));
    assert_eq!(index(&[7], -1_i128), Err(BuiltinError::IndexOutOfBounds));
    assert_eq!(index(&[7], i128::MAX), Err(BuiltinError::IndexOutOfBounds));
    assert_eq!(index(&[7], 1_i128), Err(BuiltinError::IndexOutOfBounds));
}

#[test]
fn scalar_shape_arithmetic_checks_overflow_and_zero_divisors() {
    assert_eq!(add(i128::from(u64::MAX), -2), Ok(i128::from(u64::MAX) - 2));
    assert_eq!(add(i128::MAX, 1), Err(BuiltinError::Overflow));
    assert_eq!(sub(i128::MIN, 1), Err(BuiltinError::Overflow));
    assert_eq!(mul(i128::MAX, 2), Err(BuiltinError::Overflow));
    assert_eq!(div(-7, 3), Ok(-2));
    assert_eq!(rem(-7, 3), Ok(-1));
    assert_eq!(div(1, 0), Err(BuiltinError::DivisionByZero));
    assert_eq!(rem(1, 0), Err(BuiltinError::DivisionByZero));
    assert_eq!(div(i128::MIN, -1), Err(BuiltinError::Overflow));
    assert_eq!(rem(i128::MIN, -1), Err(BuiltinError::Overflow));
}

#[test]
fn products_preserve_scalar_and_zero_sized_shapes() {
    assert_eq!(product(&[]), Ok(1));
    assert_eq!(product(&[2, 3, 4]), Ok(24));
    assert_eq!(product(&[u64::MAX, 1]), Ok(u64::MAX));
    assert_eq!(product(&[u64::MAX, 2]), Err(BuiltinError::Overflow));
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
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(
        gather(&[2, 3], &[0, 2]),
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(
        gather(&[2], &[u64::MAX]),
        Err(BuiltinError::IndexOutOfBounds)
    );
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
    assert_eq!(error, BuiltinError::Assertion("invalid permutation"));
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
    assert_eq!(range(-1_i64), Err(BuiltinError::InvalidRange));
    assert_eq!(range(u128::MAX), Err(BuiltinError::InvalidRange));
    assert_eq!(range(usize::MAX), Err(BuiltinError::CapacityExceeded));
}

#[test]
fn concatenation_preserves_nested_lists_and_requires_two_arguments() {
    assert_eq!(concat(&[&[2], &[3, 4], &[]]), Ok(vec![2, 3, 4]));
    assert_eq!(concat::<u64>(&[&[], &[]]), Ok(vec![]));
    for lists in [vec![], vec![&[] as &[u64]]] {
        assert_eq!(
            concat(&lists),
            Err(BuiltinError::Assertion(
                "concat requires at least two lists"
            )),
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
        assert_eq!(
            slice(&input, start, end),
            Err(BuiltinError::IndexOutOfBounds)
        );
    }
    assert_eq!(
        slice(&input, 0, u128::MAX),
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(replace(&input, 1, 5), Ok(vec![2, 5, 4]));
    assert_eq!(replace(&input, 0_u64, 7), Ok(vec![7, 3, 4]));
    assert_eq!(
        replace(&input, -1_i64, 5),
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(replace(&input, 3, 5), Err(BuiltinError::IndexOutOfBounds));
    assert_eq!(
        replace(&input, u128::MAX, 5),
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(
        replace::<u64>(&[], 0, 5),
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(input, [2, 3, 4]);
}

#[test]
fn signed_and_unrepresentable_axes_are_checked_without_panicking() {
    assert_eq!(
        gather(&[2, 3], &[-1_i64]),
        Err(BuiltinError::IndexOutOfBounds)
    );
    assert_eq!(
        gather(&[2, 3], &[u128::MAX]),
        Err(BuiltinError::IndexOutOfBounds)
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
    assert_eq!(sum(&[u64::MAX, 1]), Err(BuiltinError::Overflow));
    assert_eq!(sum(&[i64::MIN, -1]), Err(BuiltinError::Overflow));
    assert_eq!(sum(&[i64::MAX, 1]), Err(BuiltinError::Overflow));
    assert_eq!(product::<i64>(&[]), Ok(1));
    assert_eq!(product(&[-2_i64, 3, -4]), Ok(24));
    assert_eq!(product(&[i64::MIN, -1]), Err(BuiltinError::Overflow));
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
    let wide = i128::from(u64::MAX) + 1;
    assert_eq!(broadcast_shape(&[wide], &[1]), Ok(vec![wide]));
    assert_eq!(
        broadcast_shape(&[-1_i128], &[1]),
        Err(BuiltinError::InvalidDimension)
    );
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
            Err(BuiltinError::IncompatibleBroadcast)
        );
        assert_eq!(
            broadcast_shape(&rhs, &lhs),
            Err(BuiltinError::IncompatibleBroadcast)
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
            Err(BuiltinError::NonPositiveDivisor)
        );
        assert_eq!(
            ceil_div(i64::MIN, divisor),
            Err(BuiltinError::NonPositiveDivisor)
        );
    }
}
