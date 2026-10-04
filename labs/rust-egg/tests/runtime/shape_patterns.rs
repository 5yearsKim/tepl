// Shared integration tests for standalone shape declaration matching.
use ShapePart::{Dimension as D, Sequence as S, Wildcard as W};
use rust_egg::ir::pattern::{ShapeBindings, ShapePart};
use std::collections::HashMap;

#[derive(Clone, Default)]
struct ReferenceBindings {
    dimensions: HashMap<usize, u64>,
    sequences: HashMap<usize, Vec<u64>>,
}

// Independent reference: consume from the front and enumerate sequence splits.
fn reference(
    pattern: &[ShapePart],
    shape: &[u64],
    mut bindings: ReferenceBindings,
) -> Option<ReferenceBindings> {
    let Some((part, rest)) = pattern.split_first() else {
        return shape.is_empty().then_some(bindings);
    };
    match part {
        D(id) => {
            let (&value, tail) = shape.split_first()?;
            if bindings.dimensions.get(id).is_some_and(|old| *old != value) {
                return None;
            }
            bindings.dimensions.insert(*id, value);
            reference(rest, tail, bindings)
        }
        W => reference(rest, shape.split_first()?.1, bindings),
        S(id) => {
            for length in 0..=shape.len() {
                let mut branch = bindings.clone();
                if let Some(id) = id {
                    if branch
                        .sequences
                        .get(id)
                        .is_some_and(|old| old != &shape[..length])
                    {
                        continue;
                    }
                    branch.sequences.insert(*id, shape[..length].to_vec());
                }
                if let Some(result) = reference(rest, &shape[length..], branch) {
                    return Some(result);
                }
            }
            None
        }
    }
}

fn lists<T: Copy>(alphabet: &[T], maximum: usize) -> Vec<Vec<T>> {
    let mut all = vec![vec![]];
    let mut layer = vec![vec![]];
    for _ in 0..maximum {
        let mut next = vec![];
        for prefix in layer {
            for &item in alphabet {
                let mut list = prefix.clone();
                list.push(item);
                next.push(list);
            }
        }
        all.extend(next.clone());
        layer = next;
    }
    all
}

#[test]
fn mixed_shapes_agree_with_recursive_reference() {
    let inputs = lists(&[0_u64, 1, 2], 5);
    for pattern in lists(&[D(0), D(1), W, S(Some(2)), S(None)], 4) {
        if pattern.iter().filter(|part| matches!(part, S(_))).count() > 1 {
            continue;
        }
        for input in &inputs {
            let expected = reference(&pattern, input, ReferenceBindings::default());
            let mut actual = ShapeBindings::default();
            assert_eq!(
                actual.check(input, &pattern).is_some(),
                expected.is_some(),
                "{pattern:?} {input:?}"
            );
            if let Some(expected) = expected {
                for id in [0, 1] {
                    assert_eq!(actual.dimension(id), expected.dimensions.get(&id).copied());
                }
                assert_eq!(
                    actual.sequence(2),
                    expected.sequences.get(&2).map(Vec::as_slice)
                );
            }
        }
    }
}

#[test]
fn shared_dimensions_and_sequences_are_checked_across_calls() {
    for input in lists(&[0_u64, 1, 2], 4) {
        let first = [D(0), S(Some(2)), W, D(0)];
        if let Some(expected) = reference(&first, &input, ReferenceBindings::default()) {
            for other in lists(&[0_u64, 1, 2], 4) {
                let mut actual = ShapeBindings::default();
                actual.check(&input, &first).unwrap();
                let second = [S(Some(2)), D(0)];
                assert_eq!(
                    actual.check(&other, &second).is_some(),
                    reference(&second, &other, expected.clone()).is_some()
                );
            }
        }
    }
}

#[test]
fn failed_checks_require_discarding_partial_bindings() {
    let mut bindings = ShapeBindings::default();
    assert!(bindings.check(&[2, 3], &[D(0), D(0)]).is_none());
    assert_eq!(bindings.dimension(0), Some(2));
    let mut fresh = ShapeBindings::default();
    assert!(fresh.check(&[4, 4], &[D(0), D(0)]).is_some());
}
