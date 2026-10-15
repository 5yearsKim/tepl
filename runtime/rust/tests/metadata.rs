use tepl_runtime::values::{AttributeValue, Axis};
use tepl_runtime::{Budget, EvalError, Fact, Value};
use tepl_runtime::{builtins, intrinsics};
fn ints(xs: &[i128]) -> Value {
    Value::List(xs.iter().copied().map(Value::Integer).collect())
}
fn b(name: &str, args: Vec<Value>) -> Value {
    builtins::call(name, args, &mut Budget::default()).unwrap()
}
fn i(name: &str, args: Vec<Value>) -> Value {
    intrinsics::call(name, args, &mut Budget::default()).unwrap()
}
#[test]
fn checked_integer_and_list_contracts() {
    assert_eq!(
        b("floor_div", vec![Value::Integer(-7), Value::Integer(3)]),
        Value::Integer(-3)
    );
    assert_eq!(
        b("ceil_div", vec![Value::Integer(-7), Value::Integer(3)]),
        Value::Integer(-2)
    );
    assert_eq!(
        b(
            "product",
            vec![ints(&[u64::MAX as i128, u64::MAX as i128, 0])]
        ),
        Value::Integer(0)
    );
    assert!(builtins::arithmetic("+", i128::MAX, 1).is_err());
    assert!(builtins::arithmetic("/", i128::MIN, -1).is_err());
    assert!(
        builtins::call(
            "floor_div",
            vec![Value::Integer(1), Value::Integer(0)],
            &mut Budget::default()
        )
        .is_err()
    );
    assert!(u64::from_value(&Value::Integer(-1)).is_err());
    assert!(i64::from_value(&Value::Integer(u64::MAX as i128)).is_err());
    assert_eq!(
        b("broadcast_shape", vec![ints(&[2, 1, 4]), ints(&[3, 4])]),
        ints(&[2, 3, 4])
    );
    assert_eq!(
        b("broadcast_shape", vec![ints(&[0]), ints(&[1])]),
        ints(&[0])
    );
    assert!(
        builtins::call(
            "gather",
            vec![ints(&[1]), ints(&[9])],
            &mut Budget::default()
        )
        .is_err()
    );
    let mut budget = Budget {
        steps: 10,
        max_list: 3,
    };
    assert!(matches!(
        builtins::call("range", vec![Value::Integer(4)], &mut budget),
        Err(EvalError::Limit(_))
    ));
}
#[test]
fn identity_and_order_are_independent_of_extents() {
    let a = Value::Axis(Axis {
        owner: 0,
        group: 0,
        member: 0,
        extent: 2,
    });
    let b = Value::Axis(Axis {
        owner: 0,
        group: 1,
        member: 0,
        extent: 2,
    });
    let c = Value::Axis(Axis {
        owner: 1,
        group: 0,
        member: 0,
        extent: 2,
    });
    assert_ne!(a, b);
    assert_ne!(a, c);
    let axes = Value::List(vec![a.clone(), b.clone(), c.clone()]);
    assert_eq!(
        i(
            "positions",
            vec![axes.clone(), Value::List(vec![c.clone(), a.clone()])]
        ),
        ints(&[2, 0])
    );
    assert_eq!(i("axis_extents", vec![axes.clone()]), ints(&[2, 2, 2]));
    assert_eq!(i("distinct_axes", vec![axes]), Value::Boolean(true));
    assert_eq!(
        i(
            "distinct_axes",
            vec![Value::List(vec![a.clone(), a.clone()])]
        ),
        Value::Boolean(false)
    );
    assert_eq!(
        i(
            "place_axes",
            vec![
                Value::List(vec![a.clone()]),
                ints(&[1]),
                Value::List(vec![b.clone(), c.clone()])
            ]
        ),
        Value::List(vec![b, a, c])
    );
    assert!(
        intrinsics::call(
            "positions",
            vec![ints(&[0, 0]), ints(&[0])],
            &mut Budget::default()
        )
        .is_err()
    );
    assert_eq!(i("inverse", vec![ints(&[2, 0, 1])]), ints(&[1, 2, 0]));
    assert_eq!(
        i("extent_floor_div", vec![ints(&[7]), ints(&[2])]),
        ints(&[3])
    );
}
#[test]
fn fact_merge_reports_changes_and_conflicts() {
    let mut shape = Fact::Unknown;
    let change = shape.merge(Fact::Known(vec![2, 3]));
    assert!(change.0);
    assert!(!change.1);
    assert_eq!(shape, Fact::Known(vec![2, 3]));
    let change = shape.merge(Fact::Unknown);
    assert!(!change.0);
    assert!(change.1);
    shape.merge(Fact::Known(vec![3, 2]));
    assert_eq!(shape, Fact::Conflict);
    assert!(matches!(shape.get("shape"), Err(EvalError::Conflict(_))));
}
