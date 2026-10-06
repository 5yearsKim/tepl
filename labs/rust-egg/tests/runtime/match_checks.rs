// Public runtime checks run against the lab's verified generated module.
use egg::{EGraph, Id, Var};
use rust_egg::ir::TensorInfo;
use rust_egg::ir::rewriting::{
    AttrPattern, AttrVar, MatchBinding as Binding, MatchChecks, ShapePart, TensorConstraint,
    TensorConstraints, TensorExpr, TensorPattern, matches_at_with_checks,
    tensor_rewrite_checked_with_checks,
};
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

fn var(name: &str) -> Var {
    name.parse().unwrap()
}
fn info(shape: &[u64]) -> TensorInfo {
    TensorInfo {
        dtype: DType::I32,
        shape: shape.to_vec(),
    }
}
fn constraint(shape: Vec<ShapePart>) -> TensorConstraint {
    TensorConstraint { dtype: None, shape }
}
fn nested(x: Var, y: Var) -> TensorPattern {
    TensorPattern::bind(x, TensorPattern::Var(y))
}
fn leaf() -> (EGraph<OpNode, ()>, Id) {
    let mut graph = EGraph::default();
    let id = graph.add(OpNode::input("x"));
    graph.rebuild();
    (graph, id)
}

#[test]
fn dimensions_and_sequences_prune_before_a_later_capture_is_read() {
    let (graph, root) = leaf();
    let x = var("?x");
    let y = var("?y");
    let reads = AtomicUsize::new(0);
    let metadata = |_: &EGraph<OpNode, ()>, _: Id| {
        assert_eq!(
            reads.fetch_add(1, Ordering::Relaxed),
            0,
            "later capture was visited"
        );
        Some(info(&[2, 5]))
    };
    let checks = MatchChecks::new(
        TensorConstraints::new([
            (
                x,
                constraint(vec![ShapePart::Sequence(Some(0)), ShapePart::Dimension(1)]),
            ),
            (
                y,
                constraint(vec![ShapePart::Wildcard, ShapePart::Wildcard]),
            ),
        ]),
        vec![vec![Binding::Sequence(0), Binding::Dimension(1)]],
        |_, _, _, shapes| {
            assert_eq!(shapes.sequence(0), Some([2].as_slice()));
            Some(shapes.dimension(1)? < 4)
        },
    );
    assert!(
        matches_at_with_checks(&graph, root, &nested(x, y), &checks, &metadata)
            .unwrap()
            .is_empty()
    );
    assert_eq!(reads.load(Ordering::Relaxed), 1);
}

#[test]
fn conditions_wait_for_bindings_and_execute_in_source_order() {
    let (graph, root) = leaf();
    let x = var("?x");
    let y = var("?y");
    let events = Arc::new(Mutex::new(Vec::new()));
    let read_events = events.clone();
    let metadata = move |_: &EGraph<OpNode, ()>, _: Id| {
        read_events.lock().unwrap().push("metadata");
        Some(info(&[3]))
    };
    let evaluate_events = events.clone();
    let checks = MatchChecks::new(
        TensorConstraints::new([
            (x, constraint(vec![ShapePart::Wildcard])),
            (y, constraint(vec![ShapePart::Dimension(0)])),
        ]),
        vec![vec![Binding::Tensor(y), Binding::Dimension(0)], vec![]],
        move |index, _, _, _| {
            evaluate_events
                .lock()
                .unwrap()
                .push(if index == 0 { "first" } else { "second" });
            Some(index == 0)
        },
    );
    assert!(
        matches_at_with_checks(&graph, root, &nested(x, y), &checks, &metadata)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        *events.lock().unwrap(),
        ["metadata", "metadata", "first", "second"]
    );
}

#[test]
fn attribute_witnesses_keep_branch_cursors_independent_and_limits_count_valid_matches() {
    let x = var("?x");
    let a = AttrVar::from("a");
    let mut graph = EGraph::<OpNode, ()>::default();
    let zero = graph.add(OpNode::literal("0", DType::I32).unwrap());
    let one = graph.add(OpNode::literal("1", DType::I32).unwrap());
    graph.union(zero, one);
    graph.rebuild();
    let pattern = TensorPattern::bind(
        x,
        TensorPattern::op(Op::Literal, AttrPattern::Bind(a), vec![]),
    );
    let calls = Arc::new(Mutex::new([0usize; 2]));
    let evaluate_calls = calls.clone();
    let checks = MatchChecks::new(
        TensorConstraints::default(),
        vec![vec![Binding::Tensor(x)], vec![Binding::Attribute(a)]],
        move |index, _, matched, _| {
            evaluate_calls.lock().unwrap()[index] += 1;
            if index == 0 {
                return Some(true);
            }
            Some(matches!(matched.attrs.get(&a)?, OpAttrs::Literal { value, .. } if value == "1"))
        },
    );
    let metadata = |_: &EGraph<OpNode, ()>, _: Id| Some(info(&[]));
    let found = matches_at_with_checks(&graph, zero, &pattern, &checks, &metadata).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(*calls.lock().unwrap(), [1, 2]);
    let rule = tensor_rewrite_checked_with_checks(
        "witness",
        pattern,
        TensorExpr::Var(x),
        checks,
        metadata,
        |_: Op, _: &[TensorInfo], _: &OpAttrs| Some(info(&[])),
        move |_, matched, _| {
            assert!(
                matches!(matched.attrs.get(&a)?, OpAttrs::Literal { value, .. } if value == "1")
            );
            Some(Default::default())
        },
    )
    .unwrap();
    let found = rule
        .searcher
        .search_eclass_with_limit(&graph, zero, 1)
        .unwrap();
    assert_eq!(found.substs.len(), 1);
}

#[test]
fn constant_failure_rejects_before_metadata_and_none_is_a_failed_condition() {
    let (graph, root) = leaf();
    let x = var("?x");
    for answer in [Some(false), None] {
        let checks = MatchChecks::new(
            TensorConstraints::new([(x, constraint(vec![]))]),
            vec![vec![]],
            move |_, _, _, _| answer,
        );
        let metadata = |_: &EGraph<OpNode, ()>, _: Id| -> Option<TensorInfo> {
            panic!("constant condition must reject first")
        };
        assert!(
            matches_at_with_checks(&graph, root, &TensorPattern::Var(x), &checks, &metadata)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn unavailable_dependencies_are_construction_errors() {
    let (graph, root) = leaf();
    let x = var("?x");
    for binding in [
        Binding::Tensor(var("?missing")),
        Binding::Attribute(AttrVar::from("missing")),
        Binding::Dimension(0),
        Binding::Sequence(0),
    ] {
        let checks = MatchChecks::new(
            TensorConstraints::default(),
            vec![vec![binding]],
            |_, _, _, _| Some(true),
        );
        let metadata = |_: &EGraph<OpNode, ()>, _: Id| Some(info(&[]));
        assert!(
            matches_at_with_checks(&graph, root, &TensorPattern::Var(x), &checks, &metadata)
                .is_err()
        );
    }
}

#[test]
fn application_rechecks_changed_dimensions_before_calling_hosts() {
    let (mut graph, root) = leaf();
    let x = var("?x");
    let changed = Arc::new(AtomicBool::new(false));
    let current = changed.clone();
    let checks = MatchChecks::new(
        TensorConstraints::new([(x, constraint(vec![ShapePart::Dimension(0)]))]),
        vec![vec![Binding::Dimension(0)]],
        |_, _, _, shapes| Some(shapes.dimension(0)? == 1),
    );
    let rule = tensor_rewrite_checked_with_checks(
        "current",
        TensorPattern::Var(x),
        TensorExpr::Var(x),
        checks,
        move |_: &EGraph<OpNode, ()>, _: Id| {
            Some(info(&[if current.load(Ordering::Relaxed) {
                2
            } else {
                1
            }]))
        },
        |_: Op, _: &[TensorInfo], _: &OpAttrs| Some(info(&[1])),
        |_, _, _| -> Option<_> { panic!("a stale condition must reject before hosts") },
    )
    .unwrap();
    let found = rule.searcher.search_eclass(&graph, root).unwrap();
    changed.store(true, Ordering::Relaxed);
    let before = graph.total_number_of_nodes();
    assert!(rule.apply(&mut graph, &[found]).is_empty());
    assert_eq!(graph.total_number_of_nodes(), before);
}
