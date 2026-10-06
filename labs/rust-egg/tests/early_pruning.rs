mod support;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ShapePart::{Dimension as D, Sequence as S, Wildcard as W};
use egg::{EGraph, Id, Language, Var};
use rust_egg::host::nodes::{binary, symbol};
use rust_egg::ir::TensorInfo;
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::rewriting::{
    AttrPattern, MetadataBindings, ShapePart, TensorConstraint, TensorConstraints, TensorExpr,
    TensorPattern, matches_at, matches_at_with_constraints,
    tensor_rewrite_checked_with_constraints,
};
use rust_egg::ir::{DType, OpAttrs, OpNode};

type Facts = Arc<Mutex<HashMap<Id, TensorInfo>>>;
fn info(shape: &[u64]) -> TensorInfo {
    TensorInfo {
        shape: shape.to_vec(),
        dtype: DType::F32,
    }
}
fn metadata(facts: Facts) -> impl Fn(&EGraph<OpNode, ()>, Id) -> Option<TensorInfo> {
    move |graph, id| {
        facts
            .lock()
            .unwrap()
            .iter()
            .find(|(key, _)| graph.find(**key) == graph.find(id))
            .map(|(_, value)| value.clone())
    }
}
fn constraint(var: Var, shape: Vec<ShapePart>) -> (Var, TensorConstraint) {
    (var, TensorConstraint { dtype: None, shape })
}
fn add(children: Vec<TensorPattern>) -> TensorPattern {
    TensorPattern::op(
        tensor_lang::Op::Add,
        AttrPattern::Exact(OpAttrs::None),
        children,
    )
}

// Make the first structural alternative fail only after binding a dimension.
fn alternatives() -> (
    EGraph<OpNode, ()>,
    Id,
    [Var; 2],
    TensorPattern,
    TensorConstraints,
    Facts,
    Id,
) {
    let mut graph = EGraph::default();
    let x = graph.add(symbol("x"));
    let y = graph.add(symbol("y"));
    let xy = graph.add(binary(tensor_lang::Op::Add, x, y).unwrap());
    let yx = graph.add(binary(tensor_lang::Op::Add, y, x).unwrap());
    graph.union(xy, yx);
    graph.rebuild();
    let nodes = &graph[graph.find(xy)].nodes;
    let bad = nodes[0].children()[0];
    let good = nodes[1].children()[0];
    let [a, b] = ["?a", "?b"].map(|name| name.parse().unwrap());
    let pattern = add(vec![TensorPattern::Var(a), TensorPattern::Var(b)]);
    let constraints =
        TensorConstraints::new([constraint(a, vec![D(0), D(0)]), constraint(b, vec![W, W])]);
    let facts = Arc::new(Mutex::new(HashMap::from([
        (bad, info(&[2, 3])),
        (good, info(&[4, 4])),
        (graph.find(xy), info(&[4, 4])),
    ])));
    (graph, xy, [a, b], pattern, constraints, facts, good)
}

#[test]
fn pruning_equals_complete_filtering_and_failed_branches_do_not_leak() {
    let (graph, root, [a, b], pattern, constraints, facts, good) = alternatives();
    let metadata = metadata(facts);
    let raw = matches_at(&graph, root, &pattern);
    assert_eq!(raw.len(), 2);
    let filtered: Vec<_> = raw
        .iter()
        .filter(|matched| {
            constraints
                .check_match(&graph, matched, &metadata)
                .is_some()
        })
        .collect();
    let pruned =
        matches_at_with_constraints(&graph, root, &pattern, &constraints, &metadata).unwrap();
    assert_eq!(pruned.len(), 1);
    assert_eq!(pruned.len(), filtered.len());
    assert_eq!(pruned[0].tensors[a], good);
    assert_eq!(pruned[0].tensors[a], filtered[0].tensors[a]);
    assert_eq!(pruned[0].tensors[b], filtered[0].tensors[b]);
    assert_eq!(pruned[0].attrs, filtered[0].attrs);
}

#[test]
fn rejected_candidates_do_not_consume_limit_and_semantics_run_only_on_application() {
    let (mut graph, root, [a, _], pattern, constraints, facts, good) = alternatives();
    let calls = Arc::new(Mutex::new(0));
    let trace = calls.clone();
    let rule = tensor_rewrite_checked_with_constraints(
        "pruned-limit",
        pattern,
        TensorExpr::Var(a),
        constraints,
        metadata(facts),
        support::fixture_inference,
        move |_, _, dimensions| {
            assert_eq!(dimensions.dimension(0), Some(4));
            *trace.lock().unwrap() += 1;
            Some(Default::default())
        },
    )
    .unwrap();
    assert!(rule.search_with_limit(&graph, 0).is_empty());
    let found = rule.search_with_limit(&graph, 1);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].substs.len(), 1);
    assert_eq!(found[0].substs[0][a], good);
    assert_eq!(*calls.lock().unwrap(), 0);
    assert_eq!(rule.apply(&mut graph, &found).len(), 1);
    assert_eq!(*calls.lock().unwrap(), 1);
    assert_eq!(graph.find(root), graph.find(good));
}

#[test]
fn both_application_paths_recheck_current_metadata() {
    for single in [false, true] {
        let (mut graph, root, [a, _], pattern, constraints, facts, good) = alternatives();
        let calls = Arc::new(Mutex::new(0));
        let trace = calls.clone();
        let rule = tensor_rewrite_checked_with_constraints(
            "changed-facts",
            pattern,
            TensorExpr::Var(a),
            constraints,
            metadata(facts.clone()),
            support::fixture_inference,
            move |_, _, _| {
                *trace.lock().unwrap() += 1;
                Some(Default::default())
            },
        )
        .unwrap();
        let found = rule.search(&graph);
        assert_eq!(found.len(), 1);
        facts.lock().unwrap().insert(good, info(&[4, 5]));
        let before = graph.total_size();
        let changed = if single {
            rule.applier
                .apply_one(&mut graph, root, &found[0].substs[0], None, rule.name)
        } else {
            rule.apply(&mut graph, &found)
        };
        assert!(changed.is_empty());
        assert_eq!(*calls.lock().unwrap(), 0);
        assert_eq!(graph.total_size(), before);
        assert_ne!(graph.find(root), graph.find(good));
    }
}

#[test]
fn incompatible_first_operand_skips_later_subtree_and_metadata() {
    let mut graph = EGraph::default();
    let x = graph.add(symbol("x"));
    let y = graph.add(symbol("y"));
    let inner = graph.add(binary(tensor_lang::Op::Add, y, y).unwrap());
    let root = graph.add(binary(tensor_lang::Op::Add, x, inner).unwrap());
    graph.rebuild();
    let [a, b] = ["?a", "?b"].map(|name| name.parse().unwrap());
    let pattern = add(vec![
        TensorPattern::Var(a),
        add(vec![TensorPattern::Var(b), TensorPattern::Var(b)]),
    ]);
    let constraints =
        TensorConstraints::new([constraint(a, vec![D(0)]), constraint(b, vec![D(0)])]);
    let reads = Arc::new(Mutex::new(Vec::new()));
    let trace = reads.clone();
    let metadata = move |_: &EGraph<OpNode, ()>, id| {
        trace.lock().unwrap().push(id);
        Some(info(if id == x { &[2, 3] } else { &[2] }))
    };
    assert!(
        matches_at_with_constraints(&graph, root, &pattern, &constraints, &metadata)
            .unwrap()
            .is_empty()
    );
    assert_eq!(*reads.lock().unwrap(), vec![x]);
}

#[test]
fn mixed_middle_sequences_and_inherited_restrictions_share_bindings() {
    let [a, b] = ["?a", "?b"].map(|name| name.parse().unwrap());
    let pattern = add(vec![TensorPattern::Var(a), TensorPattern::Var(b)]);
    let constraints = TensorConstraints::new([
        constraint(a, vec![D(0), S(Some(2)), W, D(0)]),
        constraint(a, vec![W, S(None), D(1), W]),
        constraint(b, vec![S(Some(2)), D(1), D(0)]),
        (
            b,
            TensorConstraint {
                dtype: Some(rust_egg::ir::rewriting::DTypeConstraint::Exact(DType::F32)),
                shape: vec![S(None)],
            },
        ),
    ]);
    constraints.validate(&pattern).unwrap();
    for (left, right, accepted) in [
        (vec![3, 7, 8, 9, 3], vec![7, 8, 9, 3], true),
        (vec![3, 9, 3], vec![9, 3], true),
        (vec![0, 0, 0], vec![0, 0], true),
        (vec![3, 7, 8, 9, 3], vec![7, 6, 9, 3], false),
        (vec![3, 7, 8, 9, 3], vec![7, 8, 4, 3], false),
        (vec![3, 7, 8, 9, 4], vec![7, 8, 9, 3], false),
        (vec![3], vec![3], false),
    ] {
        let mut bindings = MetadataBindings::default();
        let actual = constraints
            .check_capture(a, &info(&left), &mut bindings)
            .and_then(|_| constraints.check_capture(b, &info(&right), &mut bindings))
            .is_some();
        assert_eq!(actual, accepted, "{left:?} {right:?}");
    }
}

#[test]
fn malformed_constraint_plans_are_rejected_before_search() {
    let a: Var = "?a".parse().unwrap();
    let b: Var = "?b".parse().unwrap();
    let pattern = TensorPattern::Var(a);
    for declarations in [
        vec![constraint(b, vec![])],
        vec![constraint(a, vec![S(Some(0)), S(None)])],
        vec![constraint(a, vec![D(0), S(Some(0))])],
        vec![constraint(a, vec![D(0)]), constraint(a, vec![S(Some(0))])],
    ] {
        let constraints = TensorConstraints::new(declarations);
        assert!(constraints.validate(&pattern).is_err());
        assert!(
            tensor_rewrite_checked_with_constraints::<(), _, _, _>(
                "invalid",
                pattern.clone(),
                TensorExpr::Var(a),
                constraints,
                support::fixture_metadata::<()>,
                support::fixture_inference,
                |_, _, _| Some(Default::default()),
            )
            .is_err()
        );
    }
}

#[test]
fn missing_metadata_and_dtype_mismatches_reject_only_constrained_captures() {
    let mut graph = EGraph::default();
    let root = graph.add(symbol("x"));
    graph.rebuild();
    let a: Var = "?a".parse().unwrap();
    let pattern = TensorPattern::Var(a);
    let constraints = TensorConstraints::new([(
        a,
        TensorConstraint {
            dtype: Some(rust_egg::ir::rewriting::DTypeConstraint::Exact(DType::F32)),
            shape: vec![S(None)],
        },
    )]);
    for metadata in [
        None,
        Some(TensorInfo {
            dtype: DType::I32,
            shape: vec![2],
        }),
    ] {
        let reader = move |_: &EGraph<OpNode, ()>, _| metadata.clone();
        assert!(
            matches_at_with_constraints(&graph, root, &pattern, &constraints, &reader)
                .unwrap()
                .is_empty()
        );
    }
    let never_read = |_: &EGraph<OpNode, ()>, _| -> Option<TensorInfo> {
        panic!("unconstrained captures do not require metadata")
    };
    assert_eq!(
        matches_at_with_constraints(
            &graph,
            root,
            &pattern,
            &TensorConstraints::default(),
            &never_read
        )
        .unwrap()
        .len(),
        1
    );
}

#[test]
fn mixed_constraint_search_matches_filtering_across_alternative_witnesses() {
    let shapes = [
        vec![],
        vec![0],
        vec![2, 2],
        vec![3, 9, 3],
        vec![3, 7, 8, 9, 3],
        vec![3, 7, 8, 9, 4],
        vec![7, 8, 9, 3],
    ];
    let [a, b] = ["?a", "?b"].map(|name| name.parse().unwrap());
    let pattern = add(vec![TensorPattern::Var(a), TensorPattern::Var(b)]);
    for left in &shapes {
        for right in &shapes {
            let mut graph = EGraph::default();
            let x = graph.add(symbol("x"));
            let y = graph.add(symbol("y"));
            let xy = graph.add(binary(tensor_lang::Op::Add, x, y).unwrap());
            let yx = graph.add(binary(tensor_lang::Op::Add, y, x).unwrap());
            graph.union(xy, yx);
            graph.rebuild();
            let reader = move |_: &EGraph<OpNode, ()>, id| {
                if id == x {
                    Some(info(left))
                } else if id == y {
                    Some(info(right))
                } else {
                    None
                }
            };
            for declarations in [
                vec![
                    constraint(a, vec![D(0), S(Some(2)), W, D(0)]),
                    constraint(a, vec![W, S(None), D(1), W]),
                    constraint(b, vec![S(Some(2)), D(1), D(0)]),
                ],
                vec![
                    constraint(a, vec![S(Some(2)), D(0)]),
                    constraint(b, vec![D(0), S(Some(2))]),
                ],
                vec![
                    constraint(a, vec![D(0), S(Some(2))]),
                    constraint(b, vec![D(0), S(Some(2))]),
                ],
                vec![constraint(a, vec![]), constraint(b, vec![S(None)])],
            ] {
                let constraints = TensorConstraints::new(declarations);
                let expected: Vec<_> = matches_at(&graph, xy, &pattern)
                    .into_iter()
                    .filter(|matched| constraints.check_match(&graph, matched, &reader).is_some())
                    .map(|matched| (matched.tensors, matched.attrs))
                    .collect();
                let actual: Vec<_> =
                    matches_at_with_constraints(&graph, xy, &pattern, &constraints, &reader)
                        .unwrap()
                        .into_iter()
                        .map(|matched| (matched.tensors, matched.attrs))
                        .collect();
                assert_eq!(actual, expected, "{left:?} {right:?}");
            }
        }
    }
}

#[test]
fn canonical_capture_uses_agreed_metadata_and_rejects_conflicting_alternatives() {
    use rust_egg::ir::analysis::{TensorAnalysis, TensorBindingTable, tensor_info};
    for second_shape in [vec![2], vec![3]] {
        let mut bindings = TensorBindingTable::default();
        bindings.register_symbol("x", info(&[2])).unwrap();
        bindings.register_symbol("y", info(&second_shape)).unwrap();
        let mut graph = EGraph::new(TensorAnalysis::new(bindings));
        let x = graph.add(OpNode::input("x"));
        let y = graph.add(OpNode::input("y"));
        graph.union(x, y);
        graph.rebuild();
        let a = "?a".parse().unwrap();
        let pattern = TensorPattern::Var(a);
        let constraints = TensorConstraints::new([constraint(a, vec![D(0)])]);
        let found =
            matches_at_with_constraints(&graph, y, &pattern, &constraints, &tensor_info).unwrap();
        assert_eq!(found.len(), usize::from(second_shape == vec![2]));
        if !found.is_empty() {
            assert_eq!(found[0].tensors[a], graph.find(y));
        }
    }
}

#[test]
fn rejected_binder_skips_inner_capture_checks() {
    let mut graph = EGraph::default();
    let x = graph.add(OpNode::input("x"));
    let root = graph.add(binary(tensor_lang::Op::Add, x, x).unwrap());
    graph.rebuild();
    let [bound, leaf] = ["?root", "?x"].map(|name| name.parse().unwrap());
    let pattern = TensorPattern::bind(
        bound,
        add(vec![TensorPattern::Var(leaf), TensorPattern::Var(leaf)]),
    );
    let constraints =
        TensorConstraints::new([constraint(bound, vec![D(0)]), constraint(leaf, vec![D(0)])]);
    assert_eq!(matches_at(&graph, root, &pattern).len(), 1);
    for accepted in [false, true] {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let trace = reads.clone();
        let metadata = move |_: &EGraph<OpNode, ()>, id| {
            trace.lock().unwrap().push(id);
            Some(info(if id == root && !accepted { &[] } else { &[2] }))
        };
        let found =
            matches_at_with_constraints(&graph, root, &pattern, &constraints, &metadata).unwrap();
        assert_eq!(found.len(), usize::from(accepted));
        assert_eq!(
            *reads.lock().unwrap(),
            if accepted { vec![root, x] } else { vec![root] }
        );
    }
}
