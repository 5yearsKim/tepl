use egg::{EGraph, Var};
use rust_egg::ir::patterns::{
    AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern, matches_at, tensor_rewrite,
};
use rust_egg::ir::rules::rule_commute_add;
use rust_egg::ir::{OpAttrs, OpKind, TensorLang};

#[test]
fn invalid_rhs_definitions_are_rejected_when_building_the_rule() {
    let x = "?X".parse::<Var>().unwrap();
    let y = "?Y".parse::<Var>().unwrap();
    let cases = [
        (
            TensorExpr::op(
                OpKind::Add,
                AttrExpr::Exact(OpAttrs::None),
                vec![TensorExpr::Var(x)],
            ),
            "invalid RHS arity",
        ),
        (
            TensorExpr::op(
                OpKind::Transpose,
                AttrExpr::Exact(OpAttrs::None),
                vec![TensorExpr::Var(x)],
            ),
            "invalid attributes",
        ),
        (TensorExpr::Var(y), "unbound"),
        (
            TensorExpr::op(
                OpKind::Transpose,
                AttrExpr::Captured("missing".into()),
                vec![TensorExpr::Var(x)],
            ),
            "not captured on the LHS",
        ),
    ];
    for (rhs, message) in cases {
        let error = tensor_rewrite::<(), _>("invalid", TensorPattern::Var(x), rhs, |_, _| {
            Some(Default::default())
        })
        .err()
        .expect("invalid RHS definition should be rejected");
        assert!(error.contains(message), "{error}");
    }
}

#[test]
fn failed_host_or_missing_invalid_descriptors_leave_no_partial_rhs() {
    let x = "?X".parse::<Var>().unwrap();
    let descriptor = AttrVar::from("transpose");
    for mode in 0..4 {
        let mut egraph = EGraph::<TensorLang, ()>::default();
        let input = egraph.add(TensorLang::symbol("X"));
        egraph.rebuild();
        let before = egraph.total_number_of_nodes();
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    OpKind::Negate,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(x)],
                ),
                TensorExpr::op(
                    OpKind::Transpose,
                    AttrExpr::Derived(descriptor),
                    vec![TensorExpr::Var(x)],
                ),
            ],
        );
        let rule = tensor_rewrite(
            "construct",
            TensorPattern::Var(x),
            rhs,
            move |_, _| match mode {
                0 => None,
                1 => Some(Default::default()),
                2 => Some([(descriptor, OpAttrs::None)].into()),
                _ => Some(
                    [(
                        descriptor,
                        OpAttrs::Transpose {
                            permutation: vec![0],
                        },
                    )]
                    .into(),
                ),
            },
        )
        .unwrap();
        let found = rule.search(&egraph);
        let applied = rule.apply(&mut egraph, &found);
        if mode < 3 {
            assert!(applied.is_empty());
            assert_eq!(egraph.total_number_of_nodes(), before);
            assert!(
                egraph
                    .lookup(TensorLang::unary(OpKind::Negate, input).unwrap())
                    .is_none()
            );
        } else {
            assert_eq!(applied.len(), 1);
            assert_eq!(egraph.total_number_of_nodes(), before + 3);
        }
    }
}

#[test]
fn search_limit_and_batch_application_cover_substitutions_in_one_eclass() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let mut additions = Vec::new();
    for index in 0..12 {
        let x = egraph.add(TensorLang::symbol(format!("x{index}")));
        let y = egraph.add(TensorLang::symbol(format!("y{index}")));
        let add = egraph.add(TensorLang::binary(OpKind::Add, x, y).unwrap());
        if let Some(&(root, _, _)) = additions.first() {
            egraph.union(root, add);
        }
        additions.push((add, x, y));
    }
    egraph.rebuild();

    let rule = rule_commute_add::build_rewrite::<()>().unwrap();
    let limited = rule.search_with_limit(&egraph, 1);
    assert_eq!(limited.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 12);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 12);
    egraph.rebuild();

    for (_, x, y) in additions {
        let swapped = egraph
            .lookup(TensorLang::binary(OpKind::Add, y, x).unwrap())
            .expect("every substitution should be applied");
        assert_eq!(egraph.find(swapped), egraph.find(found[0].eclass));
    }
}

#[test]
fn batch_application_keeps_distinct_attribute_witnesses() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("x"));
    let y = egraph.add(TensorLang::symbol("y"));
    let attrs = [
        OpAttrs::DotGeneral {
            lhs_contracting: vec![0],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        },
        OpAttrs::DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![1],
            lhs_batch: vec![],
            rhs_batch: vec![],
        },
    ];
    let first =
        egraph.add(TensorLang::new(OpKind::DotGeneral, vec![x, y], attrs[0].clone()).unwrap());
    let second =
        egraph.add(TensorLang::new(OpKind::DotGeneral, vec![x, y], attrs[1].clone()).unwrap());
    egraph.union(first, second);
    egraph.rebuild();

    let [x_var, y_var] = ["?x", "?y"].map(|name| name.parse::<Var>().unwrap());
    let attr_var = AttrVar::from("dot");
    let lhs = TensorPattern::op(
        OpKind::DotGeneral,
        AttrPattern::Bind(attr_var),
        vec![TensorPattern::Var(x_var), TensorPattern::Var(y_var)],
    );
    let rhs = TensorExpr::op(
        OpKind::DotGeneral,
        AttrExpr::Captured(attr_var),
        vec![TensorExpr::Var(y_var), TensorExpr::Var(x_var)],
    );
    let rule =
        tensor_rewrite("swap_dot_inputs", lhs, rhs, |_, _| Some(Default::default())).unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 2);
    egraph.rebuild();

    for attr in attrs {
        let swapped = egraph
            .lookup(TensorLang::new(OpKind::DotGeneral, vec![y, x], attr).unwrap())
            .expect("each attribute witness should produce a node");
        assert_eq!(egraph.find(swapped), egraph.find(first));
    }
}

#[test]
fn grounded_lookup_rejects_missing_nodes_and_nodes_in_another_eclass() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let [x, y, z] = ["x", "y", "z"].map(|name| egraph.add(TensorLang::symbol(name)));
    let xy = egraph.add(TensorLang::binary(OpKind::Multiply, x, y).unwrap());
    let zy = egraph.add(TensorLang::binary(OpKind::Multiply, z, y).unwrap());
    let nx = egraph.add(TensorLang::unary(OpKind::Negate, x).unwrap());
    let ny = egraph.add(TensorLang::unary(OpKind::Negate, y).unwrap());
    let valid = egraph.add(TensorLang::binary(OpKind::Add, xy, nx).unwrap());
    let wrong_class = egraph.add(TensorLang::binary(OpKind::Add, xy, ny).unwrap());
    let missing = egraph.add(TensorLang::binary(OpKind::Add, zy, ny).unwrap());
    egraph.rebuild();

    let [x_var, y_var] = ["?x", "?y"].map(|name| name.parse::<Var>().unwrap());
    let pattern = TensorPattern::op(
        OpKind::Add,
        AttrPattern::Exact(OpAttrs::None),
        vec![
            TensorPattern::op(
                OpKind::Multiply,
                AttrPattern::Exact(OpAttrs::None),
                vec![TensorPattern::Var(x_var), TensorPattern::Var(y_var)],
            ),
            TensorPattern::op(
                OpKind::Negate,
                AttrPattern::Exact(OpAttrs::None),
                vec![TensorPattern::Var(x_var)],
            ),
        ],
    );
    assert_eq!(matches_at(&egraph, valid, &pattern).len(), 1);
    assert!(matches_at(&egraph, wrong_class, &pattern).is_empty());
    assert!(matches_at(&egraph, missing, &pattern).is_empty());
}

#[test]
fn grounded_nested_subtrees_preserve_new_and_repeated_binders() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("x"));
    let y = egraph.add(TensorLang::symbol("y"));
    let xy = egraph.add(TensorLang::binary(OpKind::Multiply, x, y).unwrap());
    let neg = egraph.add(TensorLang::unary(OpKind::Negate, xy).unwrap());
    let root = egraph.add(TensorLang::binary(OpKind::Add, xy, neg).unwrap());
    egraph.rebuild();

    let [x_var, y_var, z_var] = ["?x", "?y", "?z"].map(|name| name.parse::<Var>().unwrap());
    let multiply = || {
        TensorPattern::op(
            OpKind::Multiply,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x_var), TensorPattern::Var(y_var)],
        )
    };
    // The second branch has known operands but must still introduce ?z when
    // the first branch does not bind it. Also exercise lookup through an
    // already-bound nested binder and through a plain nested operation.
    for bind_first in [false, true] {
        for bind_second in [false, true] {
            let pattern = TensorPattern::op(
                OpKind::Add,
                AttrPattern::Exact(OpAttrs::None),
                vec![
                    if bind_first {
                        TensorPattern::bind(z_var, multiply())
                    } else {
                        multiply()
                    },
                    TensorPattern::op(
                        OpKind::Negate,
                        AttrPattern::Exact(OpAttrs::None),
                        vec![if bind_second {
                            TensorPattern::bind(z_var, multiply())
                        } else {
                            multiply()
                        }],
                    ),
                ],
            );
            let found = matches_at(&egraph, root, &pattern);
            assert_eq!(found.len(), 1);
            if bind_first || bind_second {
                assert_eq!(found[0].tensors[z_var], xy);
            }
        }
    }
}

#[test]
fn known_operands_do_not_hide_unbound_or_wildcard_attribute_witnesses() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("x"));
    let first = egraph.add(
        TensorLang::new(
            OpKind::Transpose,
            vec![x],
            OpAttrs::Transpose {
                permutation: vec![0, 1],
            },
        )
        .unwrap(),
    );
    let second = egraph.add(
        TensorLang::new(
            OpKind::Transpose,
            vec![x],
            OpAttrs::Transpose {
                permutation: vec![1, 0],
            },
        )
        .unwrap(),
    );
    egraph.union(first, second);
    let root = egraph.add(TensorLang::binary(OpKind::Add, first, first).unwrap());
    egraph.rebuild();

    let x_var = "?x".parse::<Var>().unwrap();
    let outer = AttrVar::from("outer");
    let inner = AttrVar::from("inner");
    for second_attrs in [AttrPattern::Bind(inner), AttrPattern::Any] {
        let pattern = TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::op(
                    OpKind::Transpose,
                    AttrPattern::Bind(outer),
                    vec![TensorPattern::Var(x_var)],
                ),
                TensorPattern::op(
                    OpKind::Transpose,
                    second_attrs,
                    vec![TensorPattern::Var(x_var)],
                ),
            ],
        );
        // Both choices in the first child must pair with both choices in the second.
        let found = matches_at(&egraph, root, &pattern);
        assert_eq!(found.len(), 4);
        assert!(found.iter().all(|matched| matched.tensors[x_var] == x));
    }
}

#[test]
fn matching_after_union_falls_back_until_the_graph_is_rebuilt() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("x"));
    let y = egraph.add(TensorLang::symbol("y"));
    let neg_y = egraph.add(TensorLang::unary(OpKind::Negate, y).unwrap());
    let root = egraph.add(TensorLang::binary(OpKind::Add, x, neg_y).unwrap());
    // Give x more parents so it becomes the representative of the union.
    egraph.add(TensorLang::unary(OpKind::Exp, x).unwrap());
    egraph.rebuild();
    egraph.union(x, y);
    assert!(!egraph.clean);
    assert_eq!(egraph.find(y), x);
    assert!(
        egraph
            .lookup(TensorLang::unary(OpKind::Negate, x).unwrap())
            .is_none()
    );

    let var = "?x".parse::<Var>().unwrap();
    let pattern = TensorPattern::op(
        OpKind::Add,
        AttrPattern::Exact(OpAttrs::None),
        vec![
            TensorPattern::Var(var),
            TensorPattern::op(
                OpKind::Negate,
                AttrPattern::Exact(OpAttrs::None),
                vec![TensorPattern::Var(var)],
            ),
        ],
    );
    assert_eq!(matches_at(&egraph, root, &pattern).len(), 1);
    egraph.rebuild();
    assert_eq!(matches_at(&egraph, root, &pattern).len(), 1);
}
