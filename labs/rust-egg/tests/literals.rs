use egg::{EGraph, Language, Rewrite};
use rust_egg::ir::patterns::{TensorExpr, TensorPattern, matches_at, tensor_rewrite};
use rust_egg::ir::rules::{
    rule_commute_float_literal, rule_commute_integer_literal, rule_commute_negative_literal,
};
use rust_egg::ir::{NodeError, OpAttrs, OpKind, TensorLang};

#[test]
fn literals_preserve_kind_spelling_and_precision() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let values = [
        "1",
        "1.0",
        "1.00",
        "-0.0",
        "0.0",
        "+1",
        "001",
        "999999999999999999999999",
        "1.000000000000000000000001",
    ];
    let ids: Vec<_> = values
        .iter()
        .map(|value| {
            let node = TensorLang::literal(*value).unwrap();
            assert!(node.children().is_empty());
            assert_eq!(node.to_string(), *value);
            egraph.add(node)
        })
        .collect();
    egraph.rebuild();
    for (i, value) in values.iter().enumerate() {
        let pattern = TensorPattern::literal(*value).unwrap();
        for (j, id) in ids.iter().enumerate() {
            assert_eq!(
                matches_at(&egraph, *id, &pattern).len(),
                usize::from(i == j)
            );
        }
    }
}

#[test]
fn invalid_literals_and_literal_nodes_are_rejected() {
    for value in [
        "", "+", "-", "1.", ".5", "1e3", "1.0f", "NaN", "inf", "1.2.3", " 1", "1 ",
    ] {
        assert_eq!(
            TensorLang::literal(value),
            Err(NodeError::Attributes {
                op: OpKind::Literal
            })
        );
        assert!(TensorPattern::literal(value).is_err());
        assert!(TensorExpr::literal(value).is_err());
    }
    assert!(TensorLang::new(OpKind::Literal, vec![], OpAttrs::None).is_err());
    assert!(
        TensorLang::new(
            OpKind::Literal,
            vec![0.into()],
            OpAttrs::Literal { value: "1".into() }
        )
        .is_err()
    );
}

#[test]
fn reference_rules_match_only_their_literal_and_construct_the_rhs() {
    let rules: [(&str, Rewrite<TensorLang, ()>); 3] = [
        ("1", rule_commute_integer_literal::build_rewrite().unwrap()),
        ("1.0", rule_commute_float_literal::build_rewrite().unwrap()),
        (
            "-0.5",
            rule_commute_negative_literal::build_rewrite().unwrap(),
        ),
    ];
    for (value, rule) in rules {
        let mut egraph = EGraph::<TensorLang, ()>::default();
        let x = egraph.add(TensorLang::symbol("X"));
        let mut expected = None;
        for input in ["1", "1.0", "1.00", "-0.5", "2"] {
            let literal = egraph.add(TensorLang::literal(input).unwrap());
            let root = egraph.add(TensorLang::binary(OpKind::Add, x, literal).unwrap());
            if input == value {
                expected = Some((root, literal));
            }
        }
        // A named constant with numeric-looking text is still a different node.
        let named = egraph.add(TensorLang::constant(value));
        egraph.add(TensorLang::binary(OpKind::Add, x, named).unwrap());
        egraph.rebuild();
        let found = rule.search(&egraph);
        assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
        let (root, literal) = expected.unwrap();
        assert_eq!(found[0].eclass, egraph.find(root));
        assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
        egraph.rebuild();
        let rhs = egraph
            .lookup(TensorLang::binary(OpKind::Add, literal, x).unwrap())
            .unwrap();
        assert_eq!(egraph.find(root), egraph.find(rhs));
    }
}

#[test]
fn literals_can_be_rewrite_roots_and_insert_new_values() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let root = egraph.add(TensorLang::literal("1").unwrap());
    egraph.rebuild();
    // Structural fixture only: the test callback authorizes this substitution.
    let rule = tensor_rewrite(
        "literal_root",
        TensorPattern::literal("1").unwrap(),
        TensorExpr::literal("-2.5").unwrap(),
        |_, _| Some(Default::default()),
    )
    .unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.len(), 1);
    rule.apply(&mut egraph, &found);
    egraph.rebuild();
    let output = egraph.lookup(TensorLang::literal("-2.5").unwrap()).unwrap();
    assert_eq!(egraph.find(root), egraph.find(output));
}
