use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, Searcher};

use rust_egg::ir::{Arity, NodeError, OpAttrs, OpKind, TensorLang};

#[test]
fn canonical_names_and_aliases_resolve_to_one_operation() {
    for (canonical, alias, op) in [
        ("multiply", "mul", OpKind::Multiply),
        ("dot_general", "dot", OpKind::DotGeneral),
    ] {
        assert_eq!(OpKind::from_name(canonical), Some(op));
        assert_eq!(OpKind::from_name(alias), Some(op));
        assert_eq!(op.name(), canonical);
        assert_eq!(op.alias(), Some(alias));
        assert_eq!(op.display_name(), alias);
    }
    assert_eq!(OpKind::Add.alias(), None);
    assert_eq!(OpKind::Add.display_name(), "add");
    assert_eq!(OpKind::from_name("rmsnorm"), Some(OpKind::Rmsnorm));
    assert_eq!(OpKind::from_name("unknown"), None);
}

#[test]
fn construction_rejects_bad_arity_and_attributes() {
    let x = Id::from(0);
    assert_eq!(
        TensorLang::new(OpKind::Add, vec![x], OpAttrs::None),
        Err(NodeError::Arity {
            op: OpKind::Add,
            expected: Arity::Exact(2),
            actual: 1,
        })
    );
    assert_eq!(
        TensorLang::new(OpKind::Reduce, vec![x], OpAttrs::None),
        Err(NodeError::Attributes { op: OpKind::Reduce })
    );
    assert_eq!(
        TensorLang::new(
            OpKind::Add,
            vec![x, x],
            OpAttrs::Reduce {
                kind: "sum".into(),
                axes: vec![0],
            }
        ),
        Err(NodeError::Attributes { op: OpKind::Add })
    );
}

#[test]
fn attributes_follow_dialect_fields() {
    let x = Id::from(0);
    let symbol = TensorLang::symbol("x");
    assert_eq!(symbol.attrs(), &OpAttrs::Symbol { name: "x".into() });
    assert_eq!(symbol.symbol_name(), Some("x"));

    let constant = TensorLang::constant("weight");
    assert_eq!(
        constant.attrs(),
        &OpAttrs::Constant {
            name: "weight".into()
        }
    );
    assert_eq!(constant.symbol_name(), None);

    let reduce = TensorLang::reduce("sum", x, vec![0]);
    assert_eq!(reduce.reduction(), Some(("sum", &[0][..], x)));
    assert_eq!(
        TensorLang::new(
            OpKind::AllReduce,
            vec![x],
            OpAttrs::CollectiveReduce { kind: "sum".into() },
        )
        .unwrap()
        .op(),
        OpKind::AllReduce
    );
}

#[test]
fn variadic_arity_comes_from_operand_declaration() {
    assert_eq!(OpKind::Concatenate.arity(), Arity::AtLeast(2));
    let attrs = OpAttrs::Concatenate { axis: 0 };
    let x = Id::from(0);
    assert!(TensorLang::new(OpKind::Concatenate, vec![x, x], attrs.clone()).is_ok());
    assert!(TensorLang::new(OpKind::Concatenate, vec![x, x, x], attrs.clone()).is_ok());
    assert_eq!(
        TensorLang::new(OpKind::Concatenate, vec![x], attrs),
        Err(NodeError::Arity {
            op: OpKind::Concatenate,
            expected: Arity::AtLeast(2),
            actual: 1,
        })
    );
}

#[test]
fn reduction_pattern_matches_kind_and_axes() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("x"));
    let sum_axis_0 = egraph.add(TensorLang::reduce("sum", x, vec![0]));
    egraph.add(TensorLang::reduce("sum", x, vec![1]));
    egraph.add(TensorLang::reduce("max", x, vec![0]));
    egraph.rebuild();

    let mut ast = PatternAst::default();
    ast.add(ENodeOrVar::Var("?x".parse().unwrap()));
    ast.add(ENodeOrVar::ENode(TensorLang::reduce(
        "sum",
        Id::from(0),
        vec![0],
    )));
    let pattern = Pattern::new(ast);
    let matches = pattern.search(&egraph);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].eclass, egraph.find(sum_axis_0));
    assert_eq!(
        TensorLang::reduce("sum", x, vec![0]).discriminant(),
        OpKind::Reduce
    );
}
