use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, Searcher};
use rust_egg::ir::dialects::tensor_lang;

use rust_egg::ir::{Arity, NodeError, Op, OpAttrs, OpNode};

#[test]
fn canonical_names_and_aliases_resolve_to_one_operation() {
    for (canonical, alias, op) in [
        ("multiply", "mul", Op::TensorLang(tensor_lang::Op::Multiply)),
        (
            "dot_general",
            "dot",
            Op::TensorLang(tensor_lang::Op::DotGeneral),
        ),
    ] {
        assert_eq!(
            tensor_lang::Op::from_name(canonical).map(Op::TensorLang),
            Some(op)
        );
        assert_eq!(
            tensor_lang::Op::from_name(alias).map(Op::TensorLang),
            Some(op)
        );
        assert_eq!(op.name(), canonical);
        assert_eq!(op.alias(), Some(alias));
        assert_eq!(op.display_name(), alias);
    }
    assert_eq!(Op::TensorLang(tensor_lang::Op::Add).alias(), None);
    assert_eq!(Op::TensorLang(tensor_lang::Op::Add).display_name(), "add");
    assert_eq!(
        tensor_lang::Op::from_name("rmsnorm").map(Op::TensorLang),
        Some(Op::TensorLang(tensor_lang::Op::Rmsnorm))
    );
    assert_eq!(Op::from_name("unknown"), None);
}

#[test]
fn construction_rejects_bad_arity_and_attributes() {
    let x = Id::from(0);
    assert_eq!(
        OpNode::from_parts(Op::TensorLang(tensor_lang::Op::Add), vec![x], OpAttrs::None),
        Err(NodeError::Arity {
            op: Op::TensorLang(tensor_lang::Op::Add),
            expected: Arity::Exact(2),
            actual: 1,
        })
    );
    assert_eq!(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::Reduce),
            vec![x],
            OpAttrs::None
        ),
        Err(NodeError::Attributes {
            op: Op::TensorLang(tensor_lang::Op::Reduce)
        })
    );
    assert_eq!(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::Add),
            vec![x, x],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Reduce {
                kind: "sum".into(),
                axes: vec![0],
            })
        ),
        Err(NodeError::Attributes {
            op: Op::TensorLang(tensor_lang::Op::Add)
        })
    );
}

#[test]
fn attributes_follow_dialect_fields() {
    let x = Id::from(0);
    let symbol = OpNode::symbol("x");
    assert_eq!(
        symbol.attrs(),
        &OpAttrs::TensorLang(tensor_lang::OpAttrs::Symbol { name: "x".into() })
    );
    assert_eq!(symbol.symbol_name(), Some("x"));

    let constant = OpNode::constant("weight");
    assert_eq!(
        constant.attrs(),
        &OpAttrs::TensorLang(tensor_lang::OpAttrs::Constant {
            name: "weight".into()
        })
    );
    assert_eq!(constant.symbol_name(), None);

    let reduce = OpNode::reduce("sum", x, vec![0]);
    assert_eq!(reduce.reduction(), Some(("sum", &[0][..], x)));
    assert_eq!(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::AllReduce),
            vec![x],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::CollectiveReduce { kind: "sum".into() }),
        )
        .unwrap()
        .op(),
        Op::TensorLang(tensor_lang::Op::AllReduce)
    );
}

#[test]
fn variadic_arity_comes_from_operand_declaration() {
    assert_eq!(
        Op::TensorLang(tensor_lang::Op::Concatenate).arity(),
        Arity::AtLeast(2)
    );
    let attrs = OpAttrs::TensorLang(tensor_lang::OpAttrs::Concatenate { axis: 0 });
    let x = Id::from(0);
    assert!(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::Concatenate),
            vec![x, x],
            attrs.clone()
        )
        .is_ok()
    );
    assert!(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::Concatenate),
            vec![x, x, x],
            attrs.clone()
        )
        .is_ok()
    );
    assert_eq!(
        OpNode::from_parts(Op::TensorLang(tensor_lang::Op::Concatenate), vec![x], attrs),
        Err(NodeError::Arity {
            op: Op::TensorLang(tensor_lang::Op::Concatenate),
            expected: Arity::AtLeast(2),
            actual: 1,
        })
    );
}

#[test]
fn reduction_pattern_matches_kind_and_axes() {
    let mut egraph = EGraph::<OpNode, ()>::default();
    let x = egraph.add(OpNode::symbol("x"));
    let sum_axis_0 = egraph.add(OpNode::reduce("sum", x, vec![0]));
    egraph.add(OpNode::reduce("sum", x, vec![1]));
    egraph.add(OpNode::reduce("max", x, vec![0]));
    egraph.rebuild();

    let mut ast = PatternAst::default();
    ast.add(ENodeOrVar::Var("?x".parse().unwrap()));
    ast.add(ENodeOrVar::ENode(OpNode::reduce(
        "sum",
        Id::from(0),
        vec![0],
    )));
    let pattern = Pattern::new(ast);
    let matches = pattern.search(&egraph);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].eclass, egraph.find(sum_axis_0));
    assert_eq!(
        OpNode::reduce("sum", x, vec![0]).discriminant(),
        Op::TensorLang(tensor_lang::Op::Reduce)
    );
}
