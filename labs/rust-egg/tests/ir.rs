use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, Searcher};
use rust_egg::host::nodes::{constant, reduce, reduction, symbol, symbol_name};
use rust_egg::ir::dialects::tensor_lang as t;
use rust_egg::ir::{Arity, Op, OpAttrs, OpNode};

#[test]
fn canonical_names_and_aliases_resolve_to_one_operation() {
    for (canonical, alias, op) in [
        ("multiply", "mul", t::Op::Multiply),
        ("dot_general", "dot", t::Op::DotGeneral),
    ] {
        assert_eq!(t::Op::from_name(canonical), Some(op));
        assert_eq!(t::Op::from_name(alias), Some(op));
        assert_eq!(
            Op::from_name(&format!("TensorLang.{alias}")),
            Some(op.into())
        );
        assert_eq!(op.name(), canonical);
    }
    assert_eq!(t::Op::from_name("unknown"), None);
    assert_eq!(Op::from_name("add"), None);
}
#[test]
fn construction_rejects_bad_arity_and_attributes() {
    let x = Id::from(0);
    assert!(OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x]).is_err());
    assert!(OpNode::new(t::Op::Reduce, t::OpAttrs::None, vec![x]).is_err());
    assert!(
        OpNode::new(
            t::Op::Add,
            t::OpAttrs::ReduceAttrs {
                kind: "sum".into(),
                axes: vec![0]
            },
            vec![x, x]
        )
        .is_err()
    );
}
#[test]
fn attributes_follow_dialect_fields() {
    let x = Id::from(0);
    let input = symbol("x");
    assert_eq!(
        input.attrs(),
        &OpAttrs::TensorLang(t::OpAttrs::SymbolAttrs { name: "x".into() })
    );
    assert_eq!(symbol_name(&input), Some("x"));
    let weight = constant("weight");
    assert_eq!(
        weight.attrs(),
        &OpAttrs::TensorLang(t::OpAttrs::ConstantAttrs {
            name: "weight".into()
        })
    );
    assert_eq!(symbol_name(&weight), None);
    let sum = reduce("sum", x, vec![0]);
    assert_eq!(reduction(&sum), Some(("sum", &[0][..], x)));
    assert_eq!(
        OpNode::new(
            t::Op::AllReduce,
            t::OpAttrs::CollectiveReduce { kind: "sum".into() },
            vec![x]
        )
        .unwrap()
        .op(),
        t::Op::AllReduce.into()
    );
}
#[test]
fn variadic_arity_comes_from_operand_declaration() {
    assert_eq!(t::Op::Concatenate.arity(), Arity::AtLeast(2));
    let x = Id::from(0);
    for count in [2, 3] {
        assert!(
            OpNode::new(
                t::Op::Concatenate,
                t::OpAttrs::ConcatenateAttrs { axis: 0 },
                vec![x; count]
            )
            .is_ok()
        );
    }
    assert!(
        OpNode::new(
            t::Op::Concatenate,
            t::OpAttrs::ConcatenateAttrs { axis: 0 },
            vec![x]
        )
        .is_err()
    );
}
#[test]
fn reduction_pattern_matches_kind_and_axes() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(symbol("x"));
    let root = graph.add(reduce("sum", x, vec![0]));
    graph.add(reduce("sum", x, vec![1]));
    graph.add(reduce("max", x, vec![0]));
    graph.rebuild();
    let mut ast = PatternAst::default();
    ast.add(ENodeOrVar::Var("?x".parse().unwrap()));
    ast.add(ENodeOrVar::ENode(reduce("sum", Id::from(0), vec![0])));
    let pattern = Pattern::new(ast);
    let matches = pattern.search(&graph);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].eclass, graph.find(root));
    assert_eq!(
        reduce("sum", x, vec![0]).discriminant(),
        t::Op::Reduce.into()
    );
}
