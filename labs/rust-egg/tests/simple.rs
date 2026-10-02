use egg::EGraph;
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::rules::simple::rule_commute_add;
use rust_egg::ir::{Op, OpNode};

#[test]
fn commute_add_swaps_operands() {
    let mut egraph = EGraph::<OpNode, ()>::default();
    let x = egraph.add(OpNode::symbol("X"));
    let y = egraph.add(OpNode::symbol("Y"));
    let root = egraph.add(OpNode::binary(Op::TensorLang(tensor_lang::Op::Add), x, y).unwrap());
    egraph.rebuild();

    let rule = rule_commute_add::build_rewrite::<()>().unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    egraph.rebuild();

    let swapped = egraph
        .lookup(OpNode::binary(Op::TensorLang(tensor_lang::Op::Add), y, x).unwrap())
        .expect("commuted addition was inserted");
    assert_eq!(egraph.find(root), egraph.find(swapped));
}
