use egg::EGraph;
use rust_egg::ir::rules::simple;
use rust_egg::ir::{OpKind, TensorLang};

#[test]
fn commute_add_swaps_operands() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("X"));
    let y = egraph.add(TensorLang::symbol("Y"));
    let root = egraph.add(TensorLang::binary(OpKind::Add, x, y).unwrap());
    egraph.rebuild();

    let rule = simple::commute_add::build_rewrite::<()>().unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    egraph.rebuild();

    let swapped = egraph
        .lookup(TensorLang::binary(OpKind::Add, y, x).unwrap())
        .expect("commuted addition was inserted");
    assert_eq!(egraph.find(root), egraph.find(swapped));
}
