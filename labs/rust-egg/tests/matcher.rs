mod support;
use rust_egg::host::nodes::{binary, symbol};

use rust_egg::ir::dialects::tensor_lang;

use egg::{EGraph, Language, Var};

use rust_egg::ir::pattern::{AttrPattern, AttrVar, TensorPattern};
use rust_egg::ir::pattern::{TensorExpr, matches_at, tensor_rewrite_checked};
use rust_egg::ir::{Op, OpAttrs, OpNode};

#[test]
fn sibling_alternatives_are_independent_ordered_and_can_stop_early() {
    let mut egraph = EGraph::<OpNode, ()>::default();
    let x = egraph.add(symbol("x"));
    let y = egraph.add(symbol("y"));
    let xy = egraph.add(binary(tensor_lang::Op::Add, x, y).unwrap());
    let yx = egraph.add(binary(tensor_lang::Op::Add, y, x).unwrap());
    egraph.union(xy, yx);
    let root = egraph.add(binary(tensor_lang::Op::Add, xy, xy).unwrap());
    egraph.rebuild();

    let [a, b, c, d] = ["?a", "?b", "?c", "?d"].map(|s| s.parse::<Var>().unwrap());
    let add = |children| {
        TensorPattern::op(
            Op::TensorLang(tensor_lang::Op::Add),
            AttrPattern::Exact(OpAttrs::None),
            children,
        )
    };
    let pattern = add(vec![
        add(vec![TensorPattern::Var(a), TensorPattern::Var(b)]),
        add(vec![TensorPattern::Var(c), TensorPattern::Var(d)]),
    ]);
    let bindings = |m: &rust_egg::ir::pattern::TensorMatch| [a, b, c, d].map(|var| m.tensors[var]);
    // Preserve the e-graph's node order, with the left child's alternatives
    // as the outer loop and the right child's alternatives as the inner loop.
    let nodes = &egraph[egraph.find(xy)].nodes;
    let expected: Vec<_> = nodes
        .iter()
        .flat_map(|left| {
            nodes.iter().map(move |right| {
                [
                    left.children()[0],
                    left.children()[1],
                    right.children()[0],
                    right.children()[1],
                ]
            })
        })
        .collect();
    let all = matches_at(&egraph, root, &pattern);
    assert_eq!(all.len(), 4);
    assert_eq!(all.iter().map(bindings).collect::<Vec<_>>(), expected);

    let rule = tensor_rewrite_checked(
        "limit",
        pattern,
        TensorExpr::Var(a),
        support::fixture_metadata,
        support::fixture_inference,
        |_, _| Some(Default::default()),
    )
    .unwrap();
    let limited = rule.search_with_limit(&egraph, 1);
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].substs.len(), 1);
    assert_eq!(
        [a, b, c, d].map(|var| limited[0].substs[0][var]),
        expected[0]
    );
}

#[test]
fn repeated_attribute_binding_preserves_each_witness() {
    let mut egraph = EGraph::<OpNode, ()>::default();
    let x = egraph.add(symbol("x"));
    let y = egraph.add(symbol("y"));
    let first = OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting: vec![0],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    });
    let second = OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting: vec![1],
        rhs_contracting: vec![1],
        lhs_batch: vec![],
        rhs_batch: vec![],
    });
    let first_id = egraph.add(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::DotGeneral),
            vec![x, y],
            first.clone(),
        )
        .unwrap(),
    );
    let second_id = egraph.add(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::DotGeneral),
            vec![x, y],
            second.clone(),
        )
        .unwrap(),
    );
    egraph.union(first_id, second_id);
    let root = egraph.add(binary(tensor_lang::Op::Add, first_id, first_id).unwrap());
    egraph.rebuild();

    let [x_var, y_var] = ["?x", "?y"].map(|name| name.parse::<Var>().unwrap());
    let attr_var = AttrVar::from("dot");
    let dot = || {
        TensorPattern::op(
            Op::TensorLang(tensor_lang::Op::DotGeneral),
            AttrPattern::Bind(attr_var),
            vec![TensorPattern::Var(x_var), TensorPattern::Var(y_var)],
        )
    };
    let lhs = TensorPattern::op(
        Op::TensorLang(tensor_lang::Op::Add),
        AttrPattern::Exact(OpAttrs::None),
        vec![dot(), dot()],
    );
    let found = matches_at(&egraph, root, &lhs);
    assert_eq!(found.len(), 2);
    assert!(found.iter().any(|m| m.attrs[&attr_var] == first));
    assert!(found.iter().any(|m| m.attrs[&attr_var] == second));
    assert!(found.iter().all(|m| m.tensors[x_var] == x));
}
