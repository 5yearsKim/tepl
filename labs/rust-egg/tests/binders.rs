mod support;
use egg::{EGraph, Id, Var};
use rust_egg::host::nodes::*;
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::pattern::{AttrPattern, TensorInfo, TensorPattern, matches_at};
use rust_egg::ir::rules::binders::rule_shared_expression::Functions;
use rust_egg::ir::rules::binders::{rule_root_binding, rule_shared_expression};
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

fn dot_attrs() -> OpAttrs {
    OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        precision_config: vec![rust_egg::ir::types::Precision::Default; 2],
        algorithm: None,
    })
}

struct TestFunctions {
    allow: bool,
}

impl Functions for TestFunctions {
    fn is_reusable(&self, tensor: &TensorInfo, attrs: &OpAttrs) -> Option<bool> {
        Some(self.allow && tensor.shape.as_slice() == [2, 2] && *attrs == dot_attrs())
    }
}

fn metadata_for() -> impl Fn(&EGraph<OpNode, support::TestAnalysis>, Id) -> Option<TensorInfo> {
    move |_, _| {
        Some(TensorInfo {
            dtype: DType::F32,
            shape: vec![2, 2],
        })
    }
}

#[test]
fn nested_binder_reuses_the_matched_tensor_and_preserves_attrs() {
    let mut egraph = EGraph::<OpNode, support::TestAnalysis>::default();
    let x = egraph.add(symbol("X"));
    let w = egraph.add(symbol("W"));
    let z = egraph.add(symbol("Z"));
    let dot = egraph.add(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::DotGeneral),
            vec![x, w],
            dot_attrs(),
        )
        .unwrap(),
    );
    let product = egraph.add(binary(tensor_lang::Op::Multiply, dot, z).unwrap());
    let root = egraph.add(binary(tensor_lang::Op::Add, dot, product).unwrap());
    egraph.rebuild();

    let matched = matches_at(&egraph, root, &rule_shared_expression::pattern());
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].tensors["?c2".parse::<Var>().unwrap()], dot);
    assert_eq!(matched[0].attrs[&"d0".into()], dot_attrs());

    let rejected = support::configure_rule(
        &mut egraph,
        metadata_for(),
        binder_inference,
        rule_shared_expression::build_rewrite(TestFunctions { allow: false }),
    )
    .unwrap();
    let found = rejected.search(&egraph);
    assert!(rejected.apply(&mut egraph, &found).is_empty());

    let rule = support::configure_rule(
        &mut egraph,
        metadata_for(),
        binder_inference,
        rule_shared_expression::build_rewrite(TestFunctions { allow: true }),
    )
    .unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    egraph.rebuild();

    let reordered = egraph
        .lookup(binary(tensor_lang::Op::Add, product, dot).unwrap())
        .expect("RHS reuses the captured dot result");
    assert_eq!(egraph.find(root), egraph.find(reordered));
}

#[test]
fn root_binder_is_visible_to_search_and_rhs() {
    let mut egraph = EGraph::<OpNode, support::TestAnalysis>::default();
    let x = egraph.add(symbol("X"));
    let root = egraph.add(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::Transpose),
            vec![x],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::TransposeAttrs {
                permutation: vec![1, 0],
            }),
        )
        .unwrap(),
    );
    egraph.rebuild();

    let y = "?c1".parse::<Var>().unwrap();
    let matched = matches_at(&egraph, root, &rule_root_binding::pattern());
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].tensors[y], root);
    assert_eq!(
        matched[0].attrs[&"d0".into()],
        OpAttrs::TensorLang(tensor_lang::OpAttrs::TransposeAttrs {
            permutation: vec![1, 0],
        })
    );

    let rule = support::configure_rule(
        &mut egraph,
        support::fixture_metadata::<support::TestAnalysis>,
        support::fixture_inference,
        rule_root_binding::build_rewrite(()),
    )
    .unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(found[0].substs[0][y], root);
    assert!(rule.apply(&mut egraph, &found).is_empty());
}

#[test]
fn repeated_binder_uses_eclass_equality() {
    let mut egraph = EGraph::<OpNode, support::TestAnalysis>::default();
    let a = egraph.add(symbol("A"));
    let b = egraph.add(symbol("B"));
    let c = egraph.add(symbol("C"));
    let ab = egraph.add(binary(tensor_lang::Op::Add, a, b).unwrap());
    let ba = egraph.add(binary(tensor_lang::Op::Add, b, a).unwrap());
    egraph.union(ab, ba); // Assume commutativity has already been established.
    let equivalent = egraph.add(binary(tensor_lang::Op::Multiply, ab, ba).unwrap());
    let different = egraph.add(binary(tensor_lang::Op::Multiply, ab, c).unwrap());
    egraph.rebuild();

    let [a_var, b_var, y_var] = ["?a", "?b", "?y"].map(|s| s.parse::<Var>().unwrap());
    let lhs = TensorPattern::op(
        Op::TensorLang(tensor_lang::Op::Multiply),
        AttrPattern::Exact(OpAttrs::None),
        vec![
            TensorPattern::bind(
                y_var,
                TensorPattern::op(
                    Op::TensorLang(tensor_lang::Op::Add),
                    AttrPattern::Exact(OpAttrs::None),
                    vec![TensorPattern::Var(a_var), TensorPattern::Var(b_var)],
                ),
            ),
            TensorPattern::Var(y_var),
        ],
    );
    let found = matches_at(&egraph, equivalent, &lhs);
    assert!(!found.is_empty());
    assert!(
        found
            .iter()
            .all(|matched| egraph.find(matched.tensors[y_var]) == egraph.find(ab))
    );
    assert!(matches_at(&egraph, different, &lhs).is_empty());
}

fn binder_inference(_: Op, _: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
    Some(TensorInfo {
        shape: vec![2, 2],
        dtype: DType::F32,
    })
}
