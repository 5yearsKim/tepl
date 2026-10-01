use egg::{EGraph, Id, Var};
use rust_egg::ir::patterns::{AttrPattern, TensorInfo, TensorPattern, matches_at};
use rust_egg::ir::rules::binders::{self, HostFunctions};
use rust_egg::ir::{OpAttrs, OpKind, TensorLang};

fn dot_attrs() -> OpAttrs {
    OpAttrs::DotGeneral {
        lhs_contracting: vec![1],
        rhs_contracting: vec![0],
        lhs_batch: vec![],
        rhs_batch: vec![],
    }
}

struct TestFunctions {
    allow: bool,
}

impl HostFunctions for TestFunctions {
    fn reusable(&self, tensor: &TensorInfo, attrs: &OpAttrs) -> Option<bool> {
        Some(self.allow && tensor.shape.as_slice() == [2, 2] && *attrs == dot_attrs())
    }
}

fn metadata_for(dot: Id) -> impl Fn(&EGraph<TensorLang, ()>, Id) -> Option<TensorInfo> {
    move |egraph, id| {
        (egraph.find(id) == egraph.find(dot)).then_some(TensorInfo { shape: vec![2, 2] })
    }
}

#[test]
fn nested_binder_reuses_the_matched_tensor_and_preserves_attrs() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("X"));
    let w = egraph.add(TensorLang::symbol("W"));
    let z = egraph.add(TensorLang::symbol("Z"));
    let dot = egraph.add(TensorLang::new(OpKind::DotGeneral, vec![x, w], dot_attrs()).unwrap());
    let product = egraph.add(TensorLang::binary(OpKind::Multiply, dot, z).unwrap());
    let root = egraph.add(TensorLang::binary(OpKind::Add, dot, product).unwrap());
    egraph.rebuild();

    let matched = matches_at(&egraph, root, &binders::shared_expression_pattern());
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].tensors["?Y".parse::<Var>().unwrap()], dot);
    assert_eq!(matched[0].attrs[&"d".into()], dot_attrs());

    let rejected =
        binders::rule_shared_expression(metadata_for(dot), TestFunctions { allow: false }).unwrap();
    let found = rejected.search(&egraph);
    assert!(rejected.apply(&mut egraph, &found).is_empty());

    let rule =
        binders::rule_shared_expression(metadata_for(dot), TestFunctions { allow: true }).unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    egraph.rebuild();

    let reordered = egraph
        .lookup(TensorLang::binary(OpKind::Add, product, dot).unwrap())
        .expect("RHS reuses the captured dot result");
    assert_eq!(egraph.find(root), egraph.find(reordered));
}

#[test]
fn root_binder_is_visible_to_search_and_rhs() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let x = egraph.add(TensorLang::symbol("X"));
    let root = egraph.add(
        TensorLang::new(
            OpKind::Transpose,
            vec![x],
            OpAttrs::Transpose {
                permutation: vec![1, 0],
            },
        )
        .unwrap(),
    );
    egraph.rebuild();

    let y = "?Y".parse::<Var>().unwrap();
    let matched = matches_at(&egraph, root, &binders::root_binding_pattern());
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].tensors[y], root);
    assert_eq!(
        matched[0].attrs[&"t".into()],
        OpAttrs::Transpose {
            permutation: vec![1, 0],
        }
    );

    let rule = binders::rule_root_binding::<()>().unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(found[0].substs[0][y], root);
    assert!(rule.apply(&mut egraph, &found).is_empty());
}

#[test]
fn repeated_binder_uses_eclass_equality() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let a = egraph.add(TensorLang::symbol("A"));
    let b = egraph.add(TensorLang::symbol("B"));
    let c = egraph.add(TensorLang::symbol("C"));
    let ab = egraph.add(TensorLang::binary(OpKind::Add, a, b).unwrap());
    let ba = egraph.add(TensorLang::binary(OpKind::Add, b, a).unwrap());
    egraph.union(ab, ba); // Assume commutativity has already been established.
    let equivalent = egraph.add(TensorLang::binary(OpKind::Multiply, ab, ba).unwrap());
    let different = egraph.add(TensorLang::binary(OpKind::Multiply, ab, c).unwrap());
    egraph.rebuild();

    let [a_var, b_var, y_var] = ["?a", "?b", "?y"].map(|s| s.parse::<Var>().unwrap());
    let lhs = TensorPattern::op(
        OpKind::Multiply,
        AttrPattern::Exact(OpAttrs::None),
        vec![
            TensorPattern::bind(
                y_var,
                TensorPattern::op(
                    OpKind::Add,
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
