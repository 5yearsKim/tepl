//! Behavior checks for the concrete expansion of `examples/inherited.tepl`.

use egg::{EGraph, Id, Rewrite};
use rust_egg::ir::patterns::{TensorInfo, TensorMetadata};
use rust_egg::ir::rules::inherited::{
    rule_associate_add_right, rule_associate_mul_right, rule_associate_small_vectors,
    rule_commute_add, rule_commute_mul, rule_commute_small_vectors, rule_distribute_mul_over_add,
};
use rust_egg::ir::{DType, OpKind, TensorLang};

fn inputs() -> (EGraph<TensorLang, ()>, [Id; 3]) {
    let mut egraph = EGraph::default();
    let inputs = ["X", "Y", "Z"].map(|name| egraph.add(TensorLang::symbol(name)));
    (egraph, inputs)
}

fn metadata(shapes: [Option<Vec<usize>>; 3]) -> impl TensorMetadata<()> {
    move |egraph: &EGraph<TensorLang, ()>, id: Id| {
        // All symbol witnesses must agree; absent or conflicting metadata
        // cannot establish a tensor description for this e-class.
        let mut result = None;
        for node in &egraph[egraph.find(id)].nodes {
            let index = match node.symbol_name() {
                Some("X") => 0,
                Some("Y") => 1,
                Some("Z") => 2,
                _ => return None,
            };
            let info = TensorInfo {
                dtype: DType::F32,
                shape: shapes[index].clone()?,
            };
            if result.as_ref().is_some_and(|previous| previous != &info) {
                return None;
            }
            result = Some(info);
        }
        result
    }
}

struct Host {
    allowed: Option<bool>,
    expected: [Vec<usize>; 3],
}

impl Host {
    fn check(&self, x: &TensorInfo, y: &TensorInfo, z: &TensorInfo) -> Option<bool> {
        assert_eq!([&x.shape, &y.shape, &z.shape], self.expected.each_ref());
        self.allowed
    }
}

impl rule_associate_add_right::Functions for Host {
    fn can_reassociate_add(&self, x: &TensorInfo, y: &TensorInfo, z: &TensorInfo) -> Option<bool> {
        self.check(x, y, z)
    }
}

impl rule_associate_mul_right::Functions for Host {
    fn can_reassociate_mul(&self, x: &TensorInfo, y: &TensorInfo, z: &TensorInfo) -> Option<bool> {
        self.check(x, y, z)
    }
}

impl rule_associate_small_vectors::Functions for Host {
    fn can_reassociate_add(&self, x: &TensorInfo, y: &TensorInfo, z: &TensorInfo) -> Option<bool> {
        self.check(x, y, z)
    }
}

impl rule_distribute_mul_over_add::Functions for Host {
    fn can_distribute_mul_over_add(
        &self,
        x: &TensorInfo,
        y: &TensorInfo,
        z: &TensorInfo,
    ) -> Option<bool> {
        self.check(x, y, z)
    }
}

fn apply(egraph: &mut EGraph<TensorLang, ()>, rule: &Rewrite<TensorLang, ()>, accepted: bool) {
    egraph.rebuild();
    let before = egraph.total_size();
    let found = rule.search(egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    let changed = rule.apply(egraph, &found);
    if accepted {
        assert_eq!(changed.len(), 1);
    } else {
        assert!(changed.is_empty());
        assert_eq!(egraph.total_size(), before, "rejection inserted RHS nodes");
    }
    egraph.rebuild();
}

#[test]
fn commutativity_instances_swap_their_specialized_operation() {
    for (op, rule) in [
        (
            OpKind::Add,
            rule_commute_add::build_rewrite::<()>().unwrap(),
        ),
        (
            OpKind::Multiply,
            rule_commute_mul::build_rewrite::<()>().unwrap(),
        ),
    ] {
        let (mut egraph, [x, y, _]) = inputs();
        let root = egraph.add(TensorLang::binary(op, x, y).unwrap());
        let other_op = if op == OpKind::Add {
            OpKind::Multiply
        } else {
            OpKind::Add
        };
        egraph.add(TensorLang::binary(other_op, x, y).unwrap());
        apply(&mut egraph, &rule, true);
        let swapped = egraph
            .lookup(TensorLang::binary(op, y, x).unwrap())
            .unwrap();
        assert_eq!(egraph.find(root), egraph.find(swapped));
        assert!(
            egraph
                .lookup(TensorLang::binary(other_op, y, x).unwrap())
                .is_none()
        );
    }
}

#[test]
fn vector_commutativity_enforces_rank_shared_dimension_and_limit() {
    for (shapes, accepted) in [
        ([Some(vec![0]), Some(vec![0]), None], true),
        ([Some(vec![1024]), Some(vec![1024]), None], true),
        ([Some(vec![1025]), Some(vec![1025]), None], false),
        ([Some(vec![4]), Some(vec![5]), None], false),
        ([Some(vec![]), Some(vec![]), None], false),
        ([Some(vec![2, 4]), Some(vec![2, 4]), None], false),
        ([Some(vec![4]), None, None], false),
    ] {
        let (mut egraph, [x, y, _]) = inputs();
        let root = egraph.add(TensorLang::binary(OpKind::Add, x, y).unwrap());
        let rule = rule_commute_small_vectors::build_rewrite(metadata(shapes)).unwrap();
        apply(&mut egraph, &rule, accepted);
        let swapped = egraph.lookup(TensorLang::binary(OpKind::Add, y, x).unwrap());
        if accepted {
            assert_eq!(egraph.find(root), egraph.find(swapped.unwrap()));
        } else {
            assert!(swapped.is_none());
        }
    }
}

#[test]
fn association_instances_require_their_bound_host_function() {
    for op in [OpKind::Add, OpKind::Multiply] {
        for allowed in [Some(true), Some(false), None] {
            for missing_metadata in [false, true] {
                let (mut egraph, [x, y, z]) = inputs();
                let xy = egraph.add(TensorLang::binary(op, x, y).unwrap());
                let root = egraph.add(TensorLang::binary(op, xy, z).unwrap());
                let expected = [vec![2, 3], vec![3], vec![]];
                let mut shapes = expected.clone().map(Some);
                if missing_metadata {
                    shapes[2] = None;
                }
                let host = Host { allowed, expected };
                let rule = if op == OpKind::Add {
                    rule_associate_add_right::build_rewrite(metadata(shapes), host).unwrap()
                } else {
                    rule_associate_mul_right::build_rewrite(metadata(shapes), host).unwrap()
                };
                let accepted = allowed == Some(true) && !missing_metadata;
                apply(&mut egraph, &rule, accepted);
                let yz = egraph.lookup(TensorLang::binary(op, y, z).unwrap());
                if accepted {
                    let rhs = egraph
                        .lookup(TensorLang::binary(op, x, yz.unwrap()).unwrap())
                        .unwrap();
                    assert_eq!(egraph.find(root), egraph.find(rhs));
                } else {
                    assert!(yz.is_none());
                }
            }
        }
    }
}

#[test]
fn vector_association_keeps_both_inherited_and_child_checks() {
    for expected in [
        [vec![0], vec![0], vec![0]],
        [vec![1024], vec![1024], vec![1024]],
        [vec![1025], vec![1025], vec![1025]],
        [vec![4], vec![5], vec![4]],
        [vec![4], vec![4], vec![5]],
        [vec![], vec![], vec![]],
        [vec![2, 4], vec![2, 4], vec![2, 4]],
    ] {
        for allowed in [Some(true), Some(false), None] {
            for missing_metadata in [false, true] {
                let (mut egraph, [x, y, z]) = inputs();
                let xy = egraph.add(TensorLang::binary(OpKind::Add, x, y).unwrap());
                let root = egraph.add(TensorLang::binary(OpKind::Add, xy, z).unwrap());
                let mut shapes = expected.clone().map(Some);
                if missing_metadata {
                    shapes[2] = None;
                }
                let accepted = allowed == Some(true)
                    && !missing_metadata
                    && expected[0].len() == 1
                    && expected[0] == expected[1]
                    && expected[0] == expected[2]
                    && expected[0][0] <= 1024;
                let rule = rule_associate_small_vectors::build_rewrite(
                    metadata(shapes),
                    Host {
                        allowed,
                        expected: expected.clone(),
                    },
                )
                .unwrap();
                apply(&mut egraph, &rule, accepted);
                let yz = egraph.lookup(TensorLang::binary(OpKind::Add, y, z).unwrap());
                if accepted {
                    let rhs = egraph
                        .lookup(TensorLang::binary(OpKind::Add, x, yz.unwrap()).unwrap())
                        .unwrap();
                    assert_eq!(egraph.find(root), egraph.find(rhs));
                } else {
                    assert!(yz.is_none());
                }
            }
        }
    }
}

#[test]
fn distribution_reuses_x_and_requires_the_bound_host_function() {
    for allowed in [Some(true), Some(false), None] {
        for missing_metadata in [false, true] {
            let (mut egraph, [x, y, z]) = inputs();
            let yz = egraph.add(TensorLang::binary(OpKind::Add, y, z).unwrap());
            let root = egraph.add(TensorLang::binary(OpKind::Multiply, x, yz).unwrap());
            let expected = [vec![2, 3], vec![3], vec![]];
            let mut shapes = expected.clone().map(Some);
            if missing_metadata {
                shapes[2] = None;
            }
            let rule = rule_distribute_mul_over_add::build_rewrite(
                metadata(shapes),
                Host { allowed, expected },
            )
            .unwrap();
            let accepted = allowed == Some(true) && !missing_metadata;
            apply(&mut egraph, &rule, accepted);
            let xy = egraph.lookup(TensorLang::binary(OpKind::Multiply, x, y).unwrap());
            let xz = egraph.lookup(TensorLang::binary(OpKind::Multiply, x, z).unwrap());
            if accepted {
                let rhs = egraph
                    .lookup(TensorLang::binary(OpKind::Add, xy.unwrap(), xz.unwrap()).unwrap())
                    .unwrap();
                assert_eq!(egraph.find(root), egraph.find(rhs));
            } else {
                assert!(xy.is_none());
                assert!(xz.is_none());
            }
        }
    }
}
