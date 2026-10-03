//! Behavior checks for the concrete expansion of `examples/rules/inherited.tepl`.

mod support;
use egg::{EGraph, Id, Rewrite};
use rust_egg::host::nodes::*;
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::pattern::{TensorInfo, TensorMetadata};
use rust_egg::ir::rules::inherited::{
    rule_associate_add_right, rule_associate_mul_right, rule_associate_small_vectors,
    rule_commute_add, rule_commute_mul, rule_commute_small_vectors, rule_distribute_mul_over_add,
};
use rust_egg::ir::{DType, OpNode};

fn inputs() -> (EGraph<OpNode, ()>, [Id; 3]) {
    let mut egraph = EGraph::default();
    let inputs = ["X", "Y", "Z"].map(|name| egraph.add(symbol(name)));
    (egraph, inputs)
}

fn metadata(shapes: [Option<Vec<u64>>; 3]) -> impl TensorMetadata<()> {
    move |graph: &EGraph<OpNode, ()>, id: Id| {
        rust_egg::host::infer_eclass(graph, id, &|node| {
            let index = match symbol_name(node)? {
                "X" => 0,
                "Y" => 1,
                "Z" => 2,
                _ => return None,
            };
            Some(TensorInfo {
                dtype: DType::F32,
                shape: shapes[index].clone()?,
            })
        })
    }
}

fn apply(egraph: &mut EGraph<OpNode, ()>, rule: &Rewrite<OpNode, ()>, accepted: bool) {
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
            tensor_lang::Op::Add,
            rule_commute_add::build_rewrite(
                support::fixture_metadata,
                support::fixture_inference,
                (),
            )
            .unwrap(),
        ),
        (
            tensor_lang::Op::Multiply,
            rule_commute_mul::build_rewrite(
                support::fixture_metadata,
                support::fixture_inference,
                (),
            )
            .unwrap(),
        ),
    ] {
        let (mut egraph, [x, y, _]) = inputs();
        let root = egraph.add(binary(op, x, y).unwrap());
        let other_op = if op == tensor_lang::Op::Add {
            tensor_lang::Op::Multiply
        } else {
            tensor_lang::Op::Add
        };
        egraph.add(binary(other_op, x, y).unwrap());
        apply(&mut egraph, &rule, true);
        let swapped = egraph.lookup(binary(op, y, x).unwrap()).unwrap();
        assert_eq!(egraph.find(root), egraph.find(swapped));
        assert!(egraph.lookup(binary(other_op, y, x).unwrap()).is_none());
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
        let root = egraph.add(binary(tensor_lang::Op::Add, x, y).unwrap());
        let rule = rule_commute_small_vectors::build_rewrite(
            metadata(shapes),
            rust_egg::host::infer_tensor_output,
            (),
        )
        .unwrap();
        apply(&mut egraph, &rule, accepted);
        let swapped = egraph.lookup(binary(tensor_lang::Op::Add, y, x).unwrap());
        if accepted {
            assert_eq!(egraph.find(root), egraph.find(swapped.unwrap()));
        } else {
            assert!(swapped.is_none());
        }
    }
}

#[test]
fn association_instances_require_compatible_metadata() {
    for op in [tensor_lang::Op::Add, tensor_lang::Op::Multiply] {
        for missing_metadata in [false, true] {
            let (mut egraph, [x, y, z]) = inputs();
            let xy = egraph.add(binary(op, x, y).unwrap());
            let root = egraph.add(binary(op, xy, z).unwrap());
            let expected = [vec![2, 3], vec![2, 3], vec![2, 3]];
            let mut shapes = expected.clone().map(Some);
            if missing_metadata {
                shapes[2] = None;
            }
            let rule = if op == tensor_lang::Op::Add {
                rule_associate_add_right::build_rewrite(
                    metadata(shapes),
                    rust_egg::host::infer_tensor_output,
                    (),
                )
                .unwrap()
            } else {
                rule_associate_mul_right::build_rewrite(
                    metadata(shapes),
                    rust_egg::host::infer_tensor_output,
                    (),
                )
                .unwrap()
            };
            let accepted = !missing_metadata;
            apply(&mut egraph, &rule, accepted);
            let yz = egraph.lookup(binary(op, y, z).unwrap());
            if accepted {
                let rhs = egraph.lookup(binary(op, x, yz.unwrap()).unwrap()).unwrap();
                assert_eq!(egraph.find(root), egraph.find(rhs));
            } else {
                assert!(yz.is_none());
            }
        }
    }
}

#[test]
fn vector_association_enforces_child_restrictions() {
    for expected in [
        [vec![0], vec![0], vec![0]],
        [vec![1024], vec![1024], vec![1024]],
        [vec![1025], vec![1025], vec![1025]],
        [vec![4], vec![5], vec![4]],
        [vec![4], vec![4], vec![5]],
        [vec![], vec![], vec![]],
        [vec![2, 4], vec![2, 4], vec![2, 4]],
    ] {
        for missing_metadata in [false, true] {
            let (mut egraph, [x, y, z]) = inputs();
            let xy = egraph.add(binary(tensor_lang::Op::Add, x, y).unwrap());
            let root = egraph.add(binary(tensor_lang::Op::Add, xy, z).unwrap());
            let mut shapes = expected.clone().map(Some);
            if missing_metadata {
                shapes[2] = None;
            }
            let accepted = !missing_metadata
                && expected[0].len() == 1
                && expected[0] == expected[1]
                && expected[0] == expected[2]
                && expected[0][0] <= 1024;
            let rule = rule_associate_small_vectors::build_rewrite(
                metadata(shapes),
                rust_egg::host::infer_tensor_output,
                (),
            )
            .unwrap();
            apply(&mut egraph, &rule, accepted);
            let yz = egraph.lookup(binary(tensor_lang::Op::Add, y, z).unwrap());
            if accepted {
                let rhs = egraph
                    .lookup(binary(tensor_lang::Op::Add, x, yz.unwrap()).unwrap())
                    .unwrap();
                assert_eq!(egraph.find(root), egraph.find(rhs));
            } else {
                assert!(yz.is_none());
            }
        }
    }
}

#[test]
fn distribution_reuses_x_with_compatible_metadata() {
    for missing_metadata in [false, true] {
        let (mut egraph, [x, y, z]) = inputs();
        let yz = egraph.add(binary(tensor_lang::Op::Add, y, z).unwrap());
        let root = egraph.add(binary(tensor_lang::Op::Multiply, x, yz).unwrap());
        let expected = [vec![2, 3], vec![2, 3], vec![2, 3]];
        let mut shapes = expected.clone().map(Some);
        if missing_metadata {
            shapes[2] = None;
        }
        let rule = rule_distribute_mul_over_add::build_rewrite(
            metadata(shapes),
            rust_egg::host::infer_tensor_output,
            (),
        )
        .unwrap();
        let accepted = !missing_metadata;
        apply(&mut egraph, &rule, accepted);
        let xy = egraph.lookup(binary(tensor_lang::Op::Multiply, x, y).unwrap());
        let xz = egraph.lookup(binary(tensor_lang::Op::Multiply, x, z).unwrap());
        if accepted {
            let rhs = egraph
                .lookup(binary(tensor_lang::Op::Add, xy.unwrap(), xz.unwrap()).unwrap())
                .unwrap();
            assert_eq!(egraph.find(root), egraph.find(rhs));
        } else {
            assert!(xy.is_none());
            assert!(xz.is_none());
        }
    }
}
