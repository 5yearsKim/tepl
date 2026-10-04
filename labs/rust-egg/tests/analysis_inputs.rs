//! Concrete input graphs with expected metadata computed by hand.
use egg::{EGraph, Id};
use rust_egg::host::dot_attrs;
use rust_egg::host::nodes::{binary, symbol, unary};
use rust_egg::ir::analysis::{TensorAnalysis, TensorBindingTable, tensor_info};
use rust_egg::ir::dialects::tensor_lang as t;
use rust_egg::ir::pattern::TensorInfo;
use rust_egg::ir::{DType, OpNode};

fn graph(inputs: &[(&str, &[u64], DType)]) -> EGraph<OpNode, TensorAnalysis> {
    let mut bindings = TensorBindingTable::default();
    for &(name, shape, dtype) in inputs {
        bindings
            .register_symbol(
                name,
                TensorInfo {
                    shape: shape.to_vec(),
                    dtype,
                },
            )
            .unwrap();
    }
    EGraph::new(TensorAnalysis::new(bindings))
}

fn assert_known(graph: &EGraph<OpNode, TensorAnalysis>, id: Id, shape: &[u64], dtype: DType) {
    assert_eq!(
        tensor_info(graph, id),
        Some(TensorInfo {
            shape: shape.to_vec(),
            dtype
        })
    );
    assert!(!graph[graph.find(id)].data.is_invalid());
    assert!(!graph[graph.find(id)].data.is_unknown());
}

#[test]
fn add_reshape_transpose_chain_has_expected_intermediate_shapes() {
    let mut graph = graph(&[("A", &[2, 3], DType::F32), ("B", &[2, 3], DType::F32)]);
    let a = graph.add(symbol("A"));
    let b = graph.add(symbol("B"));
    let sum = graph.add(binary(t::Op::Add, a, b).unwrap());
    let reshaped = graph.add(
        OpNode::new(
            t::Op::Reshape,
            t::OpAttrs::ReshapeAttrs { shape: vec![3, 2] },
            vec![sum],
        )
        .unwrap(),
    );
    let transposed = graph.add(
        OpNode::new(
            t::Op::Transpose,
            t::OpAttrs::TransposeAttrs {
                permutation: vec![1, 0],
            },
            vec![reshaped],
        )
        .unwrap(),
    );
    graph.rebuild();
    assert_known(&graph, sum, &[2, 3], DType::F32);
    assert_known(&graph, reshaped, &[3, 2], DType::F32);
    assert_known(&graph, transposed, &[2, 3], DType::F32);
}

#[test]
fn concrete_add_inputs_cover_scalars_empty_tensors_and_type_errors() {
    let cases: &[(&[u64], DType, &[u64], DType, bool)] = &[
        (&[], DType::I64, &[], DType::I64, true),
        (&[0, 3], DType::F32, &[0, 3], DType::F32, true),
        (&[2, 3], DType::F32, &[2, 4], DType::F32, false),
        (&[2, 3], DType::F32, &[2, 3], DType::I64, false),
        (&[2, 3], DType::Bool, &[2, 3], DType::Bool, false),
    ];
    for &(lhs, lhs_dtype, rhs, rhs_dtype, valid) in cases {
        let mut graph = graph(&[("A", lhs, lhs_dtype), ("B", rhs, rhs_dtype)]);
        let a = graph.add(symbol("A"));
        let b = graph.add(symbol("B"));
        let sum = graph.add(binary(t::Op::Add, a, b).unwrap());
        let parent = graph.add(unary(t::Op::Negate, sum).unwrap());
        graph.rebuild();
        for id in [sum, parent] {
            if valid {
                assert_known(&graph, id, lhs, lhs_dtype);
            } else {
                assert!(
                    graph[graph.find(id)].data.is_invalid(),
                    "{lhs:?}:{lhs_dtype:?} + {rhs:?}:{rhs_dtype:?}"
                );
                assert!(tensor_info(&graph, id).is_none());
            }
        }
    }
}

#[test]
fn batched_dot_checks_shape_but_has_no_declared_dtype_policy() {
    // [batch, rows, inner] @ [batch, inner, columns].
    for rhs_shape in [[2, 3, 5], [2, 6, 5], [1, 3, 5]] {
        let mut graph = graph(&[("A", &[2, 4, 3], DType::F32), ("B", &rhs_shape, DType::F32)]);
        let a = graph.add(symbol("A"));
        let b = graph.add(symbol("B"));
        let dot = graph
            .add(OpNode::from_parts(t::Op::DotGeneral.into(), vec![a, b], dot_attrs()).unwrap());
        graph.rebuild();
        if rhs_shape == [2, 3, 5] {
            assert!(graph[graph.find(dot)].data.is_unknown());
            assert!(tensor_info(&graph, dot).is_none());
        } else {
            assert!(graph[graph.find(dot)].data.is_invalid());
            assert!(tensor_info(&graph, dot).is_none());
        }
    }
}

#[test]
fn unknown_merge_updates_existing_parents_and_remembers_later_conflicts() {
    for reverse in [false, true] {
        let mut graph = graph(&[
            ("A", &[2, 3], DType::F32),
            ("B", &[2, 3], DType::F32),
            ("wide", &[2, 4], DType::F32),
        ]);
        let a = graph.add(symbol("A"));
        let b = graph.add(symbol("B"));
        let missing = graph.add(symbol("unregistered"));
        let wide = graph.add(symbol("wide"));
        let sum = graph.add(binary(t::Op::Add, a, b).unwrap());
        let parent = graph.add(unary(t::Op::Negate, sum).unwrap());
        graph.rebuild();
        assert_known(&graph, parent, &[2, 3], DType::F32);

        // This deliberately merges an unknown alternative into a known input.
        if reverse {
            graph.union(missing, a);
        } else {
            graph.union(a, missing);
        }
        graph.rebuild();
        for id in [a, sum, parent] {
            assert!(graph[graph.find(id)].data.is_unknown());
            assert!(tensor_info(&graph, id).is_none());
        }

        // Unknown must not erase the original [2, 3] evidence.
        graph.union(a, wide);
        graph.rebuild();
        for id in [a, sum, parent] {
            assert!(graph[graph.find(id)].data.is_invalid());
            assert!(tensor_info(&graph, id).is_none());
        }
    }
}

#[test]
fn equivalent_reshape_cycle_rebuilds_with_stable_metadata() {
    let mut graph = graph(&[("A", &[2, 3], DType::F32)]);
    let a = graph.add(symbol("A"));
    let identity = graph.add(
        OpNode::new(
            t::Op::Reshape,
            t::OpAttrs::ReshapeAttrs { shape: vec![2, 3] },
            vec![a],
        )
        .unwrap(),
    );
    graph.rebuild();
    // A and reshape(A, [2, 3]) are equivalent. Union creates a self-reference.
    graph.union(a, identity);
    graph.rebuild();
    assert_eq!(graph.find(a), graph.find(identity));
    assert_known(&graph, a, &[2, 3], DType::F32);
    let before = graph[graph.find(a)].data.clone();
    graph.rebuild();
    assert_eq!(graph[graph.find(a)].data, before);
}

#[test]
fn same_shape_different_dtype_eclasses_are_invalid() {
    let mut graph = graph(&[("A", &[4], DType::F32), ("B", &[4], DType::BF16)]);
    let a = graph.add(symbol("A"));
    let b = graph.add(symbol("B"));
    graph.union(a, b);
    graph.rebuild();
    assert!(graph[graph.find(a)].data.is_invalid());
    assert!(tensor_info(&graph, a).is_none());
}
