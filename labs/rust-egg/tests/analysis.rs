use egg::{EGraph, Runner};
use rust_egg::host::nodes::{binary, symbol, unary};
use rust_egg::ir::TensorInfo;
use rust_egg::ir::analysis::{
    Inference, TensorAnalysis, TensorAnalysisData, TensorBindingTable, infer_dtype, infer_shape,
    infer_tensor, infer_tensor_output, tensor_info,
};
use rust_egg::ir::dialects::tensor_lang as t;
use rust_egg::ir::rules::simple::rule_commute_add;
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

fn info(shape: &[u64]) -> TensorInfo {
    TensorInfo {
        shape: shape.to_vec(),
        dtype: DType::F32,
    }
}

fn bindings() -> TensorBindingTable {
    let mut host = TensorBindingTable::default();
    for name in ["X", "Y", "Z"] {
        host.register_symbol(name, info(&[4])).unwrap();
    }
    host
}

fn join(mut a: TensorAnalysisData, b: TensorAnalysisData) -> TensorAnalysisData {
    a.merge(b);
    a
}

#[test]
fn tensor_join_is_associative_commutative_and_idempotent() {
    let a = TensorAnalysisData::from_inference(Inference::Known(info(&[4])));
    let b = TensorAnalysisData::from_inference(Inference::Known(info(&[8])));
    let unknown = TensorAnalysisData::from_inference(Inference::Unknown);
    let invalid = TensorAnalysisData::from_inference(Inference::Invalid("invalid shape"));
    let mut states = vec![a, b, unknown, invalid];
    // Include combinations such as known evidence plus an unknown alternative.
    loop {
        let mut expanded = states.clone();
        for a in &states {
            for b in &states {
                let merged = join(a.clone(), b.clone());
                if !expanded.contains(&merged) {
                    expanded.push(merged);
                }
            }
        }
        if expanded.len() == states.len() {
            break;
        }
        states = expanded;
    }
    for a in &states {
        assert_eq!(join(a.clone(), a.clone()), *a);
        for b in &states {
            assert_eq!(join(a.clone(), b.clone()), join(b.clone(), a.clone()));
            let mut merged = a.clone();
            let changed = merged.merge(b.clone());
            assert_eq!(changed.0, merged != *a);
            assert_eq!(changed.1, merged != *b);
            for c in &states {
                assert_eq!(
                    join(join(a.clone(), b.clone()), c.clone()),
                    join(a.clone(), join(b.clone(), c.clone())),
                );
            }
        }
    }
}

#[test]
fn unknown_alternatives_block_metadata_without_hiding_conflicts() {
    let a = TensorAnalysisData::from_inference(Inference::Known(info(&[4])));
    let b = TensorAnalysisData::from_inference(Inference::Known(info(&[8])));
    let unknown = TensorAnalysisData::from_inference(Inference::Unknown);
    let incomplete = join(a.clone(), unknown.clone());
    assert!(incomplete.is_unknown());
    assert!(incomplete.info().is_none());
    assert!(join(incomplete, b.clone()).is_invalid());
    assert!(join(join(a, b), unknown).is_invalid());
}

#[test]
fn direct_tensor_analysis_works_with_generated_rewrites() {
    let host = bindings();
    let analysis = TensorAnalysis::new(host);
    let mut graph = EGraph::new(analysis);
    let x = graph.add(symbol("X"));
    let y = graph.add(symbol("Y"));
    let root = graph.add(binary(t::Op::Add, x, y).unwrap());
    graph.rebuild();

    let rewrite = rule_commute_add::build(()).unwrap();
    let runner = Runner::<OpNode, TensorAnalysis>::new(TensorAnalysis::default())
        .with_egraph(graph)
        .with_iter_limit(4)
        .run(&[rewrite]);
    let graph = runner.egraph;
    let commuted = graph.lookup(binary(t::Op::Add, y, x).unwrap()).unwrap();
    assert_eq!(graph.find(root), graph.find(commuted));
    assert_eq!(tensor_info(&graph, root), Some(info(&[4])));
}

#[test]
fn missing_input_metadata_propagates_to_parents() {
    let mut graph = EGraph::new(TensorAnalysis::new(bindings()));
    let missing = graph.add(symbol("missing"));
    let negated = graph.add(unary(t::Op::Negate, missing).unwrap());
    graph.rebuild();
    assert!(graph[graph.find(missing)].data.is_unknown());
    assert!(graph[graph.find(negated)].data.is_unknown());
    assert!(tensor_info(&graph, negated).is_none());
}

#[test]
fn merged_operand_conflicts_propagate_to_existing_parents() {
    let mut host = bindings();
    host.register_symbol("wide", info(&[8])).unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(host));
    let x = graph.add(symbol("X"));
    let y = graph.add(symbol("Y"));
    let wide = graph.add(symbol("wide"));
    let parent = graph.add(binary(t::Op::Add, x, y).unwrap());
    let root = graph.add(unary(t::Op::Negate, parent).unwrap());
    graph.rebuild();
    assert_eq!(tensor_info(&graph, root), Some(info(&[4])));

    graph.union(x, wide);
    graph.rebuild();
    for id in [x, parent, root] {
        assert!(graph[graph.find(id)].data.is_invalid());
        assert!(tensor_info(&graph, id).is_none());
    }
}

#[test]
fn unsupported_inference_is_unknown_and_invalid_operands_propagate() {
    let mut host = bindings();
    host.register_symbol("wide", info(&[8])).unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(host));
    let x = graph.add(symbol("X"));
    let wide = graph.add(symbol("wide"));
    let unsupported = graph.add(
        OpNode::new(
            t::Op::AllGather,
            t::OpAttrs::AllGatherAttrs {
                all_gather_dim: 0,
                replica_groups: rust_egg::ir::types::ReplicaGroups::Explicit(vec![]),
                channel_id: 0,
                use_global_device_ids: false,
            },
            vec![x],
        )
        .unwrap(),
    );
    let bad_add = graph.add(binary(t::Op::Add, x, wide).unwrap());
    let parent = graph.add(unary(t::Op::Negate, bad_add).unwrap());
    let unknown_first = graph.add(binary(t::Op::Add, unsupported, bad_add).unwrap());
    let invalid_first = graph.add(binary(t::Op::Add, bad_add, unsupported).unwrap());
    graph.rebuild();
    assert!(graph[graph.find(unsupported)].data.is_unknown());
    assert!(graph[graph.find(bad_add)].data.is_invalid());
    assert!(graph[graph.find(parent)].data.is_invalid());
    assert!(graph[graph.find(unknown_first)].data.is_invalid());
    assert!(graph[graph.find(invalid_first)].data.is_invalid());
}

#[test]
fn undeclared_constant_metadata_is_unknown_in_analysis_and_rewrite_inference() {
    use rust_egg::host::nodes::constant;
    use rust_egg::ir::types::Elements;

    for element_type in ["f32", "unsupported"] {
        let node = constant(Elements {
            element_type: element_type.into(),
            shape: vec![1],
            data: 1f32.to_le_bytes().to_vec(),
        });
        let expected = None;
        assert_eq!(infer_tensor_output(node.op(), &[], node.attrs()), expected);
        let mut graph = EGraph::new(TensorAnalysis::default());
        let id = graph.add(node);
        graph.rebuild();
        assert_eq!(tensor_info(&graph, id), expected);
        assert_eq!(graph[id].data.is_unknown(), expected.is_none());
    }
}

#[test]
fn tensor_inference_preserves_unknown_and_invalid_results() {
    assert_eq!(
        infer_tensor(
            Op::TensorLang(t::Op::Exponential),
            &[info(&[4])],
            &OpAttrs::None,
        ),
        Inference::Known(info(&[4])),
    );
    assert_eq!(
        infer_tensor(Op::Input, &[], symbol("X").attrs()),
        Inference::Unknown
    );
    assert!(matches!(
        infer_tensor(
            Op::TensorLang(t::Op::Add),
            &[info(&[4]), info(&[8])],
            &OpAttrs::None,
        ),
        Inference::Invalid(_),
    ));
    // Even the graph-independent API rejects malformed calls to known evaluators.
    assert!(matches!(
        infer_tensor(Op::TensorLang(t::Op::Add), &[info(&[4])], &OpAttrs::None,),
        Inference::Invalid(_),
    ));
    let mut other_dtype = info(&[4]);
    other_dtype.dtype = DType::I64;
    assert!(matches!(
        infer_tensor(
            Op::TensorLang(t::Op::Add),
            &[info(&[4]), other_dtype],
            &OpAttrs::None,
        ),
        Inference::Invalid(_),
    ));
}

#[test]
fn dtype_rules_preserve_types_and_reject_invalid_arithmetic() {
    use rust_egg::ir::dialects::scalar;
    for dtype in DType::ALL {
        for op in [
            Op::TensorLang(t::Op::Add),
            Op::TensorLang(t::Op::Multiply),
            Op::Scalar(scalar::Op::Add),
        ] {
            let result = infer_dtype(op, &[dtype, dtype], &OpAttrs::None);
            if dtype == DType::Bool {
                assert!(matches!(result, Inference::Invalid(_)));
            } else {
                assert_eq!(result, Inference::Known(dtype));
            }
            assert!(matches!(
                infer_dtype(op, &[dtype], &OpAttrs::None),
                Inference::Invalid(_)
            ));
        }
        for op in [
            Op::TensorLang(t::Op::Negate),
            Op::Scalar(scalar::Op::Negate),
        ] {
            let result = infer_dtype(op, &[dtype], &OpAttrs::None);
            if dtype == DType::Bool {
                assert!(matches!(result, Inference::Invalid(_)));
            } else {
                assert_eq!(result, Inference::Known(dtype));
            }
        }
        for (op, attrs) in [
            (t::Op::Reshape, t::OpAttrs::ReshapeAttrs { shape: vec![4] }),
            (
                t::Op::Transpose,
                t::OpAttrs::TransposeAttrs {
                    permutation: vec![0],
                },
            ),
        ] {
            assert_eq!(
                infer_dtype(Op::TensorLang(op), &[dtype], &OpAttrs::TensorLang(attrs)),
                Inference::Known(dtype)
            );
            assert!(matches!(
                infer_dtype(Op::TensorLang(op), &[dtype], &OpAttrs::None),
                Inference::Invalid(_)
            ));
        }
        let literal = OpNode::literal("1", dtype).unwrap();
        assert_eq!(
            infer_dtype(literal.op(), &[], literal.attrs()),
            Inference::Known(dtype)
        );
    }
    assert!(matches!(
        infer_dtype(
            Op::TensorLang(t::Op::Add),
            &[DType::F32, DType::I32],
            &OpAttrs::None
        ),
        Inference::Invalid(_)
    ));
    assert_eq!(
        infer_dtype(
            Op::TensorLang(t::Op::Exponential),
            &[DType::F32],
            &OpAttrs::None
        ),
        Inference::Known(DType::F32)
    );
    assert_eq!(
        infer_dtype(Op::Input, &[], symbol("X").attrs()),
        Inference::Unknown
    );
    assert_eq!(
        infer_dtype(
            Op::Literal,
            &[],
            &OpAttrs::Literal {
                value: "1".into(),
                dtype: None
            }
        ),
        Inference::Unknown
    );
}

#[test]
fn undeclared_dot_dtype_remains_unknown_with_default_or_overridden_precision() {
    let op = Op::TensorLang(t::Op::DotGeneral);
    let mut attrs = rust_egg::host::dot_attrs();
    assert_eq!(
        infer_dtype(op, &[DType::F32, DType::F32], &attrs),
        Inference::Unknown
    );
    if let OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
        precision_config, ..
    }) = &mut attrs
    {
        *precision_config = vec![rust_egg::ir::types::Precision::High; 2];
    }
    assert_eq!(
        infer_dtype(op, &[DType::F32, DType::F32], &attrs),
        Inference::Unknown
    );
    let operands = [info(&[2, 4, 3]), info(&[2, 3, 5])];
    assert_eq!(infer_tensor(op, &operands, &attrs), Inference::Unknown);
}

#[test]
fn rhs_validation_and_eclass_analysis_share_shape_and_dtype_inference() {
    let host = bindings();
    let mut graph = EGraph::new(TensorAnalysis::new(host));
    let x = graph.add(symbol("X"));
    let attrs = OpAttrs::TensorLang(t::OpAttrs::ReshapeAttrs { shape: vec![2, 2] });
    let node = OpNode::from_parts(Op::TensorLang(t::Op::Reshape), vec![x], attrs.clone()).unwrap();
    let reshaped = graph.add(node);
    graph.rebuild();
    assert_eq!(
        tensor_info(&graph, reshaped),
        infer_tensor_output(Op::TensorLang(t::Op::Reshape), &[info(&[4])], &attrs),
    );
    assert_eq!(tensor_info(&graph, reshaped), Some(info(&[2, 2])));

    let mut mismatched = info(&[4]);
    mismatched.dtype = DType::I64;
    assert!(matches!(
        infer_tensor(
            Op::TensorLang(t::Op::Add),
            &[info(&[4]), mismatched],
            &OpAttrs::None,
        ),
        Inference::Invalid(_),
    ));
}

#[test]
fn generated_dot_shape_supports_nonadjacent_and_multiple_contracting_axes() {
    let attrs = OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
        lhs_contracting_dimensions: vec![1, 3],
        rhs_contracting_dimensions: vec![3, 0],
        lhs_batching_dimensions: vec![0],
        rhs_batching_dimensions: vec![2],
        precision_config: vec![rust_egg::ir::types::Precision::Default; 2],
        algorithm: None,
    });
    assert_eq!(
        infer_shape(
            Op::TensorLang(t::Op::DotGeneral),
            &[&[2, 3, 5, 7], &[7, 11, 2, 3]],
            &attrs,
        ),
        Inference::Known(vec![2, 5, 11]),
    );
}

#[test]
fn generated_shape_checks_reject_invalid_axes_and_checked_arithmetic_failures() {
    for permutation in [vec![0, 0], vec![0, 2], vec![0, u64::MAX]] {
        assert!(matches!(
            infer_shape(
                Op::TensorLang(t::Op::Transpose),
                &[&[2, 3]],
                &OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs { permutation }),
            ),
            Inference::Invalid(_),
        ));
    }
    assert_eq!(
        infer_shape(
            Op::TensorLang(t::Op::Transpose),
            &[&[2, 3]],
            &OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs {
                permutation: vec![1, 0]
            }),
        ),
        Inference::Known(vec![3, 2]),
    );
    assert!(matches!(
        infer_shape(
            Op::TensorLang(t::Op::Reshape),
            &[&[u64::MAX, 2]],
            &OpAttrs::TensorLang(t::OpAttrs::ReshapeAttrs { shape: vec![2] }),
        ),
        Inference::Invalid(_),
    ));
    assert_eq!(
        infer_shape(
            Op::TensorLang(t::Op::Reshape),
            &[&[u64::MAX, 2, 0]],
            &OpAttrs::TensorLang(t::OpAttrs::ReshapeAttrs { shape: vec![0] }),
        ),
        Inference::Known(vec![0]),
    );
}
