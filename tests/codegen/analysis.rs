use egg::{EGraph, Id};
use tepl_generated::ir::analysis::{
    Inference, TensorAnalysis, TensorAnalysisData, TensorBindingTable, TensorInfo, infer_dtype,
    infer_shape, infer_tensor,
};
use tepl_generated::ir::dialects::unfamiliar::{Op, OpAttrs};
use tepl_generated::ir::rules::analysis::{rule_bad_output, rule_commute};
use tepl_generated::ir::{DType, Op as AnyOp, OpAttrs as AnyAttrs, OpNode};

fn info(shape: &[u64], dtype: DType) -> TensorInfo {
    TensorInfo {
        shape: shape.to_vec(),
        dtype,
    }
}
fn tensor(op: Op, inputs: &[TensorInfo], attrs: OpAttrs) -> Inference<TensorInfo> {
    infer_tensor(op.into(), inputs, &attrs.into())
}

#[test]
fn declarations_control_inference_independently_of_operation_names() {
    let value = info(&[2, 3], DType::I32);
    assert_eq!(
        tensor(Op::Fuse, &[value.clone(), value.clone()], OpAttrs::None),
        Inference::Known(value.clone())
    );
    for op in [Op::Add, Op::ShapeOnly, Op::DtypeOnly] {
        assert_eq!(
            tensor(op, &[value.clone()], OpAttrs::None),
            Inference::Unknown
        );
    }
    assert_eq!(
        infer_shape(Op::ShapeOnly.into(), &[&[2, 3]], &AnyAttrs::None),
        Inference::Known(vec![2, 3])
    );
    assert_eq!(
        infer_dtype(Op::DtypeOnly.into(), &[DType::Bool], &AnyAttrs::None),
        Inference::Known(DType::Bool)
    );
    assert_eq!(
        tensor(Op::Boolean, &[value.clone()], OpAttrs::None),
        Inference::Known(info(&[2, 3], DType::Bool))
    );
    assert_eq!(
        tensor(Op::Floats, &[info(&[2], DType::BF16)], OpAttrs::None),
        Inference::Known(info(&[2], DType::BF16))
    );
    assert!(matches!(
        tensor(Op::Floats, &[value.clone()], OpAttrs::None),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        tensor(Op::Fuse, &[value, info(&[2, 3], DType::F32)], OpAttrs::None),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        tensor(
            Op::Fuse,
            &[info(&[2], DType::Bool), info(&[2], DType::Bool)],
            OpAttrs::None
        ),
        Inference::Invalid(_)
    ));
    assert_eq!(
        tensor(
            Op::Tail,
            &[info(&[2, 3], DType::Bool), info(&[4, 5], DType::Bool)],
            OpAttrs::None
        ),
        Inference::Known(info(&[2, 3, 9], DType::Bool))
    );
    assert_eq!(
        tensor(Op::Tail, &[info(&[2, 3], DType::Bool)], OpAttrs::None),
        Inference::Known(info(&[2, 3], DType::Bool))
    );
    assert!(matches!(
        tensor(Op::Variadic, &[], OpAttrs::None),
        Inference::Invalid(_)
    ));
    assert_eq!(
        infer_shape(Op::Variadic.into(), &[], &AnyAttrs::None),
        Inference::Known(vec![])
    );
    assert_eq!(
        tensor(
            Op::Variadic,
            &[info(&[2, 3], DType::Bool), info(&[4], DType::Bool)],
            OpAttrs::None
        ),
        Inference::Known(info(&[5, 4], DType::Bool))
    );
}

#[test]
fn every_builtin_and_nested_metadata_type_executes_in_generated_code() {
    let attrs = OpAttrs::UtilitiesAttrs {
        flags: vec![true, true],
        nested: vec![vec![true, false]],
        padding: vec![vec![-2, 1]],
        rank: 2,
        r#match: true,
    };
    let value = info(&[u64::MAX, 0], DType::F64);
    assert_eq!(
        tensor(Op::Utilities, &[value.clone()], attrs.clone()),
        Inference::Known(value.clone())
    );
    let bad = OpAttrs::UtilitiesAttrs {
        flags: vec![false],
        nested: vec![vec![true, false]],
        padding: vec![vec![-2]],
        rank: 2,
        r#match: true,
    };
    assert!(matches!(
        tensor(Op::Utilities, &[value], bad),
        Inference::Invalid(_)
    ));
}

#[test]
fn checked_boundaries_assertion_order_and_short_circuiting_are_preserved() {
    assert_eq!(
        tensor(Op::Branches, &[info(&[2, 3], DType::F32)], OpAttrs::None),
        Inference::Known(info(&[2, 4], DType::F32))
    );
    assert!(matches!(
        tensor(Op::Overflow, &[], OpAttrs::None),
        Inference::Invalid("shape arithmetic overflow")
    ));
    assert_eq!(
        tensor(Op::Minimum, &[], OpAttrs::None),
        Inference::Known(info(&[], DType::F32))
    );
    assert!(
        matches!(tensor(Op::Stopped, &[info(&[], DType::F32)], OpAttrs::None), Inference::Invalid(reason) if reason.contains("shape assertion"))
    );
    assert!(matches!(
        tensor(
            Op::Output,
            &[info(&[], DType::F32)],
            OpAttrs::OutputAttrs { shape: vec![-1] }
        ),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        infer_shape(Op::Fuse.into(), &[&[2]], &AnyAttrs::None),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        infer_dtype(
            Op::Fuse.into(),
            &[DType::F32; 2],
            &OpAttrs::OutputAttrs { shape: vec![] }.into()
        ),
        Inference::Invalid(_)
    ));
    assert_eq!(
        infer_tensor(
            AnyOp::Literal,
            &[],
            OpNode::literal("1", DType::I8).unwrap().attrs()
        ),
        Inference::Known(info(&[], DType::I8))
    );
}

fn graph(left: TensorInfo, right: TensorInfo) -> (EGraph<OpNode, TensorAnalysis>, [Id; 3]) {
    let mut bindings = TensorBindingTable::default();
    bindings.register_symbol("X", left).unwrap();
    bindings.register_symbol("Y", right).unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(bindings));
    let x = graph.add(OpNode::input("X"));
    let y = graph.add(OpNode::input("Y"));
    let root = graph.add(OpNode::new(Op::Fuse, OpAttrs::None, vec![x, y]).unwrap());
    graph.rebuild();
    (graph, [x, y, root])
}

#[test]
fn default_rewrites_use_analysis_for_shape_constraints_and_rhs_validation() {
    for (shape, dtype, accepted) in [
        (vec![2], DType::F32, true),
        (vec![2, 3], DType::F32, false),
        (vec![2], DType::I32, false),
    ] {
        let (mut graph, [x, y, root]) = graph(info(&shape, dtype), info(&shape, dtype));
        let rewrite = rule_commute::build_rewrite(()).unwrap();
        let matches = rewrite.search(&graph);
        assert_eq!(!matches.is_empty(), accepted);
        let changed = rewrite.apply(&mut graph, &matches);
        assert_eq!(!changed.is_empty(), accepted);
        if accepted {
            graph.rebuild();
            let swapped = graph
                .lookup(OpNode::new(Op::Fuse, OpAttrs::None, vec![y, x]).unwrap())
                .unwrap();
            assert_eq!(graph.find(root), graph.find(swapped));
        }
    }
    let (mut graph, [x, _, _]) = graph(info(&[2], DType::F32), info(&[2], DType::F32));
    let rewrite = rule_bad_output::build_rewrite(()).unwrap();
    let matches = rewrite.search(&graph);
    let before = graph.total_number_of_nodes();
    assert!(rewrite.apply(&mut graph, &matches).is_empty());
    assert_eq!(before, graph.total_number_of_nodes());
    assert!(
        graph
            .lookup(OpNode::new(Op::Boolean, OpAttrs::None, vec![x]).unwrap())
            .is_none()
    );
}

#[test]
fn eclass_merge_retains_conflict_and_unknown_evidence() {
    let known = TensorAnalysisData::from_inference(Inference::Known(info(&[2], DType::F32)));
    let mut facts = known.clone();
    facts.merge(TensorAnalysisData::from_inference(Inference::Unknown));
    assert!(facts.info().is_none());
    facts.merge(TensorAnalysisData::from_inference(Inference::Known(info(
        &[3],
        DType::F32,
    ))));
    assert!(facts.is_invalid());
    let (mut graph, [x, y, _]) = graph(info(&[2], DType::F32), info(&[3], DType::F32));
    graph.union(x, y);
    graph.rebuild();
    assert!(graph[x].data.is_invalid());
    assert!(graph[x].data.info().is_none());
}
