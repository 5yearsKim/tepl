use egg::{EGraph, Language};
use ir::analysis::tensor_info;
use ir::graphs::concrete_graphs::*;
use ir::{BuiltGraph, GraphDefinition, InputSpec, TensorAnalysis, TensorInfo};
use tepl_generated::ir::{self, DType, OpAttrs, OpNode};

fn info() -> TensorInfo {
    TensorInfo {
        shape: vec![2, 3],
        dtype: DType::F32,
    }
}

#[test]
fn sharing_attributes_and_typed_analysis() {
    let definition: GraphDefinition = graph_shared::build().unwrap();
    assert_eq!(definition.nodes().len(), 4);
    let input: &InputSpec = &definition.inputs()[0];
    assert_eq!(input.shape.as_deref(), Some([2, 3].as_slice()));
    assert_eq!(definition.get("N"), definition.get("shared_node"));
    let children = definition.nodes()[definition.root()].children();
    assert_eq!(children[0], children[1]);
    let attrs = definition.nodes()[definition.get("unused").unwrap()].attrs();
    let OpAttrs::GraphOps(ir::dialects::graph_ops::OpAttrs::ConfigAttrs {
        axis,
        offset,
        label,
        axes,
        note,
        flags,
        regions,
        ..
    }) = attrs
    else {
        panic!("wrong attrs");
    };
    assert_eq!(*axis, 1);
    assert_eq!(*offset, i64::MIN);
    assert_eq!(label, "a\"b\\c");
    assert_eq!(axes.as_deref(), Some([0, 1].as_slice()));
    assert_eq!(*note, None);
    assert!(flags.is_empty());
    assert!(regions.as_ref().unwrap().is_empty());
    let (graph, built) = definition.into_egraph().unwrap();
    assert_eq!(built.get("N"), built.get("shared_node"));
    assert!(built.get("unused").is_some());
    assert_eq!(tensor_info(&graph, built.root), Some(info()));
    let definition = graph_defaults::build().unwrap();
    let OpAttrs::GraphOps(ir::dialects::graph_ops::OpAttrs::ConfigAttrs {
        axis,
        offset,
        axes,
        note,
        ..
    }) = definition.nodes()[definition.root()].attrs()
    else {
        panic!();
    };
    assert_eq!(*axis, u64::MAX);
    assert_eq!(*offset, 1);
    assert_eq!(*axes, None);
    assert_eq!(note.as_deref(), Some("text"));
}

#[test]
fn inserting_remaps_ids_and_keeps_unused_nodes() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let existing = graph.add(OpNode::input("already"));
    let built: BuiltGraph = graph_structural::build()
        .unwrap()
        .insert_into(&mut graph)
        .unwrap();
    assert_ne!(built.get("X"), Some(existing));
    let children = graph[built.root].nodes[0].children();
    assert_eq!(
        children,
        &[built.get("X").unwrap(), built.get("X").unwrap()]
    );
    assert!(built.get("Y").is_some() && built.get("unused").is_some());
    assert_eq!(built.get("missing"), None);
    let count = graph.total_size();
    assert!(
        graph_shared::build()
            .unwrap()
            .insert_into(&mut graph)
            .is_err()
    );
    assert_eq!(graph.total_size(), count);
    let built = graph_input_output::build()
        .unwrap()
        .insert_into(&mut graph)
        .unwrap();
    assert_eq!(built.root, built.get("X").unwrap());
}

#[test]
fn literals_preserve_spelling_and_infer_scalar_types() {
    let definition = graph_literal::build().unwrap();
    assert!(
        matches!(definition.nodes()[0].attrs(), OpAttrs::Literal { value, dtype: Some(DType::I32) } if value == "+01")
    );
    let (graph, built) = definition.into_egraph().unwrap();
    assert_eq!(
        tensor_info(&graph, built.root),
        Some(TensorInfo {
            shape: vec![],
            dtype: DType::I32
        })
    );
    let (graph, built) = graph_untyped_literal::build()
        .unwrap()
        .into_egraph()
        .unwrap();
    assert_eq!(tensor_info(&graph, built.root), None);
}

#[test]
fn host_metadata_custom_analysis_and_invalid_graphs() {
    let definition = graph_partial::build()
        .unwrap()
        .with_input_info("X", info())
        .unwrap()
        .with_input_info("Y", info())
        .unwrap();
    let (graph, built) = definition.into_egraph_with(TensorAnalysis::new).unwrap();
    assert_eq!(tensor_info(&graph, built.root), Some(info()));
    assert!(
        graph_partial::build()
            .unwrap()
            .with_input_info(
                "X",
                TensorInfo {
                    shape: vec![9],
                    dtype: DType::F32
                }
            )
            .is_err()
    );
    assert!(
        graph_shared::build()
            .unwrap()
            .with_input_info(
                "X",
                TensorInfo {
                    shape: vec![2, 3],
                    dtype: DType::I32
                }
            )
            .is_err()
    );
    assert!(
        graph_structural::build()
            .unwrap()
            .with_input_info("missing", info())
            .is_err()
    );
    assert!(graph_invalid_shape::build().is_err());
    assert!(graph_invalid_dtype::build().is_err());
    assert!(graph_invalid_partial_shape::build().is_err());
    assert!(graph_invalid_partial_dtype::build().is_err());
    let (graph, built) = graph_structural::build().unwrap().into_egraph().unwrap();
    assert_eq!(tensor_info(&graph, built.root), None);
    assert!(
        ir::graphs::GraphDefinition::new(vec![OpNode::input("X")], vec![], Default::default(), 0)
            .is_err()
    );
    let bad = OpNode::new(
        ir::dialects::graph_ops::Op::Negate,
        ir::dialects::graph_ops::OpAttrs::None,
        vec![0usize.into()],
    )
    .unwrap();
    assert!(ir::graphs::GraphDefinition::new(vec![bad], vec![], Default::default(), 0).is_err());
}

#[test]
fn nested_keyword_source_paths_are_relocatable() {
    let definition = ir::graphs::r#type::r#match::graph_nested::build().unwrap();
    let (graph, built) = definition.into_egraph().unwrap();
    assert_eq!(
        tensor_info(&graph, built.root),
        Some(TensorInfo {
            shape: vec![],
            dtype: DType::I32
        })
    );
}
