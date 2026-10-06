use egg::EGraph;
use rust_egg::ir::{
    DType, TensorAnalysis,
    analysis::{TensorInfo, tensor_info},
    graphs::*,
};

fn check(definition: GraphDefinition, shape: &[u64], dtype: DType) {
    let bindings = definition.input_bindings().unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(bindings));
    let built = definition.insert_nodes(&mut graph).unwrap();
    assert_eq!(
        tensor_info(&graph, built.root),
        Some(TensorInfo {
            shape: shape.to_vec(),
            dtype
        })
    );
}
#[test]
fn all_documented_graph_examples_build_and_infer_their_outputs() {
    check(
        direct_yield::graph_transpose_example::build().unwrap(),
        &[3, 2],
        DType::F32,
    );
    check(
        bindings::graph_shared_result::build().unwrap(),
        &[3, 2],
        DType::F32,
    );
    check(
        attributes::graph_slice_and_join::build().unwrap(),
        &[8, 18],
        DType::F32,
    );
    check(
        literals::graph_add_scalar_literal::build().unwrap(),
        &[],
        DType::F32,
    );
    check(
        literals::graph_literal_output::build().unwrap(),
        &[],
        DType::I32,
    );
    let info = TensorInfo {
        shape: vec![2, 3],
        dtype: DType::F32,
    };
    let definition = untyped_inputs::graph_host_typed_add::build()
        .unwrap()
        .with_input_info("X", info.clone())
        .unwrap()
        .with_input_info("Y", info)
        .unwrap();
    check(definition, &[2, 3], DType::F32);
}
