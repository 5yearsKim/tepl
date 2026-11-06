use rust_egg::ir;
#[test]
fn composed_transposes_are_equivalent_to_the_input() {
    let built = ir::graphs::transpose::graph_example::build().unwrap();
    let x = built.bindings["X"];
    let second = built.bindings["second"];
    let rules = ir::rules::all().unwrap();
    let result =
        tepl_runtime::runner::run(tepl_runtime::runner::runner(built.graph).unwrap(), &rules)
            .unwrap();
    assert_eq!(result.egraph.find(x), result.egraph.find(second));
    assert_eq!(
        result.egraph[built.root].data,
        ir::TensorFacts::known(vec![2, 3], ir::DType::F32)
    );
}
