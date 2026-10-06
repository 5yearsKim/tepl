use egg::{EGraph, Id};
use tepl_generated::ir::TensorInfo;
use tepl_generated::ir::analysis::{TensorAnalysis, TensorBindingTable};
use tepl_generated::ir::dialects::structural as d;
use tepl_generated::ir::rules::structural::*;
use tepl_generated::ir::{DType, OpAttrs, OpNode};

struct Host;
impl rule_swap_dot::HostFunctions for Host {
    fn supports_axis(&self, axis: u64) -> Option<bool> {
        Some(axis == 0)
    }
}
impl rule_tensor_host::HostFunctions for Host {
    fn allowed(&self, _: &TensorInfo) -> Option<bool> {
        Some(true)
    }
}
impl rule_tensor_derive::HostFunctions for Host {
    fn update(&self, _: &TensorInfo, attrs: &OpAttrs) -> Option<OpAttrs> {
        Some(attrs.clone())
    }
}
fn add(lhs: Id, rhs: Id) -> OpNode {
    OpNode::new(d::Op::Add, d::OpAttrs::None, vec![lhs, rhs]).unwrap()
}
fn dot(axis: u64, lhs: Id, rhs: Id) -> OpNode {
    OpNode::new(d::Op::Dot, d::OpAttrs::DotAttrs { axis }, vec![lhs, rhs]).unwrap()
}
#[test]
fn structural_rules_run_without_tensor_information_and_preserve_attrs() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(OpNode::input("x"));
    let y = graph.add(OpNode::input("y"));
    let sum = graph.add(add(x, y));
    let product = graph.add(dot(0, x, y));
    let rejected = graph.add(dot(1, x, y));
    graph.rebuild();
    let rule = rule_swap_add::build(()).unwrap();
    let found = rule.search(&graph);
    assert!(!rule.apply(&mut graph, &found).is_empty());
    graph.rebuild();
    let swapped = graph.lookup(add(y, x)).unwrap();
    assert_eq!(graph.find(sum), graph.find(swapped));
    let rule = rule_swap_dot::build(Host).unwrap();
    assert!(rule.searcher.search_eclass(&graph, rejected).is_some());
    let found = rule.search(&graph);
    assert!(!rule.apply(&mut graph, &found).is_empty());
    graph.rebuild();
    assert_eq!(
        graph.find(product),
        graph.find(graph.lookup(dot(0, y, x)).unwrap())
    );
    assert!(graph.lookup(dot(1, y, x)).is_none());
}
#[test]
fn missing_analysis_is_a_builder_error_when_the_rule_needs_it() {
    for error in [
        rule_shaped::build::<(), _>(()).unwrap_err(),
        rule_tensor_host::build::<(), _>(Host).unwrap_err(),
        rule_tensor_derive::build::<(), _>(Host).unwrap_err(),
    ] {
        assert!(error.contains("requires tensor analysis"), "{error}");
    }
    assert!(
        rule_untyped_rhs::build::<(), _>(())
            .unwrap_err()
            .contains("untyped RHS literal")
    );
}
#[test]
fn explicit_rhs_literals_and_polymorphic_lhs_literals_work_without_analysis() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(OpNode::input("x"));
    let zero = graph.add(OpNode::literal("0", DType::I32).unwrap());
    let root = graph.add(add(x, zero));
    graph.rebuild();
    let rule = rule_typed_rhs::build(()).unwrap();
    let found = rule.search(&graph);
    assert!(!rule.apply(&mut graph, &found).is_empty());
    graph.rebuild();
    assert_eq!(
        graph.find(root),
        graph.find(graph.lookup(add(zero, x)).unwrap())
    );
    for dtype in [DType::I32, DType::F32] {
        let zero = graph.add(OpNode::literal("0", dtype).unwrap());
        let root = graph.add(add(x, zero));
        graph.rebuild();
        let rule = rule_remove_zero::build(()).unwrap();
        let found = rule.search(&graph);
        assert!(!rule.apply(&mut graph, &found).is_empty());
        graph.rebuild();
        assert_eq!(graph.find(root), graph.find(x));
    }
}
#[test]
fn unknown_checked_analysis_never_falls_back_to_structural_application() {
    let mut inputs = TensorBindingTable::default();
    for name in ["x", "y"] {
        inputs
            .register_symbol(
                name,
                TensorInfo {
                    shape: vec![3],
                    dtype: DType::I32,
                },
            )
            .unwrap();
    }
    let mut graph = EGraph::new(TensorAnalysis::new(inputs));
    let x = graph.add(OpNode::input("x"));
    let y = graph.add(OpNode::input("y"));
    let root = graph.add(add(x, y));
    graph.rebuild();
    assert!(graph[root].data.is_unknown()); // No inference blocks in this dialect.
    let rule = rule_swap_add::build(()).unwrap();
    let found = rule.search(&graph);
    assert!(!found.is_empty());
    let before = graph.total_size();
    assert!(rule.apply(&mut graph, &found).is_empty());
    assert!(graph.lookup(add(y, x)).is_none());
    assert_eq!(graph.total_size(), before);
    assert!(rule_untyped_rhs::build::<TensorAnalysis, _>(()).is_ok());
}
