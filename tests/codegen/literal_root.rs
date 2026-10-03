use egg::{EGraph, Id};
use tepl_generated::ir::pattern::{OutputInference, TensorInfo};
use tepl_generated::ir::rules::literal_root::{
    rule_capture_root, rule_decimal_root, rule_integer_root,
};
use tepl_generated::ir::{DType, Op, OpAttrs, OpNode};
struct LiteralInference;
impl OutputInference for LiteralInference {
    fn infer_output(&self, _: Op, _: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
        let OpAttrs::Literal {
            dtype: Some(dtype), ..
        } = attrs
        else {
            return None;
        };
        Some(TensorInfo {
            shape: vec![],
            dtype: *dtype,
        })
    }
}
fn metadata(graph: &EGraph<OpNode, ()>, id: Id) -> Option<TensorInfo> {
    let attrs = graph[graph.find(id)].nodes.first()?.attrs();
    LiteralInference.infer_output(Op::Literal, &[], attrs)
}
#[test]
fn untyped_literal_roots_match_and_construct_each_concrete_dtype() {
    for dtype in [DType::I32, DType::U64, DType::F32] {
        let mut graph = EGraph::<OpNode, ()>::default();
        let root = graph.add(OpNode::literal("1", dtype).unwrap());
        graph.rebuild();
        let rule = rule_integer_root::build_rewrite(metadata, LiteralInference, ()).unwrap();
        let matches = rule.search(&graph);
        assert!(!rule.apply(&mut graph, &matches).is_empty());
        let rhs = graph.lookup(OpNode::literal("2", dtype).unwrap()).unwrap();
        assert_eq!(graph.find(root), graph.find(rhs));
    }
}
#[test]
fn literal_spelling_sign_and_explicit_dtype_are_preserved() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let root = graph.add(OpNode::literal("+1.00", DType::F32).unwrap());
    graph.add(OpNode::literal("1.00", DType::F32).unwrap());
    graph.rebuild();
    let rule = rule_decimal_root::build_rewrite(metadata, LiteralInference, ()).unwrap();
    let matches = rule.search(&graph);
    assert_eq!(matches.len(), 1);
    assert!(!rule.apply(&mut graph, &matches).is_empty());
    let rhs = graph
        .lookup(OpNode::literal("-2.0", DType::F32).unwrap())
        .unwrap();
    assert_eq!(graph.find(root), graph.find(rhs));
}
#[test]
fn capture_root_generates_and_runs_without_operations_or_constraints() {
    let mut graph = EGraph::<OpNode, ()>::default();
    graph.add(OpNode::literal("3", DType::F32).unwrap());
    graph.rebuild();
    let rule = rule_capture_root::build_rewrite(metadata, LiteralInference, ()).unwrap();
    let matches = rule.search(&graph);
    assert_eq!(matches.len(), 1);
    assert!(rule.apply(&mut graph, &matches).is_empty());
}
