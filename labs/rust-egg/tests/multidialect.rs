use egg::{EGraph, Id, Language};
use rust_egg::ir::dialects::{scalar as s, tensor_lang as t};
use rust_egg::ir::pattern::{OutputInference, TensorInfo};
use rust_egg::ir::rules::{lowering::rule_scalar_add, scalar::rule_commute_add};
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

struct Inference(bool);
impl OutputInference for Inference {
    fn infer_output(&self, op: Op, inputs: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
        if !self.0 || !matches!(op, Op::Scalar(s::Op::Add) | Op::TensorLang(t::Op::Add)) {
            return None;
        }
        let [lhs, rhs] = inputs else { return None };
        (lhs == rhs).then(|| lhs.clone())
    }
}
fn metadata(_: &EGraph<OpNode, ()>, _: Id) -> Option<TensorInfo> {
    Some(TensorInfo {
        shape: vec![],
        dtype: DType::I32,
    })
}

#[test]
fn dialect_names_and_discriminants_remain_distinct() {
    let x = Id::from(0);
    let tensor = OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x, x]).unwrap();
    let scalar = OpNode::new(s::Op::Add, s::OpAttrs::None, vec![x, x]).unwrap();
    assert_ne!(tensor.discriminant(), scalar.discriminant());
    assert!(!tensor.matches(&scalar));
    assert_eq!(t::Op::from_name("add"), Some(t::Op::Add));
    assert_eq!(s::Op::from_name("add"), Some(s::Op::Add));
    assert_eq!(Op::from_name("Scalar.add"), Some(s::Op::Add.into()));
    assert_eq!(Op::from_name("add"), None);
    assert!(OpNode::new(s::Op::Add, s::OpAttrs::None, vec![x]).is_err());
    // A dynamic caller still gets runtime schema validation.
    assert!(
        OpNode::from_parts(
            s::Op::Add.into(),
            vec![x, x],
            OpAttrs::TensorLang(t::OpAttrs::None)
        )
        .is_err()
    );
}

#[test]
fn cross_dialect_lowering_and_scalar_rule_share_one_graph() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(OpNode::literal("1", DType::I32).unwrap());
    let y = graph.add(OpNode::literal("2", DType::I32).unwrap());
    let root = graph.add(OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x, y]).unwrap());
    graph.rebuild();
    let scalar_rule = rule_commute_add::build_rewrite_with(metadata, Inference(true), ()).unwrap();
    assert!(scalar_rule.search(&graph).is_empty());
    let lowering = rule_scalar_add::build_rewrite_with(metadata, Inference(true), ()).unwrap();
    let matches = lowering.search(&graph);
    assert!(!lowering.apply(&mut graph, &matches).is_empty());
    graph.rebuild();
    let matches = scalar_rule.search(&graph);
    assert!(!scalar_rule.apply(&mut graph, &matches).is_empty());
    let swapped = graph
        .lookup(OpNode::new(s::Op::Add, s::OpAttrs::None, vec![y, x]).unwrap())
        .unwrap();
    assert_eq!(graph.find(root), graph.find(swapped));
    assert_eq!(lowering.name.to_string(), "lowering::scalar_add");
    assert_eq!(scalar_rule.name.to_string(), "scalar::commute_add");
}

#[test]
fn rejected_cross_dialect_inference_is_atomic() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(OpNode::literal("1", DType::I32).unwrap());
    graph.add(OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x, x]).unwrap());
    graph.rebuild();
    let before = graph.total_size();
    let rule = rule_scalar_add::build_rewrite_with(metadata, Inference(false), ()).unwrap();
    let matches = rule.search(&graph);
    assert!(rule.apply(&mut graph, &matches).is_empty());
    assert_eq!(graph.total_size(), before);
}
