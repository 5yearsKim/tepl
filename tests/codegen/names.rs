use ::egg::{EGraph, Rewrite};
use tepl_generated::ir::{
    DType, Op, OpAttrs, OpNode,
    analysis::{Inference, TensorAnalysis, TensorBindingTable, TensorInfo, infer_shape},
    dialects::{b, egg, std, r#type},
    rules::names::{rule_match, rule_type},
};

struct Keywords;
impl rule_match::Functions for Keywords {
    fn r#type(&self, x: &TensorInfo) -> Option<bool> {
        Some(x.shape == [2, 3])
    }
    fn r#match(&self, x: &TensorInfo) -> Option<bool> {
        Some(x.dtype == DType::F32)
    }
    fn r#gen(&self, _: &TensorInfo) -> Option<bool> {
        Some(true)
    }
}

#[test]
fn crate_names_and_keyword_methods_execute_a_checked_rewrite() {
    let mut bindings = TensorBindingTable::default();
    bindings
        .register_symbol(
            "x",
            TensorInfo {
                dtype: DType::F32,
                shape: vec![2, 3],
            },
        )
        .unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(bindings));
    let x = graph.add(OpNode::input("x"));
    let inner = graph.add(OpNode::new(b::Op::BCopy, b::OpAttrs::None, vec![x]).unwrap());
    let middle = graph.add(OpNode::new(egg::Op::EggCopy, egg::OpAttrs::None, vec![inner]).unwrap());
    let root = graph.add(OpNode::new(std::Op::StdCopy, std::OpAttrs::None, vec![middle]).unwrap());
    graph.rebuild();
    let rule: Rewrite<OpNode, TensorAnalysis> = rule_match::build_rewrite(Keywords).unwrap();
    let matches = rule.search(&graph);
    assert_eq!(matches.len(), 1);
    assert!(!rule.apply(&mut graph, &matches).is_empty());
    graph.rebuild();
    assert_eq!(graph.find(root), graph.find(x));
    assert_eq!(rule.name.to_string(), "names::match");
}

#[test]
fn raw_module_and_field_tokens_preserve_tepl_identity_in_analysis() {
    let attrs = r#type::OpAttrs::Match {
        r#type: 2,
        type_: 1,
        r#match: true,
        r#gen: 2,
    };
    let attrs: OpAttrs = attrs.into();
    assert_eq!(
        infer_shape(r#type::Op::Type.into(), &[&[2, 3]], &attrs),
        Inference::Known(vec![2, 3])
    );
    assert_eq!(Op::from_name("Type.type"), Some(r#type::Op::Type.into()));
    assert!(rule_type::build_rewrite(()).is_ok());
}
