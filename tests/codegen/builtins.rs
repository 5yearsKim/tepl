use egg::{EGraph, Id, Rewrite};
use tepl_generated::ir::analysis::{TensorAnalysis, TensorBindingTable, TensorInfo};
use tepl_generated::ir::dialects::toolkit;
use tepl_generated::ir::rules::builtins::*;
use tepl_generated::ir::{DType, Inference, OpAttrs, OpNode};

fn graph(shape: &[u64]) -> (EGraph<OpNode, TensorAnalysis>, Id, Id) {
    let mut bindings = TensorBindingTable::default();
    bindings
        .register_symbol(
            "x",
            TensorInfo {
                shape: shape.to_vec(),
                dtype: DType::F32,
            },
        )
        .unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(bindings));
    let x = graph.add(OpNode::input("x"));
    let root = graph.add(OpNode::new(toolkit::Op::Copy, toolkit::OpAttrs::None, vec![x]).unwrap());
    graph.rebuild();
    (graph, x, root)
}
fn apply(
    graph: &mut EGraph<OpNode, TensorAnalysis>,
    rule: Rewrite<OpNode, TensorAnalysis>,
) -> bool {
    let matches = rule.search(graph);
    assert!(!matches.is_empty());
    !rule.apply(graph, &matches).is_empty()
}
#[test]
fn metadata_builtins_follow_the_shared_contracts() {
    let input = TensorInfo {
        shape: vec![2, 3],
        dtype: DType::F32,
    };
    assert_eq!(
        tepl_generated::ir::analysis::infer_tensor(
            toolkit::Op::Copy.into(),
            &[input.clone()],
            &OpAttrs::None,
        ),
        Inference::Known(input),
    );
}

#[test]
fn every_rule_builtin_executes_without_host_implementations() {
    let (mut graph, x, root) = graph(&[2, 3]);
    assert!(apply(&mut graph, rule_utilities::build().unwrap()));
    assert_eq!(graph.find(x), graph.find(root));
}

#[test]
fn short_circuit_skips_failed_builtins() {
    let (mut graph, _, _) = graph(&[2, 3]);
    assert!(apply(&mut graph, rule_short_circuit::build().unwrap()));
}
#[test]
fn failures_reject_candidates_before_any_insertion() {
    let cases = [
        (vec![u64::MAX, 1], rule_failed_sum::build().unwrap()),
        (vec![u64::MAX, 2], rule_failed_product::build().unwrap()),
        (vec![2, 3], rule_failed_gather::build().unwrap()),
        (vec![2, 3], rule_failed_slice::build().unwrap()),
        (vec![2, 3], rule_failed_replace::build().unwrap()),
        (vec![2, 3], rule_failed_divisor::build().unwrap()),
        (vec![2, 3], rule_failed_range::build().unwrap()),
        (vec![2, 3], rule_failed_broadcast::build().unwrap()),
    ];
    for (shape, rule) in cases {
        let (mut graph, x, root) = graph(&shape);
        let before = graph.total_number_of_nodes();
        let matches = rule.search(&graph);
        assert!(matches.is_empty());
        assert!(rule.apply(&mut graph, &matches).is_empty());
        assert_eq!(graph.total_number_of_nodes(), before);
        assert_ne!(graph.find(x), graph.find(root));
    }
}
