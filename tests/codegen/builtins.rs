use egg::{EGraph, Id, Rewrite};
use std::sync::{Arc, Mutex};
use tepl_generated::ir::analysis::{TensorAnalysis, TensorBindingTable, TensorInfo};
use tepl_generated::ir::dialects::toolkit;
use tepl_generated::ir::rules::builtins::*;
use tepl_generated::ir::{DType, OpAttrs, OpNode};

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
fn every_rule_builtin_executes_without_host_implementations() {
    let (mut graph, x, root) = graph(&[2, 3]);
    assert!(apply(
        &mut graph,
        rule_utilities::build_rewrite(()).unwrap()
    ));
    assert_eq!(graph.find(x), graph.find(root));
}

#[derive(Clone, Default)]
struct Host {
    calls: Arc<Mutex<Vec<&'static str>>>,
}
impl Host {
    fn record(&self, name: &'static str) {
        self.calls.lock().unwrap().push(name);
    }
    fn infer_attrs(&self, axes: &[u64]) -> Option<OpAttrs> {
        self.record("attrs");
        Some(
            toolkit::OpAttrs::TagAttrs {
                axes: axes.to_vec(),
            }
            .into(),
        )
    }
}
impl rule_derive_builtin::Functions for Host {
    fn infer_attrs(&self, axes: &[u64]) -> Option<OpAttrs> {
        self.infer_attrs(axes)
    }
}
impl rule_failed_derive::Functions for Host {
    fn infer_attrs(&self, axes: &[u64]) -> Option<OpAttrs> {
        self.infer_attrs(axes)
    }
}
impl rule_nested_hosts::Functions for Host {
    fn axes(&self, _: &TensorInfo) -> Option<Vec<u64>> {
        self.record("axes");
        Some(vec![1, 2])
    }
    fn axis(&self, _: &TensorInfo) -> Option<u64> {
        self.record("axis");
        Some(1)
    }
}
impl rule_short_circuit::Functions for Host {
    fn axes(&self, _: &TensorInfo) -> Option<Vec<u64>> {
        panic!("short-circuited host call");
    }
    fn legal(&self, _: u64) -> Option<bool> {
        panic!("short-circuited host call");
    }
}
#[test]
fn builtins_compose_with_derivations_and_nested_hosts_in_evaluation_order() {
    let (mut g, x, root) = graph(&[2, 3]);
    let host = Host::default();
    assert!(apply(
        &mut g,
        rule_derive_builtin::build_rewrite(host.clone()).unwrap()
    ));
    let tag = g
        .lookup(
            OpNode::new(
                toolkit::Op::Tag,
                toolkit::OpAttrs::TagAttrs { axes: vec![0, 0] },
                vec![x],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(g.find(root), g.find(tag));
    assert_eq!(*host.calls.lock().unwrap(), ["attrs"]);

    let (mut g, _, _) = graph(&[2, 3]);
    let host = Host::default();
    assert!(apply(
        &mut g,
        rule_nested_hosts::build_rewrite(host.clone()).unwrap()
    ));
    assert_eq!(
        *host.calls.lock().unwrap(),
        ["axes", "axes", "axes", "axis"]
    );
}
#[test]
fn short_circuit_skips_failed_builtins_and_host_calls() {
    let (mut graph, _, _) = graph(&[2, 3]);
    assert!(apply(
        &mut graph,
        rule_short_circuit::build_rewrite(Host::default()).unwrap()
    ));
}
#[test]
fn failures_reject_candidates_before_any_insertion_or_following_host_call() {
    let cases = [
        (
            vec![u64::MAX, 1],
            rule_failed_sum::build_rewrite(()).unwrap(),
        ),
        (
            vec![u64::MAX, 2],
            rule_failed_product::build_rewrite(()).unwrap(),
        ),
        (vec![2, 3], rule_failed_gather::build_rewrite(()).unwrap()),
        (vec![2, 3], rule_failed_slice::build_rewrite(()).unwrap()),
        (vec![2, 3], rule_failed_replace::build_rewrite(()).unwrap()),
        (vec![2, 3], rule_failed_divisor::build_rewrite(()).unwrap()),
        (vec![2, 3], rule_failed_range::build_rewrite(()).unwrap()),
        (
            vec![2, 3],
            rule_failed_broadcast::build_rewrite(()).unwrap(),
        ),
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
    let (mut graph, _, _) = graph(&[2, 3]);
    let host = Host::default();
    let before = graph.total_number_of_nodes();
    assert!(!apply(
        &mut graph,
        rule_failed_derive::build_rewrite(host.clone()).unwrap()
    ));
    assert_eq!(graph.total_number_of_nodes(), before);
    assert!(host.calls.lock().unwrap().is_empty());
}
