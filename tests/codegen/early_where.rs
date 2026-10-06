mod support;
use egg::{EGraph, Id};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tepl_generated::ir::TensorInfo;
use tepl_generated::ir::dialects::guard;
use tepl_generated::ir::rules::early_where::{
    rule_early, rule_host_boundary, rule_nested_host_boundary, rule_skipped_builtin,
};
use tepl_generated::ir::{DType, Op, OpAttrs, OpNode};

fn info(shape: &[u64]) -> TensorInfo {
    TensorInfo {
        dtype: DType::I32,
        shape: shape.to_vec(),
    }
}
fn infer(_: Op, _: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
    Some(info(&[]))
}
#[derive(Clone, Default)]
struct Host(Arc<AtomicUsize>);
impl Host {
    fn allowed(&self, _: &TensorInfo) -> Option<bool> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Some(true)
    }
}
impl rule_early::HostFunctions for Host {
    fn allowed(&self, x: &TensorInfo) -> Option<bool> {
        self.allowed(x)
    }
}
impl rule_host_boundary::HostFunctions for Host {
    fn allowed(&self, x: &TensorInfo) -> Option<bool> {
        self.allowed(x)
    }
}
impl rule_nested_host_boundary::HostFunctions for Host {
    fn allowed(&self, _: &TensorInfo) -> Option<bool> {
        panic!("short-circuited host")
    }
}

fn pair_graph() -> (EGraph<OpNode, support::TestAnalysis>, Id, Id, Id) {
    let mut graph = EGraph::<OpNode, support::TestAnalysis>::default();
    let x = graph.add(OpNode::input("x"));
    let y = graph.add(OpNode::input("y"));
    let root = graph.add(OpNode::new(guard::Op::Pair, guard::OpAttrs::None, vec![x, y]).unwrap());
    graph.rebuild();
    (graph, x, y, root)
}
fn copy_graph() -> (EGraph<OpNode, support::TestAnalysis>, Id, Id) {
    let mut graph = EGraph::<OpNode, support::TestAnalysis>::default();
    let x = graph.add(OpNode::input("x"));
    let root = graph.add(OpNode::new(guard::Op::Copy, guard::OpAttrs::None, vec![x]).unwrap());
    graph.rebuild();
    (graph, x, root)
}

#[test]
fn generated_dimension_condition_prunes_before_visiting_the_next_child() {
    for shape in [vec![8], vec![1, 8], vec![1, 1, 1, 2]] {
        let (mut graph, x, y, root) = pair_graph();
        let reads = Arc::new(Mutex::new(Vec::new()));
        let read_log = reads.clone();
        let host = Host::default();
        let rule = support::configure_rule(
            &mut graph,
            move |_: &EGraph<OpNode, support::TestAnalysis>, id: Id| {
                assert_ne!(id, y, "rejected branch visited its next child");
                read_log.lock().unwrap().push(id);
                Some(info(&shape))
            },
            infer,
            rule_early::build(host.clone()),
        )
        .unwrap();
        assert!(rule.searcher.search_eclass(&graph, root).is_none());
        assert_eq!(*reads.lock().unwrap(), [x]);
        assert_eq!(host.0.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn generated_condition_waits_for_the_later_dimension_and_hosts_run_only_on_apply() {
    for size in [3, 8] {
        let (mut graph, x, y, root) = pair_graph();
        let host = Host::default();
        let rule = support::configure_rule(
            &mut graph,
            move |_: &EGraph<OpNode, support::TestAnalysis>, id: Id| {
                Some(info(if id == y {
                    if size == 3 { &[1, 3] } else { &[1, 8] }
                } else {
                    &[1, 2]
                }))
            },
            infer,
            rule_early::build(host.clone()),
        )
        .unwrap();
        let found = rule.searcher.search_eclass(&graph, root);
        assert_eq!(found.is_some(), size == 3);
        assert_eq!(host.0.load(Ordering::Relaxed), 0);
        if let Some(found) = found {
            assert!(!rule.apply(&mut graph, &[found]).is_empty());
            assert_eq!(host.0.load(Ordering::Relaxed), 1);
            assert_eq!(graph.find(x), graph.find(root));
        }
    }
}

#[test]
fn first_host_condition_keeps_later_builtin_conditions_at_application_time() {
    let (mut graph, _, root) = copy_graph();
    let host = Host::default();
    let rule = support::configure_rule(
        &mut graph,
        |_: &EGraph<OpNode, support::TestAnalysis>, _: Id| Some(info(&[8])),
        infer,
        rule_host_boundary::build(host.clone()),
    )
    .unwrap();
    let found = rule.searcher.search_eclass(&graph, root).unwrap();
    assert_eq!(host.0.load(Ordering::Relaxed), 0);
    assert!(rule.apply(&mut graph, &[found]).is_empty());
    assert_eq!(host.0.load(Ordering::Relaxed), 1);

    let rule = support::configure_rule(
        &mut graph,
        |_: &EGraph<OpNode, support::TestAnalysis>, _: Id| Some(info(&[8])),
        infer,
        rule_nested_host_boundary::build(Host::default()),
    )
    .unwrap();
    let found = rule.searcher.search_eclass(&graph, root).unwrap();
    assert!(rule.apply(&mut graph, &[found]).is_empty());
}

#[test]
fn early_short_circuit_skips_failed_builtins_and_unneeded_capture_metadata() {
    let (mut graph, x, root) = copy_graph();
    let rule = support::configure_rule(
        &mut graph,
        move |_: &EGraph<OpNode, support::TestAnalysis>, id: Id| {
            assert_ne!(id, x, "unconstrained capture metadata is unnecessary");
            Some(info(&[]))
        },
        infer,
        rule_skipped_builtin::build(()),
    )
    .unwrap();
    let found = rule.searcher.search_eclass(&graph, root).unwrap();
    assert!(!rule.apply(&mut graph, &[found]).is_empty());
    assert!(
        graph
            .lookup(OpNode::literal("1", DType::I32).unwrap())
            .is_some()
    );
}
