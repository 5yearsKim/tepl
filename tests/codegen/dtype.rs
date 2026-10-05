use egg::{EGraph, Id};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tepl_generated::ir::analysis::{
    Inference, TensorAnalysis, TensorBindingTable, infer_dtype, infer_tensor_output, tensor_info,
};
use tepl_generated::ir::dialects::d_types::{self as d, Op};
use tepl_generated::ir::pattern::TensorInfo;
use tepl_generated::ir::rules::dtype::{
    rule_host_identity, rule_remove_f32_convert as specialized,
    rule_remove_identity_convert as identity, rule_shared,
};
use tepl_generated::ir::{DType, OpAttrs, OpNode};

fn info(dtype: DType) -> TensorInfo {
    TensorInfo {
        shape: vec![4],
        dtype,
    }
}
fn conversion(x: Id, to: DType) -> OpNode {
    OpNode::from_parts(
        Op::Convert.into(),
        vec![x],
        d::OpAttrs::ConvertAttrs { to }.into(),
    )
    .unwrap()
}
fn make_graph(input: DType, to: DType) -> (EGraph<OpNode, TensorAnalysis>, Id, Id) {
    let mut inputs = TensorBindingTable::default();
    inputs.register_symbol("x", info(input)).unwrap();
    let mut graph = EGraph::new(TensorAnalysis::new(inputs));
    let x = graph.add(OpNode::input("x"));
    let root = graph.add(conversion(x, to));
    graph.rebuild();
    (graph, x, root)
}
#[test]
fn dtype_programs_validate_operands_attributes_lists_and_lazy_branches() {
    let no = OpAttrs::None;
    assert_eq!(
        infer_dtype(Op::Fixed.into(), &[], &no),
        Inference::Known(DType::F32)
    );
    assert_eq!(
        infer_dtype(Op::Unknown.into(), &[DType::F32], &no),
        Inference::Unknown
    );
    assert_eq!(
        infer_dtype(Op::Equal.into(), &[DType::I32, DType::I32], &no),
        Inference::Known(DType::Bool)
    );
    assert!(matches!(
        infer_dtype(Op::Equal.into(), &[DType::Bool, DType::Bool], &no),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        infer_dtype(Op::Equal.into(), &[DType::I32, DType::F32], &no),
        Inference::Invalid(_)
    ));
    assert_eq!(
        infer_dtype(
            Op::Select.into(),
            &[DType::Bool, DType::BF16, DType::BF16],
            &no
        ),
        Inference::Known(DType::BF16)
    );
    assert!(matches!(
        infer_dtype(
            Op::Select.into(),
            &[DType::F32, DType::BF16, DType::BF16],
            &no
        ),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        infer_dtype(Op::Variadic.into(), &[], &no),
        Inference::Invalid(_)
    ));
    assert_eq!(
        infer_dtype(Op::Variadic.into(), &[DType::U8, DType::U8], &no),
        Inference::Known(DType::U8)
    );
    assert!(matches!(
        infer_dtype(Op::Variadic.into(), &[DType::U8, DType::I8], &no),
        Inference::Invalid(_)
    ));
    assert_eq!(
        infer_dtype(Op::Tail.into(), &[DType::F16], &no),
        Inference::Known(DType::F16)
    );
    for widen in [false, true] {
        let attrs = d::OpAttrs::ChoicesAttrs {
            to: DType::F16,
            alternatives: vec![],
            widen,
        }
        .into();
        assert_eq!(
            infer_dtype(Op::Choices.into(), &[DType::BF16], &attrs),
            Inference::Known(if widen { DType::F32 } else { DType::F16 })
        );
    }
    for input in DType::ALL {
        for to in DType::ALL {
            let attrs = d::OpAttrs::ConvertAttrs { to }.into();
            let result = infer_dtype(Op::Convert.into(), &[input], &attrs);
            if input == DType::Bool || to == DType::Bool {
                assert!(matches!(result, Inference::Invalid(_)));
            } else {
                assert_eq!(result, Inference::Known(to));
            }
        }
    }
    assert!(matches!(
        infer_dtype(Op::Convert.into(), &[DType::F32], &no),
        Inference::Invalid(_)
    ));
}
#[test]
fn identity_conversion_binds_input_dtype_and_preserves_real_conversions() {
    for input in [DType::BF16, DType::F32, DType::I64] {
        for to in [DType::BF16, DType::F32, DType::I64] {
            let (mut graph, x, root) = make_graph(input, to);
            let rule = identity::build_rewrite(()).unwrap();
            let matches = rule.search(&graph);
            assert_eq!(!matches.is_empty(), input == to);
            let before = graph.total_number_of_nodes();
            rule.apply(&mut graph, &matches);
            graph.rebuild();
            assert_eq!(graph.find(x) == graph.find(root), input == to);
            assert_eq!(graph.total_number_of_nodes(), before);
        }
    }
    for t in [DType::F32, DType::BF16] {
        let (graph, _, _) = make_graph(t, t);
        assert_eq!(
            !specialized::build_rewrite(())
                .unwrap()
                .search(&graph)
                .is_empty(),
            t == DType::F32
        );
    }
}
#[test]
fn shared_dtype_variables_bind_per_branch_and_reject_mismatches() {
    let mut graph: EGraph<OpNode, ()> = EGraph::default();
    let mut facts = HashMap::new();
    for (index, (a, b, accepted)) in [
        (DType::F32, DType::F32, true),
        (DType::BF16, DType::BF16, true),
        (DType::F32, DType::BF16, false),
        (DType::I32, DType::I32, false),
    ]
    .into_iter()
    .enumerate()
    {
        let x = graph.add(OpNode::input(format!("x{index}")));
        let y = graph.add(OpNode::input(format!("y{index}")));
        facts.insert(x, info(a));
        facts.insert(y, info(b));
        let root = graph.add(OpNode::from_parts(Op::Pair.into(), vec![x, y], no_attrs()).unwrap());
        facts.insert(root, info(a));
        graph.rebuild();
        let data = facts.clone();
        let rule = rule_shared::build_rewrite_with(
            move |_: &EGraph<OpNode, ()>, id| data.get(&id).cloned(),
            infer_tensor_output,
            (),
        )
        .unwrap();
        let checks = rule.searcher.search_eclass(&graph, root);
        assert_eq!(checks.is_some(), accepted);
    }
}
fn no_attrs() -> OpAttrs {
    OpAttrs::None
}
#[test]
fn failed_candidates_discard_dtype_bindings_before_the_next_branch() {
    let mut graph: EGraph<OpNode, ()> = EGraph::default();
    let x = graph.add(OpNode::input("f32"));
    let y = graph.add(OpNode::input("bf16"));
    let bad = graph.add(OpNode::from_parts(Op::Pair.into(), vec![x, y], no_attrs()).unwrap());
    let good = graph.add(OpNode::from_parts(Op::Pair.into(), vec![y, y], no_attrs()).unwrap());
    graph.union(bad, good);
    graph.rebuild();
    let root = graph.find(good);
    let rule = rule_shared::build_rewrite_with(
        move |_: &EGraph<OpNode, ()>, id| {
            Some(info(if id == x { DType::F32 } else { DType::BF16 }))
        },
        infer_tensor_output,
        (),
    )
    .unwrap();
    let matched = rule.searcher.search_eclass(&graph, root).unwrap();
    assert_eq!(matched.substs.len(), 1);
    for name in ["?c0", "?c1"] {
        assert_eq!(matched.substs[0][name.parse().unwrap()], y);
    }
}
#[test]
fn dtype_guard_prunes_before_reading_a_later_capture() {
    let mut graph: EGraph<OpNode, ()> = EGraph::default();
    let x = graph.add(OpNode::input("x"));
    let y = graph.add(OpNode::input("y"));
    let root = graph.add(OpNode::from_parts(Op::Pair.into(), vec![x, y], no_attrs()).unwrap());
    graph.rebuild();
    let reads = Arc::new(AtomicUsize::new(0));
    let count = reads.clone();
    let rule = rule_shared::build_rewrite_with(
        move |_: &EGraph<OpNode, ()>, id| {
            count.fetch_add(1, Ordering::Relaxed);
            assert_eq!(id, x);
            Some(info(DType::I32))
        },
        infer_tensor_output,
        (),
    )
    .unwrap();
    assert!(rule.searcher.search_eclass(&graph, root).is_none());
    assert_eq!(reads.load(Ordering::Relaxed), 1);
}
#[test]
fn missing_and_conflicting_eclass_metadata_prevent_binding() {
    let (mut graph, x, root) = make_graph(DType::F32, DType::F32);
    let unknown = graph.add(OpNode::input("missing"));
    graph.union(x, unknown);
    graph.rebuild();
    assert!(tensor_info(&graph, x).is_none());
    assert!(
        identity::build_rewrite(())
            .unwrap()
            .searcher
            .search_eclass(&graph, root)
            .is_none()
    );
    let (mut graph, x, root) = make_graph(DType::F32, DType::F32);
    graph
        .analysis
        .symbols
        .register_symbol("other", info(DType::BF16))
        .unwrap();
    let other = graph.add(OpNode::input("other"));
    graph.union(x, other);
    graph.rebuild();
    assert!(graph[x].data.is_invalid());
    assert!(
        identity::build_rewrite(())
            .unwrap()
            .searcher
            .search_eclass(&graph, root)
            .is_none()
    );
}
#[test]
fn application_rebinds_dtype_and_rechecks_guards_before_mutation() {
    let mut graph: EGraph<OpNode, ()> = EGraph::default();
    let x = graph.add(OpNode::input("x"));
    let root = graph.add(conversion(x, DType::F32));
    graph.rebuild();
    let facts = Arc::new(Mutex::new(HashMap::from([
        (x, info(DType::F32)),
        (root, info(DType::F32)),
    ])));
    let reader = facts.clone();
    let rule = identity::build_rewrite_with(
        move |_: &EGraph<OpNode, ()>, id| reader.lock().unwrap().get(&id).cloned(),
        infer_tensor_output,
        (),
    )
    .unwrap();
    let matches = rule.search(&graph);
    assert!(!matches.is_empty());
    facts.lock().unwrap().insert(x, info(DType::BF16));
    let before = graph.total_number_of_nodes();
    assert!(rule.apply(&mut graph, &matches).is_empty());
    assert_ne!(graph.find(x), graph.find(root));
    assert_eq!(graph.total_number_of_nodes(), before);
}
#[derive(Clone)]
struct Host(Arc<AtomicUsize>);
impl rule_host_identity::Functions for Host {
    fn allow_dtype(&self, dtype: DType) -> Option<bool> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Some(dtype == DType::F32)
    }
}
#[test]
fn host_functions_receive_rebound_dtype_values_at_application() {
    for t in [DType::F32, DType::BF16] {
        let (mut graph, x, root) = make_graph(t, t);
        let calls = Arc::new(AtomicUsize::new(0));
        let rule = rule_host_identity::build_rewrite(Host(calls.clone())).unwrap();
        let matches = rule.search(&graph);
        assert!(!matches.is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        rule.apply(&mut graph, &matches);
        graph.rebuild();
        assert_eq!(graph.find(x) == graph.find(root), t == DType::F32);
        assert!(calls.load(Ordering::Relaxed) > 0);
    }
}
