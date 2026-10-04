use egg::{EGraph, Id, Rewrite};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tepl_generated::ir::dialects::custom;
use tepl_generated::ir::dialects::{mirror, other};
use tepl_generated::ir::pattern::{OutputInference, TensorInfo, TensorMetadata, matches_at};
use tepl_generated::ir::rules::custom::*;
use tepl_generated::ir::{DType, Op, OpAttrs, OpNode};

fn info(shape: &[u64], dtype: DType) -> TensorInfo {
    TensorInfo {
        shape: shape.to_vec(),
        dtype,
    }
}
fn input(graph: &mut EGraph<OpNode, ()>, name: &str) -> Id {
    graph.add(
        OpNode::from_parts(
            Op::Custom(custom::Op::Input),
            vec![],
            OpAttrs::Custom(custom::OpAttrs::InputAttrs { name: name.into() }),
        )
        .unwrap(),
    )
}
fn operation(graph: &mut EGraph<OpNode, ()>, op: Op, children: Vec<Id>) -> Id {
    graph.add(OpNode::from_parts(op, children, OpAttrs::None).unwrap())
}
fn metadata(entries: Vec<(Id, TensorInfo)>) -> impl TensorMetadata<()> {
    let entries: HashMap<_, _> = entries.into_iter().collect();
    move |graph: &EGraph<OpNode, ()>, id: Id| {
        entries
            .iter()
            .find(|(key, _)| graph.find(**key) == graph.find(id))
            .map(|(_, value)| value.clone())
    }
}
#[derive(Clone, Copy)]
struct Inference;
impl OutputInference for Inference {
    fn infer_output(&self, op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
        if let OpAttrs::Literal {
            dtype: Some(dtype), ..
        } = attrs
        {
            return Some(info(&[], *dtype));
        }
        match op {
            Op::Custom(custom::Op::Add) => {
                let [left, right] = operands else {
                    return None;
                };
                if left.dtype != right.dtype {
                    return None;
                }
                if left.shape == right.shape || right.shape.is_empty() {
                    Some(left.clone())
                } else if left.shape.is_empty() {
                    Some(right.clone())
                } else {
                    None
                }
            }
            Op::Custom(custom::Op::Negate)
            | Op::Custom(custom::Op::Transform)
            | Op::Custom(custom::Op::Concat) => operands.first().cloned(),
            _ => None,
        }
    }
}
fn apply(graph: &mut EGraph<OpNode, ()>, rule: Rewrite<OpNode, ()>) -> Vec<Id> {
    graph.rebuild();
    let matches = rule.search(graph);
    rule.apply(graph, &matches)
}

#[test]
fn generated_dialects_have_distinct_operations_shared_schemas_and_variadic_arity() {
    assert_ne!(Op::Custom(custom::Op::Add), Op::Other(other::Op::Add));
    assert_ne!(Op::Custom(custom::Op::Add), Op::Mirror(mirror::Op::Add));
    assert_eq!(
        Op::from_name("Custom.add"),
        Some(Op::Custom(custom::Op::Add))
    );
    assert_eq!(
        Op::from_name("Custom.plus"),
        Some(Op::Custom(custom::Op::Add))
    );
    assert_eq!(Op::from_name("Other.add"), Some(Op::Other(other::Op::Add)));
    assert_ne!(
        Op::Custom(custom::Op::FooBar),
        Op::Custom(custom::Op::FooOther)
    );
    assert!(OpNode::from_parts(Op::Custom(custom::Op::Concat), vec![], OpAttrs::None).is_err());
    assert!(
        OpNode::from_parts(
            Op::Custom(custom::Op::Concat),
            vec![Id::from(0)],
            OpAttrs::None
        )
        .is_ok()
    );
    assert!(
        OpNode::from_parts(
            Op::Custom(custom::Op::Transform),
            vec![Id::from(0)],
            OpAttrs::Custom(custom::OpAttrs::AnotherAttrs { size: 1 })
        )
        .is_err()
    );
}

#[test]
fn middle_sequences_allow_empty_and_check_repeated_dimensions_and_dtype() {
    for (left, right, accepted) in [
        (vec![2, 4], vec![2, 4], true),
        (vec![2, 7, 8, 4], vec![2, 7, 8, 4], true),
        (vec![2, 7, 4], vec![2, 8, 4], false),
        (vec![2, 4], vec![3, 4], false),
        (vec![2], vec![2], false),
        (vec![2, 129], vec![2, 129], false),
    ] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let y = input(&mut graph, "y");
        let root = operation(&mut graph, Op::Custom(custom::Op::Add), vec![x, y]);
        let meta = metadata(vec![
            (x, info(&left, DType::F32)),
            (y, info(&right, DType::F32)),
            (root, info(&left, DType::F32)),
        ]);
        let rule = rule_commute::build_rewrite_with(meta, Inference, ()).unwrap();
        assert_eq!(!apply(&mut graph, rule).is_empty(), accepted);
    }
}

#[test]
fn inherited_restrictions_and_dtype_are_all_enforced() {
    for (shape, dtype, accepted) in [
        (vec![2, 3, 4], DType::F32, true),
        (vec![2, 3, 4], DType::I32, false),
        (vec![2, 4], DType::F32, false),
        (vec![0, 3, 4], DType::F32, false),
    ] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
        let meta = metadata(vec![(x, info(&shape, dtype)), (root, info(&shape, dtype))]);
        let rule = rule_inherited::build_rewrite_with(meta, Inference, ()).unwrap();
        assert_eq!(!apply(&mut graph, rule).is_empty(), accepted);
    }
}

#[test]
fn untyped_literal_patterns_match_any_dtype_and_rhs_uses_context() {
    for dtype in [DType::I32, DType::F32, DType::U64] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let literal = graph.add(OpNode::literal("1", dtype).unwrap());
        let root = operation(&mut graph, Op::Custom(custom::Op::Add), vec![x, literal]);
        graph.rebuild();
        assert_eq!(
            matches_at(&graph, root, &rule_literal_any::pattern()).len(),
            1
        );
        assert_eq!(
            !matches_at(&graph, root, &rule_literal_typed::pattern()).is_empty(),
            dtype == DType::I32
        );
        let meta = metadata(vec![(x, info(&[3], dtype)), (root, info(&[3], dtype))]);
        let rule = rule_literal_any::build_rewrite_with(meta, Inference, ()).unwrap();
        assert!(!apply(&mut graph, rule).is_empty());
        assert!(
            graph
                .lookup(
                    OpNode::from_parts(
                        Op::Custom(custom::Op::Add),
                        vec![literal, x],
                        OpAttrs::None
                    )
                    .unwrap()
                )
                .is_some()
        );
        assert!(
            !graph
                .classes()
                .flat_map(|class| &class.nodes)
                .any(|node| matches!(node.attrs(), OpAttrs::Literal { dtype: None, .. }))
        );
    }
}

#[test]
fn new_untyped_rhs_literal_is_resolved_before_insertion_and_failure_is_atomic() {
    struct Reject;
    impl OutputInference for Reject {
        fn infer_literal(&self, _: &str, _: Option<DType>, _: &TensorInfo) -> Option<DType> {
            None
        }
        fn infer_output(&self, _: Op, _: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
            None
        }
    }
    for accepted in [true, false] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
        let before = graph.total_size();
        let meta = metadata(vec![
            (x, info(&[3], DType::F32)),
            (root, info(&[3], DType::F32)),
        ]);
        let rule = if accepted {
            rule_construct_literal::build_rewrite_with(meta, Inference, ()).unwrap()
        } else {
            rule_construct_literal::build_rewrite_with(meta, Reject, ()).unwrap()
        };
        assert_eq!(!apply(&mut graph, rule).is_empty(), accepted);
        if accepted {
            assert!(
                graph
                    .lookup(OpNode::literal("2", DType::F32).unwrap())
                    .is_some()
            );
        } else {
            assert_eq!(graph.total_size(), before);
        }
    }
}

#[test]
fn lhs_binder_reuses_the_bound_eclass() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let y = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let root = operation(&mut graph, Op::Custom(custom::Op::Add), vec![y, y]);
    let meta = metadata(vec![
        (y, info(&[3], DType::F32)),
        (root, info(&[3], DType::F32)),
    ]);
    let rule = rule_bind::build_rewrite_with(meta, Inference, ()).unwrap();
    assert!(!apply(&mut graph, rule).is_empty());
    assert_eq!(graph.find(root), graph.find(y));
}

#[derive(Clone)]
struct Chain {
    calls: Arc<AtomicUsize>,
    wrong: bool,
}
impl rule_derive_chain::Functions for Chain {
    fn enabled(&self, _: &TensorInfo, _: &OpAttrs) -> Option<bool> {
        Some(true)
    }
    fn update(&self, attrs: &OpAttrs) -> Option<OpAttrs> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.wrong {
            return Some(OpAttrs::Custom(custom::OpAttrs::AnotherAttrs { size: 1 }));
        }
        let OpAttrs::Custom(custom::OpAttrs::Axes { axis, tags }) = attrs else {
            return None;
        };
        Some(OpAttrs::Custom(custom::OpAttrs::Axes {
            axis: axis + 1,
            tags: tags.clone(),
        }))
    }
}
#[test]
fn derivations_run_in_order_and_wrong_schemas_reject_without_insertion() {
    for wrong in [false, true] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let root = graph.add(
            OpNode::from_parts(
                Op::Custom(custom::Op::Transform),
                vec![x],
                OpAttrs::Custom(custom::OpAttrs::Axes {
                    axis: 0,
                    tags: vec![],
                }),
            )
            .unwrap(),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let before = graph.total_size();
        let meta = metadata(vec![
            (x, info(&[3], DType::F32)),
            (root, info(&[3], DType::F32)),
        ]);
        let rule = rule_derive_chain::build_rewrite_with(
            meta,
            Inference,
            Chain {
                calls: calls.clone(),
                wrong,
            },
        )
        .unwrap();
        assert_eq!(!apply(&mut graph, rule).is_empty(), !wrong);
        assert_eq!(calls.load(Ordering::SeqCst), if wrong { 1 } else { 2 });
        if wrong {
            assert_eq!(graph.total_size(), before);
        } else {
            assert!(
                graph
                    .lookup(
                        OpNode::from_parts(
                            Op::Custom(custom::Op::Transform),
                            vec![x],
                            OpAttrs::Custom(custom::OpAttrs::Axes {
                                axis: 2,
                                tags: vec![]
                            })
                        )
                        .unwrap()
                    )
                    .is_some()
            );
        }
    }
}

struct NoCalls;
impl rule_short_circuit::Functions for NoCalls {
    fn fails(&self, _: &TensorInfo) -> Option<bool> {
        panic!("short circuit must skip this call")
    }
}
#[test]
fn short_circuit_skips_fallible_host_calls() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let meta = metadata(vec![
        (x, info(&[], DType::F32)),
        (root, info(&[], DType::F32)),
    ]);
    let rule = rule_short_circuit::build_rewrite_with(meta, Inference, NoCalls).unwrap();
    assert!(!apply(&mut graph, rule).is_empty());
}

#[test]
fn overflow_and_division_by_zero_reject_in_debug_and_release() {
    for n in [2, u64::MAX] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
        let meta = metadata(vec![
            (x, info(&[n], DType::F32)),
            (root, info(&[n], DType::F32)),
        ]);
        let rule = rule_overflow::build_rewrite_with(meta, Inference, ()).unwrap();
        assert_eq!(!apply(&mut graph, rule).is_empty(), n != u64::MAX);
        let meta = metadata(vec![
            (x, info(&[n], DType::F32)),
            (root, info(&[n], DType::F32)),
        ]);
        let rule = rule_divide_zero::build_rewrite_with(meta, Inference, ()).unwrap();
        assert!(apply(&mut graph, rule).is_empty());
    }
}
struct Numeric;
impl rule_nested_calls::Functions for Numeric {
    fn twice(&self, value: u64) -> Option<u64> {
        value.checked_mul(2)
    }
    fn greater(&self, left: u64, right: u64) -> Option<bool> {
        Some(left > right)
    }
}
impl rule_signed_float::Functions for Numeric {
    fn signed(&self, _: &TensorInfo) -> Option<i64> {
        Some(-1)
    }
    fn floating(&self, _: &TensorInfo) -> Option<f64> {
        Some(1.5)
    }
}
#[test]
fn nested_host_calls_and_signed_float_expressions_execute() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let meta = metadata(vec![
        (x, info(&[2], DType::F32)),
        (root, info(&[2], DType::F32)),
    ]);
    assert!(
        !apply(
            &mut graph,
            rule_nested_calls::build_rewrite_with(meta, Inference, Numeric).unwrap()
        )
        .is_empty()
    );
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let meta = metadata(vec![
        (x, info(&[], DType::F32)),
        (root, info(&[], DType::F32)),
    ]);
    assert!(
        !apply(
            &mut graph,
            rule_signed_float::build_rewrite_with(meta, Inference, Numeric).unwrap()
        )
        .is_empty()
    );
}

#[test]
fn incompatible_output_metadata_rejects_without_partial_rhs() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let meta = metadata(vec![
        (x, info(&[3], DType::I32)),
        (root, info(&[3], DType::F32)),
    ]);
    let before = graph.total_size();
    let rule = rule_construct_literal::build_rewrite_with(meta, Inference, ()).unwrap();
    assert!(apply(&mut graph, rule).is_empty());
    assert_eq!(graph.total_size(), before);
}

#[test]
fn rhs_validation_finishes_before_host_inference() {
    use tepl_generated::ir::pattern::{
        AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern, tensor_rewrite_checked,
    };

    for valid_attrs in [false, true] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
        let var = "?x".parse().unwrap();
        let out = AttrVar::from("out");
        let lhs = TensorPattern::op(
            custom::Op::Negate,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(var)],
        );
        let rhs = TensorExpr::op(
            custom::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    custom::Op::Add,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(var), TensorExpr::Var(var)],
                ),
                TensorExpr::op(
                    custom::Op::Transform,
                    AttrExpr::Derived(out),
                    vec![TensorExpr::Var(var)],
                ),
            ],
        );
        let attrs = if valid_attrs {
            OpAttrs::Custom(custom::OpAttrs::Axes {
                axis: 0,
                tags: vec![],
            })
        } else {
            OpAttrs::Custom(custom::OpAttrs::AnotherAttrs { size: 1 })
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let inference_calls = calls.clone();
        let rule = tensor_rewrite_checked(
            "preflight",
            lhs,
            rhs,
            metadata(vec![
                (x, info(&[3], DType::F32)),
                (root, info(&[3], DType::F32)),
            ]),
            move |_: Op, _: &[TensorInfo], _: &OpAttrs| {
                inference_calls.fetch_add(1, Ordering::SeqCst);
                Some(info(&[3], DType::F32))
            },
            move |_, _| Some(HashMap::from([(out, attrs.clone())])),
        )
        .unwrap();
        let before = graph.total_size();
        assert_eq!(!apply(&mut graph, rule).is_empty(), valid_attrs);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            if valid_attrs { 3 } else { 0 }
        );
        if !valid_attrs {
            assert_eq!(graph.total_size(), before);
        }
    }
}

struct TypedHost;
impl rule_typed_pipeline::Functions for TypedHost {
    fn host_tensor(&self, tensor: &TensorInfo) -> Option<TensorInfo> {
        Some(tensor.clone())
    }
    fn host_dims(&self, dimensions: &[u64]) -> Option<Vec<u64>> {
        Some(dimensions.to_vec())
    }
    fn host_check(
        &self,
        tensor: &TensorInfo,
        dimensions: &[u64],
        boolean: bool,
        signed: i64,
        float: f64,
        attrs: &OpAttrs,
    ) -> Option<bool> {
        Some(
            tensor.shape == dimensions
                && boolean
                && signed == -1
                && float == 1.0
                && matches!(attrs, OpAttrs::Custom(custom::OpAttrs::Axes { .. })),
        )
    }
}

struct KeywordHost;
impl rule_reserved_names::Functions for KeywordHost {
    fn r#match(&self, tensor: &TensorInfo) -> Option<bool> {
        Some(tensor.dtype == DType::F32)
    }
}

#[test]
fn keyword_host_method_uses_raw_identifier_in_declaration_and_call() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let meta = metadata(vec![
        (x, info(&[2], DType::F32)),
        (root, info(&[2], DType::F32)),
    ]);
    let rule = rule_reserved_names::build_rewrite_with(meta, Inference, KeywordHost).unwrap();
    assert!(!apply(&mut graph, rule).is_empty());
    assert_eq!(graph.find(root), graph.find(x));
}
#[test]
fn nested_owned_tensor_and_sequence_results_borrow_correctly_in_host_calls() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = graph.add(
        OpNode::from_parts(
            Op::Custom(custom::Op::Transform),
            vec![x],
            OpAttrs::Custom(custom::OpAttrs::Axes {
                axis: 0,
                tags: vec![],
            }),
        )
        .unwrap(),
    );
    let meta = metadata(vec![
        (x, info(&[2, 3, 4], DType::F32)),
        (root, info(&[2, 3, 4], DType::F32)),
    ]);
    let rule = rule_typed_pipeline::build_rewrite_with(meta, Inference, TypedHost).unwrap();
    assert!(!apply(&mut graph, rule).is_empty());
    assert!(
        graph
            .lookup(
                OpNode::from_parts(Op::Custom(custom::Op::Negate), vec![x], OpAttrs::None).unwrap()
            )
            .is_some()
    );
}

impl rule_short_literal::Functions for NoCalls {
    fn fails(&self, _: &TensorInfo) -> Option<bool> {
        panic!("skipped branch must not require input metadata")
    }
}
#[test]
fn skipped_host_branch_does_not_eagerly_read_missing_capture_metadata() {
    let mut graph = EGraph::default();
    let x = input(&mut graph, "x");
    let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
    let meta = metadata(vec![(root, info(&[], DType::F32))]);
    let rule = rule_short_literal::build_rewrite_with(meta, Inference, NoCalls).unwrap();
    assert!(!apply(&mut graph, rule).is_empty());
    assert!(
        graph
            .lookup(OpNode::literal("1", DType::F32).unwrap())
            .is_some()
    );
}

#[test]
fn generated_mixed_shapes_and_where_prune_with_inherited_bindings() {
    for (shape, dtype, searched, applied) in [
        (vec![3, 7, 8, 9, 3], DType::F32, true, true),
        (vec![3, 9, 3], DType::F32, true, true),
        (vec![0, 0, 0], DType::F32, true, true),
        (vec![3, 3], DType::F32, false, false),
        (vec![3, 7, 8, 9, 4], DType::F32, false, false),
        (vec![3, 9, 3], DType::I32, false, false),
        (vec![5, 9, 5], DType::F32, false, false),
    ] {
        let mut graph = EGraph::default();
        let x = input(&mut graph, "x");
        let root = operation(&mut graph, Op::Custom(custom::Op::Negate), vec![x]);
        let meta = metadata(vec![(x, info(&shape, dtype)), (root, info(&shape, dtype))]);
        let rule = rule_mixed_inherited::build_rewrite_with(meta, Inference, ()).unwrap();
        graph.rebuild();
        let matches = rule.search(&graph);
        assert_eq!(!matches.is_empty(), searched, "{shape:?} {dtype:?}");
        let before = graph.total_size();
        assert_eq!(!rule.apply(&mut graph, &matches).is_empty(), applied);
        assert_eq!(graph.total_size(), before);
    }
}
