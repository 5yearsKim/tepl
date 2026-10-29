use e::dialects::d;
use rt::values::AttributeValue;
use s::dialects::tensor_lang as t;
use tepl_generated_tests::current as c;
use tepl_generated_tests::{expressions as e, semantic as s};
use tepl_runtime::{self as rt, EvalError, Fact, NodeKind, Program, TensorFacts, Value};

fn run<P: Program>(
    graph: rt::egg::EGraph<rt::Node<P>, rt::TensorAnalysis<P>>,
    rules: Vec<rt::RuleDefinition<P>>,
) -> rt::EvalResult<rt::runner::CheckedRunner<P>> {
    let rules = rules
        .into_iter()
        .map(|r| r.build())
        .collect::<rt::EvalResult<Vec<_>>>()?;
    rt::runner::run(
        rt::runner::runner(graph)?
            .with_iter_limit(8)
            .with_node_limit(10_000),
        &rules,
    )
}
fn input<P: Program>(
    g: &mut rt::egg::EGraph<rt::Node<P>, rt::TensorAnalysis<P>>,
    name: &str,
    shape: Vec<u64>,
) -> rt::egg::Id {
    g.analysis
        .register_input(name, TensorFacts::known(shape, rt::DType::F32))
        .unwrap();
    g.add(rt::Node::input(name))
}
fn snode(
    g: &mut s::EGraph,
    op: t::Op,
    attrs: impl Into<s::OpAttrs>,
    children: Vec<rt::egg::Id>,
) -> rt::egg::Id {
    g.add(s::OpNode::operation(op, Some(attrs.into()), children).unwrap())
}
fn enode(g: &mut e::EGraph, op: d::Op, children: Vec<rt::egg::Id>) -> rt::egg::Id {
    g.add(e::OpNode::operation(op, None, children).unwrap())
}
fn eattrs(p: Vec<u64>) -> d::Shared {
    d::Shared {
        permutation: p,
        note: None,
        enabled: true,
        precision: rt::types::Precision::Default,
        signed: 0,
    }
}
fn etranspose(g: &mut e::EGraph, p: Vec<u64>, x: rt::egg::Id) -> rt::egg::Id {
    g.add(e::OpNode::operation(d::Op::Transpose, Some(eattrs(p).into()), vec![x]).unwrap())
}
#[test]
fn source_and_json_programs_build_the_same_rules() {
    assert_eq!(
        e::rules::all().unwrap().len(),
        tepl_generated_tests::loaded::rules::all().unwrap().len()
    );
    assert!(!s::rules::all().unwrap().is_empty());
    assert_eq!(s::SEMANTIC_POLICY, "algebraic");
    let a = e::graphs::expressions::graph_composed::build().unwrap();
    let b = tepl_generated_tests::loaded::graphs::expressions::graph_composed::build().unwrap();
    assert_eq!(a.graph[a.root].data, b.graph[b.root].data);
}
#[test]
fn constructors_locals_comprehensions_and_shared_schemas() {
    let built = e::graphs::expressions::graph_composed::build().unwrap();
    let root = built.root;
    let x = built.bindings["X"];
    let out = run(
        built.graph,
        vec![e::rules::expressions::rule_compose::definition()],
    )
    .unwrap();
    let class = &out.egraph[root];
    assert_eq!(class.data, TensorFacts::known(vec![2, 3], rt::DType::F32));
    assert!(class.nodes.iter().any(|n| {
        n.operands() == [out.egraph.find(x)]
            && n.attrs().is_some_and(|a| match a {
                e::OpAttrs::D(d::OpAttrs::Shared(a)) => {
                    a.permutation == [0, 1]
                        && a.note.as_deref() == Some("derived")
                        && a.signed == -1
                        && a.precision == rt::types::Precision::High
                }
            })
    }));
    let out = run(
        out.egraph,
        vec![e::rules::expressions::rule_identity::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph.find(root), out.egraph.find(x));
    let variadic = e::graphs::expressions::graph_variadic::build().unwrap();
    assert_eq!(
        variadic.graph[variadic.root].data,
        TensorFacts::known(vec![2, 3], rt::DType::F32)
    );
    let literal = e::graphs::expressions::graph_literal::build().unwrap();
    assert_eq!(
        literal.graph[literal.root].data,
        TensorFacts::known(vec![], rt::DType::I32)
    );
    let attrs = eattrs(vec![0]);
    // A second operation reuses the exact same schema type.
    e::OpNode::operation(d::Op::Other, Some(attrs.into()), vec![0.into()]).unwrap();
}
#[test]
fn auto_transpose_composition_and_reduction() {
    let mut graph = s::EGraph::default();
    let x = input(&mut graph, "X", vec![2, 3, 4]);
    let inner = snode(
        &mut graph,
        t::Op::Transpose,
        t::TransposeAttrs {
            permutation: vec![1, 2, 0],
        },
        vec![x],
    );
    let root = snode(
        &mut graph,
        t::Op::Transpose,
        t::TransposeAttrs {
            permutation: vec![2, 0, 1],
        },
        vec![inner],
    );
    let out = run(
        graph,
        vec![s::rules::reductions::rule_transpose_compose::definition()],
    )
    .unwrap();
    assert!(out.egraph[root].nodes.iter().any(|n| n.operands()==[out.egraph.find(x)] && n.attrs().is_some_and(|a|matches!(a,s::OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs(a)) if a.permutation==[0,1,2]))));
    let mut graph = s::EGraph::default();
    let a = input(&mut graph, "A", vec![2, 3, 4]);
    let transposed = snode(
        &mut graph,
        t::Op::Transpose,
        t::TransposeAttrs {
            permutation: vec![2, 0, 1],
        },
        vec![a],
    );
    let root = snode(
        &mut graph,
        t::Op::ReduceSum,
        t::ReduceSumAttrs {
            dimensions: vec![1],
        },
        vec![transposed],
    );
    let out = run(
        graph,
        vec![s::rules::reductions::rule_transpose_reduce::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph[root].data.shape, Fact::Known(vec![4, 3]));
    let corrected=out.egraph[root].nodes.iter().find(|n|n.attrs().is_some_and(|a|matches!(a,s::OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs(a)) if a.permutation==[1,0]))).unwrap();
    assert!(out.egraph[corrected.operands()[0]].nodes.iter().any(|n| n.operands()==[out.egraph.find(a)] && n.attrs().is_some_and(|a|matches!(a,s::OpAttrs::TensorLang(t::OpAttrs::ReduceSumAttrs(a)) if a.dimensions==[0]))));
}
#[test]
fn auto_broadcast_and_slice() {
    let mut graph = s::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    let inner = snode(
        &mut graph,
        t::Op::BroadcastInDim,
        t::BroadcastInDimAttrs {
            dimensions: vec![0],
            new_extents: vec![3],
        },
        vec![x],
    );
    let root = snode(
        &mut graph,
        t::Op::BroadcastInDim,
        t::BroadcastInDimAttrs {
            dimensions: vec![0, 1],
            new_extents: vec![4],
        },
        vec![inner],
    );
    let out = run(
        graph,
        vec![s::rules::broadcast::rule_broadcast_compose::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph[root].data.shape, Fact::Known(vec![2, 3, 4]));
    assert!(out.egraph[root].nodes.iter().any(|n|n.operands()==[out.egraph.find(x)] && n.attrs().is_some_and(|a|matches!(a,s::OpAttrs::TensorLang(t::OpAttrs::BroadcastInDimAttrs(a)) if a.dimensions==[0] && a.new_extents==[3,4]))));
    let mut graph = s::EGraph::default();
    let x = input(&mut graph, "X", vec![10]);
    let inner = snode(
        &mut graph,
        t::Op::Slice,
        t::SliceAttrs {
            starts: vec![1],
            sizes: vec![6],
            strides: vec![1],
        },
        vec![x],
    );
    let root = snode(
        &mut graph,
        t::Op::Slice,
        t::SliceAttrs {
            starts: vec![1],
            sizes: vec![2],
            strides: vec![1],
        },
        vec![inner],
    );
    let out = run(
        graph,
        vec![s::rules::slice::rule_compose_slice::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph[root].data.shape, Fact::Known(vec![2]));
    assert!(out.egraph[root].nodes.iter().any(|n|n.operands()==[out.egraph.find(x)] && n.attrs().is_some_and(|a|matches!(a,s::OpAttrs::TensorLang(t::OpAttrs::SliceAttrs(a)) if a.starts==[2] && a.sizes==[2] && a.strides==[1]))));
}
#[test]
fn convolution_uses_axes_and_generated_attributes() {
    let mut graph = s::EGraph::default();
    let a = input(&mut graph, "A", vec![1, 5, 2]);
    let b = input(&mut graph, "B", vec![1, 5, 2]);
    let w = input(&mut graph, "W", vec![3, 2, 4]);
    let add = graph.add(s::OpNode::operation(t::Op::Add, None, vec![a, b]).unwrap());
    let root = snode(
        &mut graph,
        t::Op::Convolution,
        t::ConvolutionAttrs {
            strides: vec![1],
            dilations: vec![1],
        },
        vec![add, w],
    );
    let out = run(
        graph,
        vec![s::rules::convolution::rule_convolution_distribute_input::definition()],
    )
    .unwrap();
    assert_eq!(
        out.egraph[root].data,
        TensorFacts::known(vec![1, 3, 4], rt::DType::F32)
    );
    let add = out.egraph[root]
        .nodes
        .iter()
        .find(|n| matches!(n.kind(), NodeKind::Operation(s::Op::TensorLang(t::Op::Add))))
        .unwrap();
    for (child, input) in add.operands().iter().zip([a, b]) {
        assert!(out.egraph[*child].nodes.iter().any(|n| n.operands()==[out.egraph.find(input),out.egraph.find(w)] && n.attrs().is_some_and(|a|matches!(a,s::OpAttrs::TensorLang(t::OpAttrs::ConvolutionAttrs(a)) if a.strides==[1] && a.dilations==[1]))));
    }
}
#[test]
fn attribute_witnesses_remain_paired_with_captures() {
    let mut graph = e::EGraph::default();
    let a = input(&mut graph, "A", vec![2, 2]);
    let b = input(&mut graph, "B", vec![2, 2]);
    let left = etranspose(&mut graph, vec![1, 0], a);
    let right = etranspose(&mut graph, vec![0, 1], b);
    graph.union(left, right);
    graph.rebuild();
    let out = run(
        graph,
        vec![e::rules::expressions::rule_identity::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph.find(left), out.egraph.find(b));
    assert_ne!(out.egraph.find(left), out.egraph.find(a));
}
#[test]
fn short_circuit_and_required_metadata() {
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    let root = enode(&mut graph, d::Op::Copy, vec![x]);
    let out = run(
        graph,
        vec![e::rules::expressions::rule_short_circuit::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph.find(root), out.egraph.find(x));
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    let root = enode(&mut graph, d::Op::Sink, vec![x]);
    let out = run(
        graph,
        vec![e::rules::expressions::rule_false_guard::definition()],
    )
    .unwrap();
    assert_ne!(out.egraph.find(root), out.egraph.find(x));
    let mut graph = e::EGraph::default();
    let x = graph.add(e::OpNode::input("X"));
    let root = enode(&mut graph, d::Op::Unknown, vec![x]);
    let out = run(
        graph,
        vec![e::rules::expressions::rule_require_shape::definition()],
    )
    .unwrap();
    assert_ne!(out.egraph.find(root), out.egraph.find(x));
}
#[test]
fn invalid_candidates_are_reported_without_inserting_rhs_or_stopping_other_rules() {
    for definition in [
        e::rules::expressions::rule_invalid_guard::definition(),
        e::rules::expressions::rule_invalid_rhs::definition(),
    ] {
        let mut graph = e::EGraph::default();
        let x = input(&mut graph, "X", vec![2]);
        let root = enode(&mut graph, d::Op::Sink, vec![x]);
        let before = graph.total_size();
        let rule_name = definition.name;
        let source = definition.source;
        let rule = definition.build().unwrap();
        let found = rule.search(&graph);
        assert!(rule.apply(&mut graph, &found).is_empty());
        assert_eq!(before, graph.total_size());
        assert_ne!(graph.find(root), graph.find(x));
        graph.analysis.check().unwrap();
        let rejected = graph.analysis.rejections();
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].rule, rule_name);
        assert_eq!(rejected[0].source, source);
        assert!(matches!(rejected[0].error, EvalError::Invalid(_)));
        let out = run(
            graph,
            vec![e::rules::expressions::rule_valid_sink::definition()],
        )
        .unwrap();
        assert_eq!(out.egraph.find(root), out.egraph.find(x));
    }
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    let root = enode(&mut graph, d::Op::Sink, vec![x]);
    let before = graph.total_size();
    let out = run(
        graph,
        vec![e::rules::expressions::rule_incompatible_rhs::definition()],
    )
    .unwrap();
    assert_eq!(before, out.egraph.total_size());
    assert_ne!(out.egraph.find(root), out.egraph.find(x));
}
#[test]
fn conflicts_stop_execution_and_partial_facts_propagate() {
    let mut graph = e::EGraph::default();
    let a = input(&mut graph, "A", vec![2, 3]);
    let b = input(&mut graph, "B", vec![3, 2]);
    graph.union(a, b);
    graph.rebuild();
    assert!(matches!(
        graph.analysis.check(),
        Err(EvalError::Conflict(_))
    ));
    assert!(rt::runner::runner(graph).is_err());
    let mut analysis = e::TensorAnalysis::default();
    analysis
        .register_input(
            "X",
            TensorFacts {
                shape: Fact::Known(vec![2]),
                dtype: Fact::Unknown,
            },
        )
        .unwrap();
    let mut graph = e::EGraph::new(analysis);
    let x = graph.add(e::OpNode::input("X"));
    let copy = enode(&mut graph, d::Op::Copy, vec![x]);
    assert_eq!(graph[copy].data.shape, Fact::Known(vec![2]));
    assert_eq!(graph[copy].data.dtype, Fact::Unknown);
    let unknown = graph.add(e::OpNode::input("unknown"));
    let parent = enode(&mut graph, d::Op::Copy, vec![unknown]);
    assert_eq!(graph[parent].data.shape, Fact::Unknown);
    graph.union(unknown, x);
    graph.rebuild();
    assert_eq!(graph[parent].data.shape, Fact::Known(vec![2]));
}
#[test]
fn schemas_and_literals_are_checked_at_the_boundary() {
    let wrong = e::OpNode::operation(d::Op::Copy, Some(eattrs(vec![0]).into()), vec![0.into()]);
    assert!(wrong.is_err());
    assert!(e::OpNode::literal("256", Some(rt::DType::U8)).is_err());
    assert!(e::OpNode::operation(d::Op::Copy, None, vec![]).is_err());
    assert!(
        e::OpAttrs::from_value(&Value::Descriptor {
            schema: 0,
            fields: vec![]
        })
        .is_err()
    );
}

#[test]
fn literal_matching_contextual_dtype_and_binders() {
    let mut graph = e::EGraph::default();
    graph
        .analysis
        .register_input("X", TensorFacts::known(vec![], rt::DType::I32))
        .unwrap();
    let x = graph.add(e::OpNode::input("X"));
    let one = graph.add(e::OpNode::literal("01", Some(rt::DType::I32)).unwrap());
    let sum = enode(&mut graph, d::Op::Add, vec![x, one]);
    let out = run(
        graph,
        vec![e::rules::expressions::rule_match_literal::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph.find(sum), out.egraph.find(x));
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![]);
    let root = enode(&mut graph, d::Op::Copy, vec![x]);
    let out = run(
        graph,
        vec![e::rules::expressions::rule_literal::definition()],
    )
    .unwrap();
    assert!(out.egraph[root].nodes.iter().any(
        |n| matches!(n.kind(),NodeKind::Literal{value,dtype:Some(rt::DType::F32)} if value=="0")
    ));
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    let inner = enode(&mut graph, d::Op::Copy, vec![x]);
    let root = enode(&mut graph, d::Op::Copy, vec![inner]);
    let out = run(graph, vec![e::rules::expressions::rule_bind::definition()]).unwrap();
    assert_eq!(out.egraph.find(root), out.egraph.find(inner));
}
#[test]
fn dimensions_dtype_bindings_and_repeated_captures() {
    for definition in [
        e::rules::expressions::rule_sequence::definition(),
        e::rules::expressions::rule_dtype_variable::definition(),
        e::rules::expressions::rule_float_guard::definition(),
    ] {
        let mut graph = e::EGraph::default();
        let x = input(&mut graph, "X", vec![2, 3, 4]);
        let root = enode(&mut graph, d::Op::Copy, vec![x]);
        let out = run(graph, vec![definition]).unwrap();
        assert_eq!(out.egraph.find(root), out.egraph.find(x));
    }
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    let y = input(&mut graph, "Y", vec![2]);
    let root = enode(&mut graph, d::Op::Add, vec![x, y]);
    let out = run(
        graph,
        vec![e::rules::expressions::rule_repeated::definition()],
    )
    .unwrap();
    assert_ne!(out.egraph.find(root), out.egraph.find(x));
    let mut graph = out.egraph;
    graph.union(x, y);
    graph.rebuild();
    let out = run(
        graph,
        vec![e::rules::expressions::rule_repeated::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph.find(root), out.egraph.find(x));
}
#[test]
fn input_registration_is_atomic_and_precedes_insertion() {
    let mut graph = e::EGraph::default();
    let x = input(&mut graph, "X", vec![2]);
    assert!(
        graph
            .analysis
            .register_input("X", TensorFacts::known(vec![3], rt::DType::F32))
            .is_err()
    );
    let copy = enode(&mut graph, d::Op::Copy, vec![x]);
    assert_eq!(graph[copy].data.shape, Fact::Known(vec![2]));
    let late = graph.add(e::OpNode::input("late"));
    assert!(
        graph
            .analysis
            .register_input("late", TensorFacts::known(vec![2], rt::DType::F32))
            .is_err()
    );
    assert_eq!(graph[late].data.shape, Fact::Unknown);
    graph.analysis.check().unwrap();
}

#[test]
fn logical_axis_guards_run_in_egg_with_empty_zero_and_singleton_dimensions() {
    for built in [
        c::graphs::current::graph_identity::build().unwrap(),
        c::graphs::current::graph_empty::build().unwrap(),
        c::graphs::current::graph_zero_singleton::build().unwrap(),
    ] {
        let root = built.root;
        let x = built.bindings["X"];
        let expected = built.graph[root].data.clone();
        let out = run(
            built.graph,
            vec![c::rules::current::rule_logical_axes::definition()],
        )
        .unwrap();
        assert_eq!(out.egraph[root].data, expected);
        assert!(
            out.egraph[root]
                .nodes
                .iter()
                .any(|node| node.operands() == [out.egraph.find(x)])
        );
        assert!(out.egraph.analysis.rejections().is_empty());
    }
    let built = c::graphs::current::graph_nonidentity::build().unwrap();
    let before = built.graph.total_size();
    let out = run(
        built.graph,
        vec![c::rules::current::rule_logical_axes::definition()],
    )
    .unwrap();
    assert_eq!(out.egraph.total_size(), before);
    assert!(out.egraph.analysis.rejections().is_empty());
}
