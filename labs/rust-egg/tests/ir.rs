use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, Searcher};
use rust_egg::host::nodes::{constant, reduce, reduction, symbol, symbol_name};
use rust_egg::ir::dialects::tensor_lang as t;
use rust_egg::ir::types::{Elements, Region, ReplicaGroups};
use rust_egg::ir::{Arity, Op, OpAttrs, OpNode};

fn sum_region() -> Region {
    Region(b"^bb0(%a: i32, %b: i32): %v = add %a, %b; return %v".to_vec())
}
fn max_region() -> Region {
    Region(b"^bb0(%a: i32, %b: i32): %v = maximum %a, %b; return %v".to_vec())
}

#[test]
fn canonical_names_and_aliases_resolve_to_one_operation() {
    for (canonical, alias, op) in [
        ("multiply", "mul", t::Op::Multiply),
        ("exponential", "exp", t::Op::Exponential),
        ("dot_general", "dot", t::Op::DotGeneral),
    ] {
        assert_eq!(t::Op::from_name(canonical), Some(op));
        assert_eq!(t::Op::from_name(alias), Some(op));
        assert_eq!(
            Op::from_name(&format!("TensorLang.{alias}")),
            Some(op.into())
        );
        assert_eq!(op.name(), canonical);
    }
    assert_eq!(t::Op::from_name("unknown"), None);
    assert_eq!(Op::from_name("add"), None);
}
#[test]
fn construction_rejects_bad_arity_and_attributes() {
    let x = Id::from(0);
    assert!(OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x]).is_err());
    assert!(OpNode::new(t::Op::Reduce, t::OpAttrs::None, vec![x]).is_err());
    assert!(
        OpNode::new(
            t::Op::Add,
            t::OpAttrs::ReduceAttrs {
                body: sum_region(),
                dimensions: vec![0]
            },
            vec![x, x]
        )
        .is_err()
    );
}
#[test]
fn attributes_follow_dialect_fields() {
    let x = Id::from(0);
    let input = symbol("x");
    assert_eq!(input.attrs(), &OpAttrs::Input { name: "x".into() });
    assert_eq!(symbol_name(&input), Some("x"));
    let value = Elements {
        element_type: "i32".into(),
        shape: vec![],
        data: 42i32.to_le_bytes().to_vec(),
    };
    let weight = constant(value.clone());
    assert_eq!(
        weight.attrs(),
        &OpAttrs::TensorLang(t::OpAttrs::ConstantAttrs { value })
    );
    assert_eq!(symbol_name(&weight), None);
    let sum = reduce(sum_region(), x, x, vec![0]);
    assert_eq!(reduction(&sum), Some((&sum_region(), &[0][..], x, x)));
    assert_eq!(
        OpNode::new(
            t::Op::AllReduce,
            t::OpAttrs::CollectiveReduce {
                replica_groups: ReplicaGroups::Explicit(vec![vec![0, 1]]),
                channel_id: 0,
                use_global_device_ids: false,
                computation: sum_region()
            },
            vec![x]
        )
        .unwrap()
        .op(),
        t::Op::AllReduce.into()
    );
}
#[test]
fn variadic_arity_comes_from_operand_declaration() {
    assert_eq!(t::Op::Concatenate.arity(), Arity::AtLeast(0));
    let x = Id::from(0);
    for count in [0, 1, 2, 3] {
        assert!(
            OpNode::new(
                t::Op::Concatenate,
                t::OpAttrs::ConcatenateAttrs { dimension: 0 },
                vec![x; count]
            )
            .is_ok()
        );
    }
}
#[test]
fn reduction_pattern_matches_region_and_dimensions() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(symbol("x"));
    let root = graph.add(reduce(sum_region(), x, x, vec![0]));
    graph.add(reduce(sum_region(), x, x, vec![1]));
    graph.add(reduce(max_region(), x, x, vec![0]));
    graph.rebuild();
    let mut ast = PatternAst::default();
    ast.add(ENodeOrVar::Var("?x".parse().unwrap()));
    ast.add(ENodeOrVar::ENode(reduce(
        sum_region(),
        Id::from(0),
        Id::from(0),
        vec![0],
    )));
    let pattern = Pattern::new(ast);
    let matches = pattern.search(&graph);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].eclass, graph.find(root));
    assert_eq!(
        reduce(sum_region(), x, x, vec![0]).discriminant(),
        t::Op::Reduce.into()
    );
}

#[test]
fn removed_operations_do_not_resolve() {
    for name in [
        "broadcast",
        "relu",
        "scale",
        "square",
        "alias",
        "rmsnorm",
        "vocab_cross_entropy",
        "online_attention",
        "symbol",
    ] {
        assert_eq!(t::Op::from_name(name), None, "{name}");
    }
}

#[test]
fn descriptors_preserve_signed_nested_optional_and_opaque_metadata() {
    use rust_egg::ir::types::{DotAlgorithm, Precision};
    let x = Id::from(0);
    let groups = ReplicaGroups::Explicit(vec![vec![0, 2], vec![1, 3]]);
    let cases = [
        (
            t::Op::BroadcastInDim,
            t::OpAttrs::BroadcastInDimAttrs {
                broadcast_dimensions: vec![1],
                shape: vec![2, 3],
            },
            vec![x],
        ),
        (
            t::Op::Slice,
            t::OpAttrs::SliceAttrs {
                start_indices: vec![0],
                limit_indices: vec![3],
                strides: vec![1],
            },
            vec![x],
        ),
        (
            t::Op::DotGeneral,
            t::OpAttrs::DotGeneralAttrs {
                lhs_contracting_dimensions: vec![1],
                rhs_contracting_dimensions: vec![0],
                lhs_batching_dimensions: vec![],
                rhs_batching_dimensions: vec![],
                precision_config: vec![Precision::High, Precision::Highest],
                algorithm: Some(DotAlgorithm {
                    lhs_precision_type: "f32".into(),
                    rhs_precision_type: "f32".into(),
                    accumulation_type: "f32".into(),
                    lhs_component_count: 1,
                    rhs_component_count: 1,
                    num_primitive_operations: 1,
                    allow_imprecise_accumulation: false,
                }),
            },
            vec![x, x],
        ),
        (
            t::Op::Convolution,
            t::OpAttrs::ConvolutionAttrs {
                window_strides: vec![1],
                padding: vec![vec![-1, 2]],
                lhs_dilation: vec![1],
                rhs_dilation: vec![1],
                window_reversal: vec![false],
                input_batch_dimension: 0,
                input_feature_dimension: 2,
                input_spatial_dimensions: vec![1],
                kernel_input_feature_dimension: 1,
                kernel_output_feature_dimension: 2,
                kernel_spatial_dimensions: vec![0],
                output_batch_dimension: 0,
                output_feature_dimension: 2,
                output_spatial_dimensions: vec![1],
                feature_group_count: 1,
                batch_group_count: 1,
                precision_config: vec![Precision::Default; 2],
            },
            vec![x, x],
        ),
        (
            t::Op::AllGather,
            t::OpAttrs::AllGatherAttrs {
                all_gather_dim: 0,
                replica_groups: groups.clone(),
                channel_id: -1,
                use_global_device_ids: true,
            },
            vec![x],
        ),
        (
            t::Op::ReduceScatter,
            t::OpAttrs::ReduceScatterAttrs {
                scatter_dimension: 0,
                replica_groups: groups.clone(),
                channel_id: 1,
                use_global_device_ids: false,
                computation: sum_region(),
            },
            vec![x],
        ),
        (
            t::Op::AllToAll,
            t::OpAttrs::AllToAllAttrs {
                split_dimension: 0,
                concat_dimension: 1,
                split_count: 2,
                replica_groups: groups,
                channel_id: 0,
            },
            vec![x],
        ),
    ];
    for (op, attrs, children) in cases {
        let node = OpNode::new(op, attrs.clone(), children).unwrap();
        assert_eq!(node.attrs(), &attrs.into());
        assert!(OpNode::new(op, t::OpAttrs::None, node.children().to_vec()).is_err());
    }
    let init = Id::from(1);
    let sum = reduce(sum_region(), x, init, vec![0]);
    assert_eq!(sum.children(), &[x, init]);
    assert!(!sum.matches(&reduce(max_region(), x, init, vec![0])));
    assert!(
        OpNode::new(
            t::Op::Reduce,
            t::OpAttrs::ReduceAttrs {
                dimensions: vec![0],
                body: sum_region(),
            },
            vec![x]
        )
        .is_err()
    );
}

#[test]
fn constant_payload_type_shape_and_bytes_participate_in_identity() {
    let value = Elements {
        element_type: "i32".into(),
        shape: vec![],
        data: 1i32.to_le_bytes().to_vec(),
    };
    let node = constant(value.clone());
    for other in [
        Elements {
            element_type: "f32".into(),
            ..value.clone()
        },
        Elements {
            shape: vec![1],
            ..value.clone()
        },
        Elements {
            data: 2i32.to_le_bytes().to_vec(),
            ..value.clone()
        },
    ] {
        assert!(!node.matches(&constant(other)));
    }
    assert_eq!(
        rust_egg::ir::analysis::infer_tensor_output(node.op(), &[], node.attrs()),
        Some(rust_egg::ir::pattern::TensorInfo {
            shape: vec![],
            dtype: rust_egg::ir::DType::I32
        })
    );
}
