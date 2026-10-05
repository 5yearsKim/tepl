//! Behavior specification for the complete TEPL-shaped Rust inference reference.
use rust_egg::ir::analysis::{Inference, TensorInfo, infer_dtype, infer_shape, infer_tensor};
use rust_egg::ir::dialects::{scalar, tensor_lang as t};
use rust_egg::ir::types::{Elements, Precision, Region, ReplicaGroups};
use rust_egg::ir::{DType, OpAttrs};

fn shape(op: t::Op, operands: &[&[u64]], attrs: t::OpAttrs) -> Inference<Vec<u64>> {
    infer_shape(op.into(), operands, &attrs.into())
}

fn invalid(op: t::Op, operands: &[&[u64]], attrs: t::OpAttrs) {
    assert!(
        matches!(shape(op, operands, attrs), Inference::Invalid(_)),
        "{op:?}"
    );
}

fn dot_attrs() -> t::OpAttrs {
    t::OpAttrs::DotGeneralAttrs {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        precision_config: vec![Precision::Default; 2],
        algorithm: None,
    }
}

fn convolution_attrs() -> t::OpAttrs {
    // NCHW input, OIHW kernel, NCHW output.
    t::OpAttrs::ConvolutionAttrs {
        window_strides: vec![1, 1],
        padding: vec![vec![0, 0], vec![0, 0]],
        lhs_dilation: vec![1, 1],
        rhs_dilation: vec![1, 1],
        window_reversal: vec![false, false],
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: vec![2, 3],
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 0,
        kernel_spatial_dimensions: vec![2, 3],
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: vec![2, 3],
        feature_group_count: 1,
        batch_group_count: 1,
        precision_config: vec![Precision::Default; 2],
    }
}

fn collective_attrs() -> t::OpAttrs {
    t::OpAttrs::CollectiveReduce {
        replica_groups: ReplicaGroups::Explicit(vec![vec![0, 1]]),
        channel_id: 0,
        use_global_device_ids: false,
        computation: Region(vec![]),
    }
}

fn all_to_all_attrs(split: u64, join: u64, count: u64) -> t::OpAttrs {
    t::OpAttrs::AllToAllAttrs {
        split_dimension: split,
        concat_dimension: join,
        split_count: count,
        replica_groups: ReplicaGroups::Explicit(vec![vec![0, 1]]),
        channel_id: 0,
    }
}

#[test]
fn every_tepl_shape_block_has_a_working_evaluator_and_signature_checks() {
    // A valid example for each definition; equality expectations are handwritten,
    // independently of the implementation. Unsupported ops are tested below.
    let mut covered = 0;
    for op in [
        t::Op::Add,
        t::Op::Subtract,
        t::Op::Multiply,
        t::Op::Divide,
        t::Op::Maximum,
        t::Op::Minimum,
    ] {
        assert_eq!(
            shape(op, &[&[2, 3], &[2, 3]], t::OpAttrs::None),
            Inference::Known(vec![2, 3])
        );
        invalid(op, &[&[2, 3], &[2, 4]], t::OpAttrs::None);
        invalid(op, &[&[2, 3]], t::OpAttrs::None);
        covered += 1;
    }
    for op in [
        t::Op::Negate,
        t::Op::Exponential,
        t::Op::Log,
        t::Op::Abs,
        t::Op::Sqrt,
        t::Op::Rsqrt,
    ] {
        assert_eq!(
            shape(op, &[&[0, 3]], t::OpAttrs::None),
            Inference::Known(vec![0, 3])
        );
        invalid(op, &[], t::OpAttrs::None);
        covered += 1;
    }
    let cases: Vec<(t::Op, Vec<Vec<u64>>, t::OpAttrs, Vec<u64>)> = vec![
        (
            t::Op::Reshape,
            vec![vec![2, 3]],
            t::OpAttrs::ReshapeAttrs { shape: vec![3, 2] },
            vec![3, 2],
        ),
        (
            t::Op::Transpose,
            vec![vec![2, 3]],
            t::OpAttrs::TransposeAttrs {
                permutation: vec![1, 0],
            },
            vec![3, 2],
        ),
        (
            t::Op::BroadcastInDim,
            vec![vec![1, 3]],
            t::OpAttrs::BroadcastInDimAttrs {
                broadcast_dimensions: vec![0, 1],
                shape: vec![2, 3],
            },
            vec![2, 3],
        ),
        (
            t::Op::Slice,
            vec![vec![9, 7]],
            t::OpAttrs::SliceAttrs {
                start_indices: vec![1, 0],
                limit_indices: vec![8, 7],
                strides: vec![2, 3],
            },
            vec![4, 3],
        ),
        (
            t::Op::Concatenate,
            vec![vec![2, 3], vec![2, 4]],
            t::OpAttrs::ConcatenateAttrs { dimension: 1 },
            vec![2, 7],
        ),
        (
            t::Op::Reduce,
            vec![vec![2, 3, 4], vec![]],
            t::OpAttrs::ReduceAttrs {
                dimensions: vec![2, 0],
                body: Region(vec![]),
            },
            vec![3],
        ),
        (
            t::Op::DotGeneral,
            vec![vec![2, 3], vec![3, 4]],
            dot_attrs(),
            vec![2, 4],
        ),
        (
            t::Op::Convolution,
            vec![vec![2, 3, 7, 8], vec![4, 3, 3, 3]],
            convolution_attrs(),
            vec![2, 4, 5, 6],
        ),
        (
            t::Op::AllReduce,
            vec![vec![2, 3]],
            collective_attrs(),
            vec![2, 3],
        ),
        (
            t::Op::AllToAll,
            vec![vec![4, 3]],
            all_to_all_attrs(0, 1, 2),
            vec![2, 6],
        ),
    ];
    for (op, operands, attrs, expected) in cases {
        let refs: Vec<_> = operands.iter().map(Vec::as_slice).collect();
        assert_eq!(
            shape(op, &refs, attrs.clone()),
            Inference::Known(expected),
            "{op:?}"
        );
        // All of these definitions require a schema. Wrong schema is Invalid.
        invalid(op, &refs, t::OpAttrs::None);
        covered += 1;
    }
    for (op, operands) in [
        (scalar::Op::Add, vec![vec![], vec![]]),
        (scalar::Op::Negate, vec![vec![]]),
    ] {
        let refs: Vec<_> = operands.iter().map(Vec::as_slice).collect();
        assert_eq!(
            infer_shape(op.into(), &refs, &OpAttrs::None),
            Inference::Known(vec![])
        );
        assert!(matches!(
            infer_shape(op.into(), &[], &OpAttrs::None),
            Inference::Invalid(_)
        ));
        covered += 1;
    }
    // Guard reference coverage against shape definitions added to the examples.
    let declared = include_str!("../../../examples/dialects/tensor.tepl")
        .matches("        shape(")
        .count()
        + include_str!("../../../examples/dialects/scalar.tepl")
            .matches("        shape(")
            .count();
    assert_eq!(covered, declared);
    assert_eq!(covered, 24);
}

#[test]
fn absent_definitions_are_unknown_including_payload_bearing_constants() {
    let constant = t::OpAttrs::ConstantAttrs {
        value: Elements {
            element_type: "f32".into(),
            shape: vec![2, 3],
            data: vec![],
        },
    };
    let gather = t::OpAttrs::AllGatherAttrs {
        all_gather_dim: 0,
        replica_groups: ReplicaGroups::Explicit(vec![]),
        channel_id: 0,
        use_global_device_ids: false,
    };
    let scatter = t::OpAttrs::ReduceScatterAttrs {
        scatter_dimension: 0,
        replica_groups: ReplicaGroups::Explicit(vec![]),
        channel_id: 0,
        use_global_device_ids: false,
        computation: Region(vec![]),
    };
    for (op, attrs, operands) in [
        (t::Op::Constant, constant.clone(), vec![]),
        (t::Op::AllGather, gather, vec![vec![2, 3]]),
        (t::Op::ReduceScatter, scatter, vec![vec![2, 3]]),
    ] {
        let refs: Vec<_> = operands.iter().map(Vec::as_slice).collect();
        assert_eq!(shape(op, &refs, attrs.clone()), Inference::Unknown);
        if op != t::Op::Constant {
            let infos: Vec<_> = operands
                .into_iter()
                .map(|shape| TensorInfo {
                    shape,
                    dtype: DType::F32,
                })
                .collect();
            assert_eq!(
                infer_tensor(op.into(), &infos, &attrs.into()),
                Inference::Unknown
            );
        }
    }
    assert_eq!(
        infer_tensor(t::Op::Constant.into(), &[], &constant.into()),
        Inference::Unknown
    );
}

#[test]
fn reshape_products_use_i128_without_narrowing_input_dimensions() {
    assert_eq!(
        shape(
            t::Op::Reshape,
            &[&[u64::MAX, 2]],
            t::OpAttrs::ReshapeAttrs {
                shape: vec![2, u64::MAX]
            }
        ),
        Inference::Known(vec![2, u64::MAX])
    );
    invalid(
        t::Op::Reshape,
        &[&[u64::MAX, u64::MAX]],
        t::OpAttrs::ReshapeAttrs {
            shape: vec![u64::MAX, u64::MAX],
        },
    );
    assert_eq!(
        shape(
            t::Op::Reshape,
            &[&[u64::MAX, u64::MAX, 0]],
            t::OpAttrs::ReshapeAttrs { shape: vec![0] }
        ),
        Inference::Known(vec![0])
    );
}

#[test]
fn broadcast_mapping_checks_rank_axes_and_each_dimension() {
    for (mapping, output) in [
        (vec![0], vec![2, 3]),
        (vec![0, 0], vec![2, 3]),
        (vec![0, 2], vec![2, 3]),
        (vec![0, u64::MAX], vec![2, 3]),
        (vec![0, 1], vec![2, 4]),
    ] {
        invalid(
            t::Op::BroadcastInDim,
            &[&[1, 3]],
            t::OpAttrs::BroadcastInDimAttrs {
                broadcast_dimensions: mapping,
                shape: output,
            },
        );
    }
    assert_eq!(
        shape(
            t::Op::BroadcastInDim,
            &[&[]],
            t::OpAttrs::BroadcastInDimAttrs {
                broadcast_dimensions: vec![],
                shape: vec![2, 0]
            }
        ),
        Inference::Known(vec![2, 0])
    );
    assert_eq!(
        shape(
            t::Op::BroadcastInDim,
            &[&[3, 1]],
            t::OpAttrs::BroadcastInDimAttrs {
                broadcast_dimensions: vec![1, 0],
                shape: vec![2, 3]
            }
        ),
        Inference::Known(vec![2, 3])
    );
}

#[test]
fn slice_checks_all_attribute_lengths_bounds_and_positive_strides() {
    for (start, limit, stride) in [
        (vec![], vec![5], vec![1]),
        (vec![0], vec![], vec![1]),
        (vec![0], vec![5], vec![]),
        (vec![4], vec![3], vec![1]),
        (vec![0], vec![6], vec![1]),
        (vec![0], vec![5], vec![0]),
    ] {
        invalid(
            t::Op::Slice,
            &[&[5]],
            t::OpAttrs::SliceAttrs {
                start_indices: start,
                limit_indices: limit,
                strides: stride,
            },
        );
    }
    assert_eq!(
        shape(
            t::Op::Slice,
            &[&[u64::MAX]],
            t::OpAttrs::SliceAttrs {
                start_indices: vec![0],
                limit_indices: vec![u64::MAX],
                strides: vec![2]
            }
        ),
        Inference::Known(vec![1 << 63])
    );
}

#[test]
fn concatenate_checks_variadic_inputs_and_validates_yielded_dimensions() {
    let attrs = t::OpAttrs::ConcatenateAttrs { dimension: 1 };
    invalid(t::Op::Concatenate, &[], attrs.clone());
    invalid(t::Op::Concatenate, &[&[2, 3], &[2]], attrs.clone());
    invalid(t::Op::Concatenate, &[&[2, 3], &[4, 3]], attrs);
    invalid(
        t::Op::Concatenate,
        &[&[2, 3]],
        t::OpAttrs::ConcatenateAttrs {
            dimension: u64::MAX,
        },
    );
    invalid(
        t::Op::Concatenate,
        &[&[u64::MAX], &[1]],
        t::OpAttrs::ConcatenateAttrs { dimension: 0 },
    );
    assert_eq!(
        shape(
            t::Op::Concatenate,
            &[&[0], &[2], &[3]],
            t::OpAttrs::ConcatenateAttrs { dimension: 0 }
        ),
        Inference::Known(vec![5])
    );
}

#[test]
fn reduce_checks_initializer_and_axes_before_gathering_the_result() {
    invalid(
        t::Op::Reduce,
        &[&[2, 3], &[1]],
        t::OpAttrs::ReduceAttrs {
            dimensions: vec![0],
            body: Region(vec![]),
        },
    );
    for axes in [vec![0, 0], vec![2], vec![u64::MAX]] {
        invalid(
            t::Op::Reduce,
            &[&[2, 3], &[]],
            t::OpAttrs::ReduceAttrs {
                dimensions: axes,
                body: Region(vec![]),
            },
        );
    }
    assert_eq!(
        shape(
            t::Op::Reduce,
            &[&[2, 3], &[]],
            t::OpAttrs::ReduceAttrs {
                dimensions: vec![0, 1],
                body: Region(vec![])
            }
        ),
        Inference::Known(vec![])
    );
}

#[test]
fn all_to_all_checks_counts_and_preserves_equal_split_and_join_axes() {
    for (split, join, count) in [(2, 0, 2), (0, 2, 2), (0, 1, 0), (0, 1, 3)] {
        invalid(
            t::Op::AllToAll,
            &[&[4, 3]],
            all_to_all_attrs(split, join, count),
        );
    }
    assert_eq!(
        shape(
            t::Op::AllToAll,
            &[&[u64::MAX]],
            all_to_all_attrs(0, 0, u64::MAX)
        ),
        Inference::Known(vec![u64::MAX])
    );
    invalid(
        t::Op::AllToAll,
        &[&[2, u64::MAX]],
        all_to_all_attrs(0, 1, 2),
    );
}

#[test]
fn convolution_handles_padding_dilation_layouts_and_groups() {
    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs {
        padding,
        window_strides,
        ..
    } = &mut attrs
    {
        *padding = vec![vec![-1, 2], vec![1, -2]];
        *window_strides = vec![2, 2];
    }
    assert_eq!(
        shape(t::Op::Convolution, &[&[2, 3, 7, 8], &[4, 3, 3, 3]], attrs),
        Inference::Known(vec![2, 4, 3, 3])
    );

    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs {
        lhs_dilation,
        rhs_dilation,
        ..
    } = &mut attrs
    {
        *lhs_dilation = vec![2, 1];
        *rhs_dilation = vec![1, 2];
    }
    assert_eq!(
        shape(t::Op::Convolution, &[&[2, 3, 7, 8], &[4, 3, 3, 3]], attrs),
        Inference::Known(vec![2, 4, 11, 4])
    );

    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs {
        input_feature_dimension,
        input_spatial_dimensions,
        kernel_input_feature_dimension,
        kernel_output_feature_dimension,
        kernel_spatial_dimensions,
        output_batch_dimension,
        output_feature_dimension,
        output_spatial_dimensions,
        window_reversal,
        ..
    } = &mut attrs
    {
        // NHWC input, HWIO kernel, HWCN output: none of the layouts is assumed.
        *input_feature_dimension = 3;
        *input_spatial_dimensions = vec![1, 2];
        *kernel_input_feature_dimension = 2;
        *kernel_output_feature_dimension = 3;
        *kernel_spatial_dimensions = vec![0, 1];
        *output_batch_dimension = 3;
        *output_feature_dimension = 2;
        *output_spatial_dimensions = vec![0, 1];
        *window_reversal = vec![true, false];
    }
    assert_eq!(
        shape(t::Op::Convolution, &[&[2, 7, 8, 3], &[3, 3, 3, 4]], attrs),
        Inference::Known(vec![5, 6, 4, 2])
    );

    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs {
        feature_group_count,
        ..
    } = &mut attrs
    {
        *feature_group_count = 2;
    }
    assert_eq!(
        shape(t::Op::Convolution, &[&[2, 6, 7, 8], &[4, 3, 3, 3]], attrs),
        Inference::Known(vec![2, 4, 5, 6])
    );

    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs {
        batch_group_count, ..
    } = &mut attrs
    {
        *batch_group_count = 2;
    }
    assert_eq!(
        shape(t::Op::Convolution, &[&[4, 3, 7, 8], &[4, 3, 3, 3]], attrs),
        Inference::Known(vec![2, 4, 5, 6])
    );
}

#[test]
fn convolution_checks_each_schema_list_axes_and_group_constraints() {
    // Invalid mutations exercise the separate assertions in the TEPL block.
    for case in 0..19 {
        let mut attrs = convolution_attrs();
        let t::OpAttrs::ConvolutionAttrs {
            window_strides,
            padding,
            lhs_dilation,
            rhs_dilation,
            window_reversal,
            input_batch_dimension,
            input_spatial_dimensions,
            kernel_spatial_dimensions,
            output_spatial_dimensions,
            feature_group_count,
            batch_group_count,
            ..
        } = &mut attrs
        else {
            unreachable!()
        };
        match case {
            0 => input_spatial_dimensions.pop().map(|_| ()).unwrap(),
            1 => kernel_spatial_dimensions.clear(),
            2 => output_spatial_dimensions.clear(),
            3 => *input_batch_dimension = 1,
            4 => kernel_spatial_dimensions[0] = u64::MAX,
            5 => output_spatial_dimensions[0] = 1,
            6 => window_strides.clear(),
            7 => padding.clear(),
            8 => padding[0].pop().map(|_| ()).unwrap(),
            9 => lhs_dilation.clear(),
            10 => rhs_dilation.clear(),
            11 => window_reversal.clear(),
            12 => window_strides[0] = 0,
            13 => lhs_dilation[0] = 0,
            14 => rhs_dilation[0] = 0,
            15 => *feature_group_count = 0,
            16 => *batch_group_count = 0,
            17 => {
                *feature_group_count = 2;
                *batch_group_count = 2;
            }
            18 => *batch_group_count = 3,
            _ => unreachable!(),
        }
        invalid(t::Op::Convolution, &[&[2, 3, 7, 8], &[4, 3, 3, 3]], attrs);
    }
    invalid(t::Op::Convolution, &[&[2], &[3]], convolution_attrs());
    invalid(
        t::Op::Convolution,
        &[&[2, 3, 7, 8], &[4, 3, 3]],
        convolution_attrs(),
    );
    invalid(
        t::Op::Convolution,
        &[&[2, 3, 7, 8], &[4, 2, 3, 3]],
        convolution_attrs(),
    );
    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs {
        feature_group_count,
        ..
    } = &mut attrs
    {
        *feature_group_count = 2;
    }
    invalid(t::Op::Convolution, &[&[2, 6, 7, 8], &[3, 3, 3, 3]], attrs);
}

#[test]
fn convolution_empty_windows_and_checked_integer_boundaries_follow_tepl() {
    assert_eq!(
        shape(
            t::Op::Convolution,
            &[&[2, 3, 0, 8], &[4, 3, 3, 3]],
            convolution_attrs()
        ),
        Inference::Known(vec![2, 4, 0, 6])
    );
    assert_eq!(
        shape(
            t::Op::Convolution,
            &[&[2, 3, 1, 8], &[4, 3, 3, 3]],
            convolution_attrs()
        ),
        Inference::Known(vec![2, 4, 0, 6])
    );
    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs { padding, .. } = &mut attrs {
        padding[0] = vec![i64::MIN, 0];
    }
    assert_eq!(
        shape(t::Op::Convolution, &[&[2, 3, 7, 8], &[4, 3, 3, 3]], attrs),
        Inference::Known(vec![2, 4, 0, 6])
    );
    // A zero-sized input skips the enormous dilation multiplication.
    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs { lhs_dilation, .. } = &mut attrs {
        lhs_dilation[0] = u64::MAX;
    }
    assert_eq!(
        shape(
            t::Op::Convolution,
            &[&[2, 3, 0, 8], &[4, 3, 3, 3]],
            attrs.clone()
        ),
        Inference::Known(vec![2, 4, 0, 6])
    );
    invalid(
        t::Op::Convolution,
        &[&[2, 3, u64::MAX, 8], &[4, 3, 3, 3]],
        attrs,
    );
    let mut attrs = convolution_attrs();
    if let t::OpAttrs::ConvolutionAttrs { padding, .. } = &mut attrs {
        padding[0] = vec![i64::MAX, i64::MAX];
    }
    // Computation fits i128, but the output dimension cannot fit u64.
    invalid(
        t::Op::Convolution,
        &[&[2, 3, u64::MAX, 8], &[4, 3, 1, 3]],
        attrs,
    );
}

#[test]
fn generated_dtype_program_preserves_data_movement_and_checks_numeric_inputs() {
    for dtype in DType::ALL {
        let attrs = t::OpAttrs::BroadcastInDimAttrs {
            broadcast_dimensions: vec![0],
            shape: vec![2],
        }
        .into();
        assert_eq!(
            infer_dtype(t::Op::BroadcastInDim.into(), &[dtype], &attrs),
            Inference::Known(dtype)
        );
        let attrs = t::OpAttrs::ConcatenateAttrs { dimension: 0 }.into();
        assert_eq!(
            infer_dtype(t::Op::Concatenate.into(), &[dtype, dtype], &attrs),
            Inference::Known(dtype)
        );
        let attrs = t::OpAttrs::ReduceAttrs {
            dimensions: vec![0],
            body: Region(vec![]),
        }
        .into();
        assert_eq!(
            infer_dtype(t::Op::Reduce.into(), &[dtype, dtype], &attrs),
            Inference::Known(dtype)
        );
        for op in [
            t::Op::Subtract,
            t::Op::Divide,
            t::Op::Maximum,
            t::Op::Minimum,
        ] {
            let result = infer_dtype(op.into(), &[dtype, dtype], &OpAttrs::None);
            if dtype == DType::Bool {
                assert!(matches!(result, Inference::Invalid(_)));
            } else {
                assert_eq!(result, Inference::Known(dtype));
            }
        }
        for op in [t::Op::Exponential, t::Op::Log, t::Op::Sqrt, t::Op::Rsqrt] {
            let result = infer_dtype(op.into(), &[dtype], &OpAttrs::None);
            if matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
                assert_eq!(result, Inference::Known(dtype));
            } else {
                assert!(matches!(result, Inference::Invalid(_)));
            }
        }
    }
    let attrs = t::OpAttrs::ConcatenateAttrs { dimension: 0 }.into();
    assert!(matches!(
        infer_dtype(t::Op::Concatenate.into(), &[DType::F32, DType::I64], &attrs),
        Inference::Invalid(_)
    ));
    assert!(matches!(
        infer_dtype(t::Op::Concatenate.into(), &[], &attrs),
        Inference::Invalid(_)
    ));
    let mut attrs = convolution_attrs();
    assert_eq!(
        infer_dtype(
            t::Op::Convolution.into(),
            &[DType::F32, DType::F32],
            &attrs.clone().into()
        ),
        Inference::Unknown
    );
    if let t::OpAttrs::ConvolutionAttrs {
        precision_config, ..
    } = &mut attrs
    {
        *precision_config = vec![Precision::High; 2];
    }
    assert_eq!(
        infer_dtype(
            t::Op::Convolution.into(),
            &[DType::F32, DType::F32],
            &attrs.into()
        ),
        Inference::Unknown
    );
}
