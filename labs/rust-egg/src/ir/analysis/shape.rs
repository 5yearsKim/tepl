//! Reference output for every shape block in examples/dialects/*.tepl.
//! Evaluators preserve TEPL statement order and use the shared builtin library.
//! External shapes are u64; expressions use checked i128; yields validate u64.
use super::super::dialects::{scalar, tensor_lang as t};
use super::super::{Op, OpAttrs};
use super::Inference;
use super::shape_builtins::{self as shape, ShapeResult};

pub fn infer_shape(op: Op, operands: &[&[u64]], attrs: &OpAttrs) -> Inference<Vec<u64>> {
    // These operations have no TEPL shape block. Constants have a separate
    // typed-payload policy; process-grid-dependent collectives stay unknown.
    if matches!(
        op,
        Op::Input | Op::TensorLang(t::Op::Constant | t::Op::AllGather | t::Op::ReduceScatter)
    ) {
        return Inference::Unknown;
    }
    if !op.arity().accepts(operands.len()) || !op.accepts_attrs(attrs) {
        return Inference::Invalid("operand count or attributes do not match the shape definition");
    }
    let operands: Vec<_> = operands.iter().map(|s| shape::integers(s)).collect();
    let result = evaluate(op, &operands, attrs).and_then(|s| shape::dimensions(&s));
    match result {
        Ok(s) => Inference::Known(s),
        Err(error) => Inference::Invalid(error.message()),
    }
}

fn evaluate(op: Op, operands: &[Vec<i128>], attrs: &OpAttrs) -> ShapeResult<Vec<i128>> {
    match op {
        Op::Literal => Ok(vec![]),
        Op::Scalar(scalar::Op::Add) => scalar_add_shape(&operands[0], &operands[1]),
        Op::Scalar(scalar::Op::Negate) => scalar_negate_shape(&operands[0]),
        Op::TensorLang(op) => match op {
            t::Op::Add => add_shape(&operands[0], &operands[1]),
            t::Op::Subtract => subtract_shape(&operands[0], &operands[1]),
            t::Op::Multiply => multiply_shape(&operands[0], &operands[1]),
            t::Op::Divide => divide_shape(&operands[0], &operands[1]),
            t::Op::Maximum => maximum_shape(&operands[0], &operands[1]),
            t::Op::Minimum => minimum_shape(&operands[0], &operands[1]),
            t::Op::Negate => negate_shape(&operands[0]),
            t::Op::Exponential => exponential_shape(&operands[0]),
            t::Op::Log => log_shape(&operands[0]),
            t::Op::Abs => abs_shape(&operands[0]),
            t::Op::Sqrt => sqrt_shape(&operands[0]),
            t::Op::Rsqrt => rsqrt_shape(&operands[0]),
            t::Op::Reshape => {
                let OpAttrs::TensorLang(t::OpAttrs::ReshapeAttrs { shape: output }) = attrs else {
                    unreachable!()
                };
                reshape_shape(&operands[0], &shape::integers(output))
            }
            t::Op::Transpose => {
                let OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs { permutation }) = attrs else {
                    unreachable!()
                };
                transpose_shape(&operands[0], &shape::integers(permutation))
            }
            t::Op::BroadcastInDim => {
                let OpAttrs::TensorLang(t::OpAttrs::BroadcastInDimAttrs {
                    broadcast_dimensions,
                    shape: output,
                }) = attrs
                else {
                    unreachable!()
                };
                broadcast_in_dim_shape(
                    &operands[0],
                    &shape::integers(broadcast_dimensions),
                    &shape::integers(output),
                )
            }
            t::Op::Slice => {
                let OpAttrs::TensorLang(t::OpAttrs::SliceAttrs {
                    start_indices,
                    limit_indices,
                    strides,
                }) = attrs
                else {
                    unreachable!()
                };
                slice_shape(
                    &operands[0],
                    &shape::integers(start_indices),
                    &shape::integers(limit_indices),
                    &shape::integers(strides),
                )
            }
            t::Op::Concatenate => {
                let OpAttrs::TensorLang(t::OpAttrs::ConcatenateAttrs { dimension }) = attrs else {
                    unreachable!()
                };
                concatenate_shape(operands, i128::from(*dimension))
            }
            t::Op::Reduce => {
                let OpAttrs::TensorLang(t::OpAttrs::ReduceAttrs { dimensions, .. }) = attrs else {
                    unreachable!()
                };
                reduce_shape(&operands[0], &operands[1], &shape::integers(dimensions))
            }
            t::Op::DotGeneral => {
                let OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
                    lhs_contracting_dimensions,
                    rhs_contracting_dimensions,
                    lhs_batching_dimensions,
                    rhs_batching_dimensions,
                    ..
                }) = attrs
                else {
                    unreachable!()
                };
                dot_general_shape(
                    &operands[0],
                    &operands[1],
                    &shape::integers(lhs_contracting_dimensions),
                    &shape::integers(rhs_contracting_dimensions),
                    &shape::integers(lhs_batching_dimensions),
                    &shape::integers(rhs_batching_dimensions),
                )
            }
            t::Op::Convolution => {
                let OpAttrs::TensorLang(attrs) = attrs else {
                    unreachable!()
                };
                convolution_shape(&operands[0], &operands[1], attrs)
            }
            t::Op::AllReduce => all_reduce_shape(&operands[0]),
            t::Op::AllToAll => {
                let OpAttrs::TensorLang(t::OpAttrs::AllToAllAttrs {
                    split_dimension,
                    concat_dimension,
                    split_count,
                    ..
                }) = attrs
                else {
                    unreachable!()
                };
                all_to_all_shape(
                    &operands[0],
                    i128::from(*split_dimension),
                    i128::from(*concat_dimension),
                    i128::from(*split_count),
                )
            }
            // The public dispatcher returns Unknown before reaching these arms.
            t::Op::Constant | t::Op::AllGather | t::Op::ReduceScatter => unreachable!(),
        },
        Op::Input => unreachable!(),
    }
}

fn add_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == r, "add: operand shapes must match")?;
    Ok(l.to_vec())
}

fn subtract_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == r, "subtract: operand shapes must match")?;
    Ok(l.to_vec())
}

fn multiply_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == r, "multiply: operand shapes must match")?;
    Ok(l.to_vec())
}

fn divide_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == r, "divide: operand shapes must match")?;
    Ok(l.to_vec())
}

fn maximum_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == r, "maximum: operand shapes must match")?;
    Ok(l.to_vec())
}

fn minimum_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == r, "minimum: operand shapes must match")?;
    Ok(l.to_vec())
}

fn negate_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn exponential_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn log_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn abs_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn sqrt_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn rsqrt_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn all_reduce_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    Ok(s.to_vec())
}

fn scalar_add_shape(l: &[i128], r: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(l == [] && r == [], "scalar add requires rank-zero operands")?;
    Ok(vec![])
}

fn scalar_negate_shape(s: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(s == [], "scalar negate requires a rank-zero operand")?;
    Ok(vec![])
}

fn reshape_shape(s: &[i128], output: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(
        shape::product(s)? == shape::product(output)?,
        "reshape must preserve element count",
    )?;
    Ok(output.to_vec())
}

fn transpose_shape(s: &[i128], permutation: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(
        shape::is_valid_axis_list(permutation, shape::len(s)),
        "transpose: invalid axes",
    )?;
    shape::ensure(
        shape::len(permutation) == shape::len(s),
        "transpose: rank mismatch",
    )?;
    shape::gather(s, permutation)
}

fn broadcast_in_dim_shape(
    s: &[i128],
    broadcast_dimensions: &[i128],
    output: &[i128],
) -> ShapeResult<Vec<i128>> {
    shape::ensure(
        shape::len(broadcast_dimensions) == shape::len(s),
        "broadcast: mapping rank mismatch",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(broadcast_dimensions, shape::len(output)),
        "broadcast: invalid axes",
    )?;
    let compatible = shape::range(shape::len(s))?
        .iter()
        .map(|&i| -> ShapeResult<bool> {
            // || is short-circuit, just as in TEPL.
            Ok(*shape::index(s, i)? == 1
                || *shape::index(s, i)?
                    == *shape::index(output, *shape::index(broadcast_dimensions, i)?)?)
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    shape::ensure(
        shape::all(&compatible),
        "broadcast: incompatible dimensions",
    )?;
    Ok(output.to_vec())
}

fn slice_shape(
    s: &[i128],
    start_indices: &[i128],
    limit_indices: &[i128],
    strides: &[i128],
) -> ShapeResult<Vec<i128>> {
    shape::ensure(
        shape::len(start_indices) == shape::len(s),
        "slice: start rank mismatch",
    )?;
    shape::ensure(
        shape::len(limit_indices) == shape::len(s),
        "slice: limit rank mismatch",
    )?;
    shape::ensure(
        shape::len(strides) == shape::len(s),
        "slice: stride rank mismatch",
    )?;
    let valid = shape::range(shape::len(s))?
        .iter()
        .map(|&i| -> ShapeResult<bool> {
            Ok(
                shape::index(start_indices, i)? <= shape::index(limit_indices, i)?
                    && shape::index(limit_indices, i)? <= shape::index(s, i)?
                    && *shape::index(strides, i)? > 0,
            )
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    shape::ensure(shape::all(&valid), "slice: invalid bounds or stride")?;
    shape::range(shape::len(s))?
        .iter()
        .map(|&i| {
            shape::ceil_div(
                shape::sub(
                    *shape::index(limit_indices, i)?,
                    *shape::index(start_indices, i)?,
                )?,
                *shape::index(strides, i)?,
            )
        })
        .collect()
}

fn concatenate_shape(inputs: &[Vec<i128>], axis: i128) -> ShapeResult<Vec<i128>> {
    shape::ensure(shape::len(inputs) > 0, "concatenate requires an input")?;
    let first = shape::index(inputs, 0)?;
    shape::ensure(
        shape::is_valid_axis_list(&[axis], shape::len(first)),
        "concatenate: invalid axis",
    )?;
    let ranks: Vec<_> = inputs
        .iter()
        .map(|s| shape::len(s) == shape::len(first))
        .collect();
    shape::ensure(shape::all(&ranks), "concatenate: rank mismatch")?;
    let other_axes = shape::exclude(&shape::integers(&shape::range(shape::len(first))?), &[axis]);
    let compatible = inputs
        .iter()
        .map(|s| -> ShapeResult<bool> {
            Ok(shape::gather(s, &other_axes)? == shape::gather(first, &other_axes)?)
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    shape::ensure(
        shape::all(&compatible),
        "concatenate: other dimensions must match",
    )?;
    let sizes = inputs
        .iter()
        .map(|s| shape::index(s, axis).copied())
        .collect::<ShapeResult<Vec<_>>>()?;
    shape::replace(first, axis, shape::sum(&sizes)?)
}

fn reduce_shape(s: &[i128], init: &[i128], dimensions: &[i128]) -> ShapeResult<Vec<i128>> {
    shape::ensure(init == [], "reduce requires a scalar initializer")?;
    shape::ensure(
        shape::is_valid_axis_list(dimensions, shape::len(s)),
        "reduce: invalid axes",
    )?;
    shape::gather(
        s,
        &shape::exclude(&shape::integers(&shape::range(shape::len(s))?), dimensions),
    )
}

fn dot_general_shape(
    l: &[i128],
    r: &[i128],
    lc: &[i128],
    rc: &[i128],
    lb: &[i128],
    rb: &[i128],
) -> ShapeResult<Vec<i128>> {
    shape::ensure(
        shape::is_valid_axis_list(lc, shape::len(l)),
        "dot: invalid lhs contracting axes",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(rc, shape::len(r)),
        "dot: invalid rhs contracting axes",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(lb, shape::len(l)),
        "dot: invalid lhs batch axes",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(rb, shape::len(r)),
        "dot: invalid rhs batch axes",
    )?;
    shape::ensure(shape::is_disjoint(lb, lc), "dot: overlapping lhs axes")?;
    shape::ensure(shape::is_disjoint(rb, rc), "dot: overlapping rhs axes")?;
    shape::ensure(
        shape::len(lb) == shape::len(rb),
        "dot: batch axis count mismatch",
    )?;
    shape::ensure(
        shape::len(lc) == shape::len(rc),
        "dot: contracting axis count mismatch",
    )?;
    shape::ensure(
        shape::gather(l, lb)? == shape::gather(r, rb)?,
        "dot: batch dimensions must match",
    )?;
    shape::ensure(
        shape::gather(l, lc)? == shape::gather(r, rc)?,
        "dot: contracting dimensions must match",
    )?;
    let lf = shape::exclude(
        &shape::integers(&shape::range(shape::len(l))?),
        &shape::concat(&[lb, lc])?,
    );
    let rf = shape::exclude(
        &shape::integers(&shape::range(shape::len(r))?),
        &shape::concat(&[rb, rc])?,
    );
    shape::concat(&[
        &shape::gather(l, lb)?,
        &shape::gather(l, &lf)?,
        &shape::gather(r, &rf)?,
    ])
}

fn all_to_all_shape(s: &[i128], split: i128, join: i128, count: i128) -> ShapeResult<Vec<i128>> {
    shape::ensure(
        shape::is_valid_axis_list(&[split], shape::len(s)),
        "all_to_all: invalid split axis",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(&[join], shape::len(s)),
        "all_to_all: invalid concatenate axis",
    )?;
    shape::ensure(count > 0, "all_to_all: split count must be positive")?;
    shape::ensure(
        shape::rem(*shape::index(s, split)?, count)? == 0,
        "all_to_all: split dimension must be divisible",
    )?;
    let divided = shape::replace(s, split, shape::div(*shape::index(s, split)?, count)?)?;
    shape::replace(
        &divided,
        join,
        shape::mul(*shape::index(&divided, join)?, count)?,
    )
}

fn convolution_shape(l: &[i128], r: &[i128], attrs: &t::OpAttrs) -> ShapeResult<Vec<i128>> {
    let t::OpAttrs::ConvolutionAttrs {
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        window_reversal,
        input_batch_dimension,
        input_feature_dimension,
        input_spatial_dimensions,
        kernel_input_feature_dimension,
        kernel_output_feature_dimension,
        kernel_spatial_dimensions,
        output_batch_dimension,
        output_feature_dimension,
        output_spatial_dimensions,
        feature_group_count,
        batch_group_count,
        ..
    } = attrs
    else {
        unreachable!()
    };
    // Schema boundary conversions. All subsequent arithmetic is checked i128.
    let window_strides = shape::integers(window_strides);
    let padding: Vec<Vec<i128>> = padding
        .iter()
        .map(|p| p.iter().copied().map(i128::from).collect())
        .collect();
    let lhs_dilation = shape::integers(lhs_dilation);
    let rhs_dilation = shape::integers(rhs_dilation);
    let input_batch_dimension = i128::from(*input_batch_dimension);
    let input_feature_dimension = i128::from(*input_feature_dimension);
    let input_spatial_dimensions = shape::integers(input_spatial_dimensions);
    let kernel_input_feature_dimension = i128::from(*kernel_input_feature_dimension);
    let kernel_output_feature_dimension = i128::from(*kernel_output_feature_dimension);
    let kernel_spatial_dimensions = shape::integers(kernel_spatial_dimensions);
    let output_batch_dimension = i128::from(*output_batch_dimension);
    let output_feature_dimension = i128::from(*output_feature_dimension);
    let output_spatial_dimensions = shape::integers(output_spatial_dimensions);
    let feature_group_count = i128::from(*feature_group_count);
    let batch_group_count = i128::from(*batch_group_count);

    // One statement for each TEPL statement, in source order.
    let n = shape::len(l);
    shape::ensure(
        n >= 2 && shape::len(r) == n,
        "convolution: invalid operand ranks",
    )?;
    let spatial_rank = n - 2; // n >= 2 was established above.
    let axes = shape::range(spatial_rank)?;
    let is = input_spatial_dimensions;
    let ks = kernel_spatial_dimensions;
    let os = output_spatial_dimensions;
    shape::ensure(
        shape::len(&is) == spatial_rank,
        "convolution: input spatial rank mismatch",
    )?;
    shape::ensure(
        shape::len(&ks) == spatial_rank,
        "convolution: kernel spatial rank mismatch",
    )?;
    shape::ensure(
        shape::len(&os) == spatial_rank,
        "convolution: output spatial rank mismatch",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(
            &shape::concat(&[&[input_batch_dimension], &is, &[input_feature_dimension]])?,
            n,
        ),
        "convolution: invalid input axes",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(
            &shape::concat(&[
                &[kernel_input_feature_dimension],
                &ks,
                &[kernel_output_feature_dimension],
            ])?,
            n,
        ),
        "convolution: invalid kernel axes",
    )?;
    shape::ensure(
        shape::is_valid_axis_list(
            &shape::concat(&[&[output_batch_dimension], &os, &[output_feature_dimension]])?,
            n,
        ),
        "convolution: invalid output axes",
    )?;
    shape::ensure(
        shape::len(&window_strides) == spatial_rank,
        "convolution: stride rank mismatch",
    )?;
    shape::ensure(
        shape::len(&padding) == spatial_rank,
        "convolution: padding rank mismatch",
    )?;
    let padding_pairs: Vec<_> = padding.iter().map(|p| shape::len(p) == 2).collect();
    shape::ensure(
        shape::all(&padding_pairs),
        "convolution: padding must contain pairs",
    )?;
    shape::ensure(
        shape::len(&lhs_dilation) == spatial_rank,
        "convolution: input dilation rank mismatch",
    )?;
    shape::ensure(
        shape::len(&rhs_dilation) == spatial_rank,
        "convolution: kernel dilation rank mismatch",
    )?;
    shape::ensure(
        shape::len(window_reversal) == spatial_rank,
        "convolution: reversal rank mismatch",
    )?;
    let positive = axes
        .iter()
        .map(|&i| -> ShapeResult<bool> {
            Ok(*shape::index(&window_strides, i)? > 0
                && *shape::index(&lhs_dilation, i)? > 0
                && *shape::index(&rhs_dilation, i)? > 0)
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    shape::ensure(
        shape::all(&positive),
        "convolution: strides and dilations must be positive",
    )?;
    let fg = feature_group_count;
    let bg = batch_group_count;
    shape::ensure(
        fg > 0 && bg > 0,
        "convolution: group counts must be positive",
    )?;
    shape::ensure(
        fg == 1 || bg == 1,
        "convolution: only one group count may exceed one",
    )?;
    shape::ensure(
        shape::rem(*shape::index(l, input_batch_dimension)?, bg)? == 0,
        "convolution: batch group mismatch",
    )?;
    shape::ensure(
        shape::rem(*shape::index(l, input_feature_dimension)?, fg)? == 0,
        "convolution: feature group mismatch",
    )?;
    shape::ensure(
        *shape::index(r, kernel_input_feature_dimension)?
            == shape::div(*shape::index(l, input_feature_dimension)?, fg)?,
        "convolution: kernel input feature mismatch",
    )?;
    shape::ensure(
        shape::rem(*shape::index(r, kernel_output_feature_dimension)?, bg)? == 0,
        "convolution: kernel output batch group mismatch",
    )?;
    shape::ensure(
        shape::rem(*shape::index(r, kernel_output_feature_dimension)?, fg)? == 0,
        "convolution: kernel output feature group mismatch",
    )?;
    let dilated_input = axes
        .iter()
        .map(|&i| -> ShapeResult<i128> {
            if *shape::index(l, *shape::index(&is, i)?)? == 0 {
                Ok(0)
            } else {
                shape::add(
                    shape::mul(
                        shape::sub(*shape::index(l, *shape::index(&is, i)?)?, 1)?,
                        *shape::index(&lhs_dilation, i)?,
                    )?,
                    1,
                )
            }
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    let padded_input = axes
        .iter()
        .map(|&i| {
            shape::add(
                shape::add(
                    *shape::index(shape::index(&padding, i)?, 0)?,
                    *shape::index(&dilated_input, i)?,
                )?,
                *shape::index(shape::index(&padding, i)?, 1)?,
            )
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    let window = axes
        .iter()
        .map(|&i| -> ShapeResult<i128> {
            if *shape::index(r, *shape::index(&ks, i)?)? == 0 {
                Ok(0)
            } else {
                shape::add(
                    shape::mul(
                        shape::sub(*shape::index(r, *shape::index(&ks, i)?)?, 1)?,
                        *shape::index(&rhs_dilation, i)?,
                    )?,
                    1,
                )
            }
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    let windows = axes
        .iter()
        .map(|&i| -> ShapeResult<i128> {
            if *shape::index(&padded_input, i)? == 0
                || shape::index(&window, i)? > shape::index(&padded_input, i)?
            {
                Ok(0)
            } else {
                shape::add(
                    shape::floor_div(
                        shape::sub(*shape::index(&padded_input, i)?, *shape::index(&window, i)?)?,
                        *shape::index(&window_strides, i)?,
                    )?,
                    1,
                )
            }
        })
        .collect::<ShapeResult<Vec<_>>>()?;
    shape::range(n)?
        .iter()
        .map(|&axis| -> ShapeResult<i128> {
            if i128::from(axis) == output_batch_dimension {
                shape::div(*shape::index(l, input_batch_dimension)?, bg)
            } else if i128::from(axis) == output_feature_dimension {
                Ok(*shape::index(r, kernel_output_feature_dimension)?)
            } else {
                let selected = axes
                    .iter()
                    .map(|&i| -> ShapeResult<i128> {
                        if *shape::index(&os, i)? == i128::from(axis) {
                            Ok(*shape::index(&windows, i)?)
                        } else {
                            Ok(0)
                        }
                    })
                    .collect::<ShapeResult<Vec<_>>>()?;
                shape::sum(&selected)
            }
        })
        .collect()
}
