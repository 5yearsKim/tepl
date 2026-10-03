//! Handwritten sample of code to be emitted from TEPL shape definitions.
//! Covers add, multiply, negate, reshape, transpose, scalar ops, and dot_general.
//! Other operations intentionally return Unknown. No host or e-graph dependency.

use super::super::dialects::{scalar, tensor_lang as t};
use super::super::{Op, OpAttrs};
use super::Inference;
use super::shape_builtins::{self as shape, ShapeResult};

pub fn infer_shape(op: Op, operands: &[&[u64]], attrs: &OpAttrs) -> Inference<Vec<u64>> {
    let result = match (op, operands, attrs) {
        (Op::Literal, [], OpAttrs::Literal { .. }) => Ok(vec![]),
        (Op::TensorLang(t::Op::Add), [lhs, rhs], OpAttrs::None) => add_shape(lhs, rhs),
        (Op::TensorLang(t::Op::Multiply), [lhs, rhs], OpAttrs::None) => multiply_shape(lhs, rhs),
        (Op::TensorLang(t::Op::Negate), [input], OpAttrs::None) => Ok(input.to_vec()),
        (Op::Scalar(scalar::Op::Add), [lhs, rhs], OpAttrs::None) => scalar_add_shape(lhs, rhs),
        (Op::Scalar(scalar::Op::Negate), [input], OpAttrs::None) => scalar_negate_shape(input),
        (
            Op::TensorLang(t::Op::Reshape),
            [input],
            OpAttrs::TensorLang(t::OpAttrs::ReshapeAttrs { shape }),
        ) => reshape_shape(input, shape),
        (
            Op::TensorLang(t::Op::Transpose),
            [input],
            OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs { permutation }),
        ) => transpose_shape(input, permutation),
        (
            Op::TensorLang(t::Op::DotGeneral),
            [lhs, rhs],
            OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
                lhs_contracting_dimensions: lc,
                rhs_contracting_dimensions: rc,
                lhs_batching_dimensions: lb,
                rhs_batching_dimensions: rb,
                ..
            }),
        ) => dot_general_shape(lhs, rhs, lc, rc, lb, rb),
        _ if matches!(
            op,
            Op::Literal
                | Op::Scalar(_)
                | Op::TensorLang(
                    t::Op::Add
                        | t::Op::Multiply
                        | t::Op::Negate
                        | t::Op::Reshape
                        | t::Op::Transpose
                        | t::Op::DotGeneral
                )
        ) =>
        {
            return Inference::Invalid(
                "operand count or attributes do not match the shape definition",
            );
        }
        _ => return Inference::Unknown,
    };
    match result {
        Ok(shape) => Inference::Known(shape),
        Err(error) => Inference::Invalid(error.message()),
    }
}

fn add_shape(lhs: &[u64], rhs: &[u64]) -> ShapeResult<Vec<u64>> {
    shape::ensure(lhs == rhs, "elementwise operand shapes must match")?;
    Ok(lhs.to_vec())
}

fn multiply_shape(lhs: &[u64], rhs: &[u64]) -> ShapeResult<Vec<u64>> {
    shape::ensure(lhs == rhs, "elementwise operand shapes must match")?;
    Ok(lhs.to_vec())
}

fn scalar_add_shape(lhs: &[u64], rhs: &[u64]) -> ShapeResult<Vec<u64>> {
    shape::ensure(
        lhs.is_empty() && rhs.is_empty(),
        "scalar add requires rank-zero operands",
    )?;
    Ok(vec![])
}

fn scalar_negate_shape(input: &[u64]) -> ShapeResult<Vec<u64>> {
    shape::ensure(
        input.is_empty(),
        "scalar negate requires a rank-zero operand",
    )?;
    Ok(vec![])
}

fn reshape_shape(input: &[u64], output: &[u64]) -> ShapeResult<Vec<u64>> {
    shape::ensure(
        shape::product(input)? == shape::product(output)?,
        "reshape must preserve element count",
    )?;
    Ok(output.to_vec())
}

fn transpose_shape(input: &[u64], permutation: &[u64]) -> ShapeResult<Vec<u64>> {
    shape::ensure(
        shape::is_valid_axis_list(permutation, input.len()),
        "transpose requires a permutation of operand axes",
    )?;
    shape::ensure(
        permutation.len() == input.len(),
        "transpose requires a permutation of operand axes",
    )?;
    shape::gather(input, permutation)
}

fn dot_general_shape(
    lhs: &[u64],
    rhs: &[u64],
    lc: &[u64],
    rc: &[u64],
    lb: &[u64],
    rb: &[u64],
) -> ShapeResult<Vec<u64>> {
    shape::ensure(
        shape::is_valid_axis_list(lc, lhs.len())
            && shape::is_valid_axis_list(rc, rhs.len())
            && shape::is_valid_axis_list(lb, lhs.len())
            && shape::is_valid_axis_list(rb, rhs.len())
            && shape::is_disjoint(lb, lc)
            && shape::is_disjoint(rb, rc)
            && lb.len() == rb.len()
            && lc.len() == rc.len(),
        "invalid dot dimension numbers",
    )?;
    shape::ensure(
        shape::gather(lhs, lb)? == shape::gather(rhs, rb)?
            && shape::gather(lhs, lc)? == shape::gather(rhs, rc)?,
        "dot batch and contracting dimensions must match",
    )?;
    let mut output = shape::gather(lhs, lb)?;
    for (input, batch, contract) in [(lhs, lb, lc), (rhs, rb, rc)] {
        output.extend(input.iter().enumerate().filter_map(|(axis, &dim)| {
            let axis = axis as u64;
            (!batch.contains(&axis) && !contract.contains(&axis)).then_some(dim)
        }));
    }
    Ok(output)
}
