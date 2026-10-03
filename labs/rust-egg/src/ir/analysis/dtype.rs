//! Handwritten reference for future generated dtype rules.
//!
//! Sample contract: arithmetic preserves matching non-bool operand types;
//! reshape/transpose preserve any type; literals use their explicit type.
//! Dot supports only default precision with no algorithm override. Other
//! operations/settings return Unknown. These rules are not yet encoded in TEPL.

use super::super::dialects::{scalar, tensor_lang as t};
use super::super::types::Precision;
use super::super::{DType, Op, OpAttrs};
use super::Inference;

pub fn infer_dtype(op: Op, operands: &[DType], attrs: &OpAttrs) -> Inference<DType> {
    match (op, operands, attrs) {
        (Op::Literal, [], OpAttrs::Literal { dtype, .. }) => {
            dtype.map(Inference::Known).unwrap_or(Inference::Unknown)
        }
        (Op::TensorLang(t::Op::Add | t::Op::Multiply), [lhs, rhs], OpAttrs::None)
        | (Op::Scalar(scalar::Op::Add), [lhs, rhs], OpAttrs::None) => {
            binary_arithmetic_dtype(*lhs, *rhs)
        }
        (Op::TensorLang(t::Op::Negate), [input], OpAttrs::None)
        | (Op::Scalar(scalar::Op::Negate), [input], OpAttrs::None) => arithmetic_dtype(*input),
        (
            Op::TensorLang(t::Op::Reshape),
            [input],
            OpAttrs::TensorLang(t::OpAttrs::ReshapeAttrs { .. }),
        )
        | (
            Op::TensorLang(t::Op::Transpose),
            [input],
            OpAttrs::TensorLang(t::OpAttrs::TransposeAttrs { .. }),
        ) => Inference::Known(*input),
        (
            Op::TensorLang(t::Op::DotGeneral),
            [lhs, rhs],
            OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
                precision_config,
                algorithm,
                ..
            }),
        ) => {
            if algorithm.is_some() || precision_config.as_slice() != [Precision::Default; 2] {
                return Inference::Unknown;
            }
            binary_arithmetic_dtype(*lhs, *rhs)
        }
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
            Inference::Invalid("operand count or attributes do not match the dtype definition")
        }
        _ => Inference::Unknown,
    }
}

fn arithmetic_dtype(dtype: DType) -> Inference<DType> {
    if dtype == DType::Bool {
        Inference::Invalid("sample arithmetic does not accept bool operands")
    } else {
        Inference::Known(dtype)
    }
}

fn binary_arithmetic_dtype(lhs: DType, rhs: DType) -> Inference<DType> {
    if lhs != rhs {
        Inference::Invalid("operand dtypes must match")
    } else {
        arithmetic_dtype(lhs)
    }
}
