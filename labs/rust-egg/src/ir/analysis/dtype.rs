//! Simple Rust dtype policy for the reference dialects, maintained outside TEPL.
//! Data movement preserves dtype; arithmetic requires matching numeric dtypes.
//! Transcendental operations require floating-point inputs. Precision overrides
//! and unsupported policies return Unknown instead of guessing a result type.
use super::super::dialects::{scalar, tensor_lang as t};
use super::super::types::Precision;
use super::super::{DType, Op, OpAttrs};
use super::Inference;

pub fn infer_dtype(op: Op, operands: &[DType], attrs: &OpAttrs) -> Inference<DType> {
    if op == Op::Input {
        return Inference::Unknown;
    }
    if !op.arity().accepts(operands.len()) || !op.accepts_attrs(attrs) {
        return Inference::Invalid("operand count or attributes do not match the dtype policy");
    }
    match op {
        Op::Literal => {
            let OpAttrs::Literal { dtype, .. } = attrs else {
                unreachable!()
            };
            dtype.map(Inference::Known).unwrap_or(Inference::Unknown)
        }
        Op::Scalar(scalar::Op::Add | scalar::Op::Negate) => numeric(operands),
        Op::TensorLang(op) => match op {
            t::Op::Add
            | t::Op::Subtract
            | t::Op::Multiply
            | t::Op::Divide
            | t::Op::Maximum
            | t::Op::Minimum
            | t::Op::Negate
            | t::Op::Abs => numeric(operands),
            t::Op::Exponential | t::Op::Log | t::Op::Sqrt | t::Op::Rsqrt => {
                match common(operands) {
                    Inference::Known(dtype)
                        if matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) =>
                    {
                        Inference::Known(dtype)
                    }
                    Inference::Known(_) => Inference::Invalid(
                        "transcendental operations require floating-point operands",
                    ),
                    result => result,
                }
            }
            t::Op::Reshape
            | t::Op::Transpose
            | t::Op::BroadcastInDim
            | t::Op::Slice
            | t::Op::Concatenate
            | t::Op::Reduce
            | t::Op::AllGather
            | t::Op::AllReduce
            | t::Op::ReduceScatter
            | t::Op::AllToAll => common(operands),
            t::Op::DotGeneral => {
                let OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
                    precision_config,
                    algorithm,
                    ..
                }) = attrs
                else {
                    unreachable!()
                };
                if algorithm.is_some() || precision_config.as_slice() != [Precision::Default; 2] {
                    Inference::Unknown
                } else {
                    numeric(operands)
                }
            }
            t::Op::Convolution => {
                let OpAttrs::TensorLang(t::OpAttrs::ConvolutionAttrs {
                    precision_config, ..
                }) = attrs
                else {
                    unreachable!()
                };
                if precision_config.as_slice() != [Precision::Default; 2] {
                    Inference::Unknown
                } else {
                    numeric(operands)
                }
            }
            t::Op::Constant => {
                let OpAttrs::TensorLang(t::OpAttrs::ConstantAttrs { value }) = attrs else {
                    unreachable!()
                };
                value
                    .element_type
                    .parse()
                    .map(Inference::Known)
                    .unwrap_or(Inference::Unknown)
            }
        },
        Op::Input => unreachable!(),
    }
}

fn common(operands: &[DType]) -> Inference<DType> {
    let Some(&first) = operands.first() else {
        return Inference::Invalid("dtype policy requires at least one operand");
    };
    if operands.iter().all(|&dtype| dtype == first) {
        Inference::Known(first)
    } else {
        Inference::Invalid("operand dtypes must match")
    }
}

fn numeric(operands: &[DType]) -> Inference<DType> {
    match common(operands) {
        Inference::Known(DType::Bool) => {
            Inference::Invalid("arithmetic does not accept bool operands")
        }
        result => result,
    }
}
