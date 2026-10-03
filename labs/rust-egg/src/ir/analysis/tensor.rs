//! Infer one operation, without accessing an e-graph.
use super::super::dialects::tensor_lang;
use super::super::pattern::TensorInfo;
use super::super::{Op, OpAttrs};
use super::{Inference, infer_dtype, infer_shape};

/// Used by TensorAnalysis: preserve Known, Unknown, and Invalid results.
pub fn infer_tensor(op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Inference<TensorInfo> {
    // This reference dialect stores shape and dtype in constant payloads.
    if let (
        Op::TensorLang(tensor_lang::Op::Constant),
        [],
        OpAttrs::TensorLang(tensor_lang::OpAttrs::ConstantAttrs { value }),
    ) = (op, operands, attrs)
    {
        return match value.element_type.parse() {
            Ok(dtype) => Inference::Known(TensorInfo {
                shape: value.shape.clone(),
                dtype,
            }),
            Err(_) => Inference::Unknown,
        };
    }

    // Every other operation uses the generated shape and dtype rules.
    let shapes: Vec<_> = operands.iter().map(|info| info.shape.as_slice()).collect();
    let dtypes: Vec<_> = operands.iter().map(|info| info.dtype).collect();
    match (
        infer_shape(op, &shapes, attrs),
        infer_dtype(op, &dtypes, attrs),
    ) {
        (Inference::Invalid(reason), _) | (_, Inference::Invalid(reason)) => {
            Inference::Invalid(reason)
        }
        (Inference::Known(shape), Inference::Known(dtype)) => {
            Inference::Known(TensorInfo { shape, dtype })
        }
        _ => Inference::Unknown,
    }
}

/// Used by rewrite builders: accept only known output metadata.
pub fn infer_tensor_output(op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
    infer_tensor(op, operands, attrs).into_option()
}
