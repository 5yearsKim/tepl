//! Infer one operation, without accessing an e-graph.
use super::super::pattern::TensorInfo;
use super::super::{Op, OpAttrs};
use super::{Inference, infer_dtype, infer_shape};

/// Used by TensorAnalysis: preserve Known, Unknown, and Invalid results.
pub fn infer_tensor(op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Inference<TensorInfo> {
    if !op.arity().accepts(operands.len()) || !op.accepts_attrs(attrs) {
        return Inference::Invalid(
            "operand count or attributes do not match the operation signature",
        );
    }
    if let Some(result) = super::payload::infer_payload(op, attrs) {
        return result;
    }

    // Reusable combination of pure shape inference and the Rust dtype policy.
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
