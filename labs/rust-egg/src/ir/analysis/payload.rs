//! Reference dialect policy for typed payloads, separate from TEPL shape blocks.
//! Future generation must select these policies explicitly for payload-bearing
//! operations. This is the only combined-inference exception for an operation
//! without a shape block; process-grid-dependent collectives remain Unknown.
use super::super::dialects::tensor_lang as t;
use super::super::{Op, OpAttrs};
use super::{Inference, TensorInfo, infer_dtype};

pub(super) fn infer_payload(op: Op, attrs: &OpAttrs) -> Option<Inference<TensorInfo>> {
    let (Op::TensorLang(t::Op::Constant), OpAttrs::TensorLang(t::OpAttrs::ConstantAttrs { value })) =
        (op, attrs)
    else {
        return None;
    };
    Some(match infer_dtype(op, &[], attrs) {
        Inference::Known(dtype) => Inference::Known(TensorInfo {
            shape: value.shape.clone(),
            dtype,
        }),
        Inference::Unknown => Inference::Unknown,
        Inference::Invalid(reason) => Inference::Invalid(reason),
    })
}
