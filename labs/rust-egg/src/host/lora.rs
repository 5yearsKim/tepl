//! LoRA-specific rewrite conditions and rank-three dot helpers.
use crate::ir::TensorInfo;
use crate::ir::analysis::infer_shape;
use crate::ir::dialects::tensor_lang;
use crate::ir::rules::lora::rule_lora;
use crate::ir::{Op, OpAttrs};
pub struct DemoLoraFunctions {
    pub allow_reassociation: bool,
}

impl rule_lora::HostFunctions for DemoLoraFunctions {
    fn is_broadcastable(&self, batch: &[u64], weight_batch: &[u64]) -> Option<bool> {
        Some(batch == weight_batch)
    }

    fn is_reassociable(
        &self,
        _x: &TensorInfo,
        _a: &TensorInfo,
        _b: &TensorInfo,
        outer: &OpAttrs,
        inner: &OpAttrs,
    ) -> Option<bool> {
        Some(self.allow_reassociation && *outer == dot_attrs() && *inner == dot_attrs())
    }

    fn infer_dot(&self, lhs: &TensorInfo, rhs: &TensorInfo, attrs: &OpAttrs) -> Option<OpAttrs> {
        batched_dot_shape(&lhs.shape, &rhs.shape, attrs)?;
        Some(attrs.clone())
    }
    fn infer_lora_out(
        &self,
        x: &TensorInfo,
        a: &TensorInfo,
        b: &TensorInfo,
        outer: &OpAttrs,
        inner: &OpAttrs,
    ) -> Option<OpAttrs> {
        // This demo supports equal rank-three batches and fixed matrix axes.
        // The final dot has the inner dot's axes under those conventions.
        if x.shape.len() != 3
            || a.shape.len() != 3
            || b.shape.len() != 3
            || *outer != dot_attrs()
            || *inner != dot_attrs()
            || x.shape[0] != a.shape[0]
            || a.shape[0] != b.shape[0]
            || x.shape[2] != a.shape[1]
            || a.shape[2] != b.shape[1]
        {
            return None;
        }
        Some(inner.clone())
    }
}

pub fn dot_attrs() -> OpAttrs {
    OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting_dimensions: vec![2],
        rhs_contracting_dimensions: vec![1],
        lhs_batching_dimensions: vec![0],
        rhs_batching_dimensions: vec![0],
        precision_config: vec![crate::ir::types::Precision::Default; 2],
        algorithm: None,
    })
}

pub fn batched_dot_shape(lhs: &[u64], rhs: &[u64], attrs: &OpAttrs) -> Option<Vec<u64>> {
    // The LoRA example specializes the more general generated shape definition.
    if lhs.len() != 3 || rhs.len() != 3 || *attrs != dot_attrs() {
        return None;
    }
    infer_shape(
        Op::TensorLang(tensor_lang::Op::DotGeneral),
        &[lhs, rhs],
        attrs,
    )
    .into_option()
}
