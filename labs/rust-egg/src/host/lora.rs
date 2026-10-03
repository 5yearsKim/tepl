use super::{batched_dot_shape, dot_attrs};
use crate::ir::OpAttrs;
use crate::ir::pattern::TensorInfo;
use crate::ir::rules::lora::rule_lora;
pub struct DemoLoraFunctions {
    pub allow_reassociation: bool,
}

impl rule_lora::Functions for DemoLoraFunctions {
    fn broadcastable(&self, batch: &[u64], weight_batch: &[u64]) -> Option<bool> {
        Some(batch == weight_batch)
    }

    fn reassociable(
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
