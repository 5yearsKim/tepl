// Generated host functions. Keep user implementations in a separate file.
use rust_egg::ir::OpAttrs;
use rust_egg::ir::patterns::{InferredTensor, TensorInfo};

pub trait HostFunctions: Send + Sync {
    fn broadcastable(&self, arg0: &[usize], arg1: &[usize]) -> Option<bool>;
    fn infer_dot(
        &self,
        arg0: &TensorInfo,
        arg1: &TensorInfo,
        arg2: &OpAttrs,
    ) -> Option<InferredTensor>;
    fn reassociable(
        &self,
        arg0: &TensorInfo,
        arg1: &TensorInfo,
        arg2: &TensorInfo,
        arg3: &OpAttrs,
        arg4: &OpAttrs,
    ) -> Option<bool>;
}
