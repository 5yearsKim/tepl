// Generated per-rule function modules. Keep implementations in a separate file.
use rust_egg::ir::OpAttrs;
use rust_egg::ir::patterns::TensorInfo;

pub mod lora {
    use super::*;

    pub trait Functions: Send + Sync {
        fn broadcastable(&self, arg0: &[usize], arg1: &[usize]) -> Option<bool>;
        fn infer_dot(
            &self,
            arg0: &TensorInfo,
            arg1: &TensorInfo,
            arg2: &OpAttrs,
        ) -> Option<OpAttrs>;
        fn infer_lora_out(
            &self,
            arg0: &TensorInfo,
            arg1: &TensorInfo,
            arg2: &TensorInfo,
            arg3: &OpAttrs,
            arg4: &OpAttrs,
        ) -> Option<OpAttrs>;
        fn reassociable(
            &self,
            arg0: &TensorInfo,
            arg1: &TensorInfo,
            arg2: &TensorInfo,
            arg3: &OpAttrs,
            arg4: &OpAttrs,
        ) -> Option<bool>;
    }
}
