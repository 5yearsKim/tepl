// Generated per-rule function modules. Keep implementations in a separate file.
use rust_egg::ir::patterns::TensorInfo;

pub mod commute_same_dtype {
    use super::*;

    pub trait Functions: Send + Sync {
        fn same_dtype(&self, arg0: &TensorInfo, arg1: &TensorInfo) -> Option<bool>;
    }
}
