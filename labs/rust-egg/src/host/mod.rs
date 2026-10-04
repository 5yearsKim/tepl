//! Application-specific node helpers and explicit LoRA analysis/legality policies.
//! Reusable tensor inference and e-class analysis live in ir::analysis.
mod lora;
mod lora_analysis;
pub mod nodes;

pub use lora::{DemoLoraFunctions, batched_dot_shape, dot_attrs};

pub use lora_analysis::{LoraAnalysis, infer_lora_tensor_output, lora_tensor_info};
