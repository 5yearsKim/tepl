//! Application-specific node helpers and custom LoRA rewrite functions.
//! Reusable tensor inference and e-class analysis live in ir::analysis.
mod lora;
pub mod nodes;

pub use lora::{DemoLoraFunctions, batched_dot_shape, dot_attrs};
