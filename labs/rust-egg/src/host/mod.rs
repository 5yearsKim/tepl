//! Start with analysis.rs: input bindings -> operation inference -> e-class facts.
mod analysis;
mod bindings;
mod facts;
mod inference;
mod lora;
mod metadata;
pub mod nodes;

pub use analysis::{ShapeAnalysis, tensor_info};
pub use bindings::TensorBindings;
pub use facts::TensorFacts;
pub use inference::{infer_tensor, infer_tensor_output};
pub use lora::{DemoLoraFunctions, batched_dot_shape, dot_attrs};
pub use metadata::infer_eclass;
