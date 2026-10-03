//! Ready-to-use tensor analysis: input bindings, inference, and e-class facts.
//! Shape evaluators are reference output for TEPL; dtype policies are maintained
//! in Rust. Shared support is a reference for copied runtime code, and
//! shape_builtins is synchronized with its maintained source. Pure inference is independent
//! of egg, while tensor_analysis.rs connects them to the graph.
mod bindings;
mod dtype;
mod inference;
mod payload;
mod shape;
mod tensor;
mod tensor_analysis;
mod tensor_analysis_data;
// Keep the complete builtin reference even when this dialect uses only a subset.
#[allow(dead_code)]
mod shape_builtins;

pub use super::pattern::TensorInfo;
pub use bindings::TensorBindingTable;
pub use dtype::infer_dtype;
pub use inference::Inference;
pub use shape::infer_shape;
pub use tensor::{infer_tensor, infer_tensor_output};
pub use tensor_analysis::{TensorAnalysis, tensor_info};
pub use tensor_analysis_data::TensorAnalysisData;
