//! Pure shape and dtype inference, independent of host policy and e-classes.
//! Operation evaluators are a handwritten reference for future generation;
//! shape_builtins is copied from the maintained Rust runtime.
mod dtype;
mod inference;
mod shape;
// Keep the complete builtin reference even when this dialect uses only a subset.
#[allow(dead_code)]
mod shape_builtins;

pub use dtype::infer_dtype;
pub use inference::Inference;
pub use shape::infer_shape;
