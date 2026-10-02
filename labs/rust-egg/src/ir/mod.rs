//! Tensor dialect operations, patterns, and rules for `egg`.

pub mod dialects;
pub mod patterns;
pub mod rules;
pub mod types;

pub use types::DType;

pub use dialects::tensor_lang::{Arity, NodeError, OpAttrs, OpKind, TensorLang};
