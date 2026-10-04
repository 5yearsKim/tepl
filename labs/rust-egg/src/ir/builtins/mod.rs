//! Pure TEPL builtin implementations shared by generated expression evaluators.
pub mod common;
pub mod dtype;
mod error;
pub mod shape;

pub use error::{BuiltinError, BuiltinResult};
