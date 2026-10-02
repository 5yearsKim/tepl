//! Generic tensor pattern matching and egg rewrite support.
//! Generated rules supply patterns and a semantic callback.

mod context;
mod matcher;
mod pattern;
mod rewrite;

pub use context::{MatchContext, TensorInfo, TensorMetadata};
pub use matcher::{TensorMatch, matches_at};
pub use pattern::{AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern};
pub use rewrite::{DerivedAttrs, tensor_rewrite};
