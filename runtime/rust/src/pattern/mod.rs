//! Shared matching and checked rewrite support for generated dialects.
mod context;
mod matcher;
mod pattern;
mod rewrite;
mod shape;

pub use context::{MatchContext, OutputInference, TensorInfo, TensorMetadata};
pub use matcher::{TensorMatch, matches_at};
pub use pattern::{AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern};
pub use rewrite::{DerivedAttrs, tensor_rewrite_checked};
pub use shape::{ShapeBindings, ShapePart, finite};
