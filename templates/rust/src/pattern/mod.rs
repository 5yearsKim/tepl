//! Shared matching and checked rewrite support for generated dialects.
mod checks;
mod constraints;
mod context;
mod matcher;
mod pattern;
mod rewrite;
mod shape;

pub use checks::{MatchBinding, MatchChecks};
pub use constraints::{TensorConstraint, TensorConstraints};
pub use context::{MatchContext, OutputInference, TensorInfo, TensorMetadata};
pub use matcher::{TensorMatch, matches_at, matches_at_with_checks, matches_at_with_constraints};
pub use pattern::{AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern};
pub use rewrite::{
    DerivedAttrs, tensor_rewrite_checked, tensor_rewrite_checked_with_checks,
    tensor_rewrite_checked_with_constraints,
};
pub use shape::{ShapeBindings, ShapePart, finite};
