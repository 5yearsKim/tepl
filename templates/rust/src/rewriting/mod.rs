//! Shared matching and rewrite support for generated dialects.
mod checks;
mod constraints;
mod context;
mod matcher;
mod pattern;
mod rewrite;
mod shape;

#[allow(unused_imports)] // Projects without concrete rules do not use this adapter.
pub(crate) use context::AnalysisMetadata;

pub use checks::{MatchBinding, MatchChecks};
pub use constraints::{DTypeConstraint, TensorConstraint, TensorConstraints};
pub use context::{MatchContext, OutputInference, RewriteAnalysis, TensorMetadata};
pub use matcher::{TensorMatch, matches_at, matches_at_with_checks, matches_at_with_constraints};
pub use pattern::{AttrExpr, AttrPattern, AttrVar, TensorExpr, TensorPattern};
pub use rewrite::{
    DerivedAttrs, tensor_rewrite_checked, tensor_rewrite_checked_with_checks,
    tensor_rewrite_checked_with_constraints, tensor_rewrite_with_checks,
};
pub use shape::{MetadataBindings, ShapePart};
