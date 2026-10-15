//! Shared execution machinery for compiled TEPL programs.
pub mod analysis;
pub mod axes;
pub mod builtins;
pub mod error;
pub mod graphs;
pub mod intrinsics;
pub mod matching;
pub mod metadata;
pub mod node;
pub mod rewriting;
pub mod runner;
pub mod types;
pub mod values;

pub use analysis::{Fact, TensorAnalysis, TensorFacts};
pub use egg;
pub use error::{EvalError, EvalResult};
pub use node::{Node, NodeKind, Program};
pub use rewriting::{PreparedReplacement, RuleContext, RuleDefinition};
pub use types::DType;
pub use values::{Budget, Value};

/// Generated code and runtime must use the same execution ABI.
pub const RUNTIME_ABI: u32 = 1;
