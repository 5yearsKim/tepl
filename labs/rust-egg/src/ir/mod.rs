// Shared inference and e-class analysis reference; the compiler currently emits
// only shape_builtins from this module.
pub mod analysis;
pub mod dialects;
pub mod op_node;
pub mod pattern;
pub mod rules;
pub mod types;
pub use op_node::{Arity, DialectOp, NodeError, Op, OpAttrs, OpNode};
pub use types::DType;
