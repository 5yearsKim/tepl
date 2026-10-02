use crate::ir::dialects::tensor_lang;
use egg::{Analysis, EGraph, Id, Var};

use crate::ir::{DType, Op, OpAttrs, OpNode};

use super::matcher::TensorMatch;
use super::pattern::AttrVar;

/// The concrete tensor description available to a rule's semantic functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Infer a constructed operation's output separately from descriptor derivation.
/// Implementations can share the same operation semantics with e-class analysis.
pub trait OutputInference: Send + Sync {
    fn infer_output(
        &self,
        op: crate::ir::Op,
        operands: &[TensorInfo],
        attrs: &OpAttrs,
    ) -> Option<TensorInfo>;
}

impl<F> OutputInference for F
where
    F: Fn(crate::ir::Op, &[TensorInfo], &OpAttrs) -> Option<TensorInfo> + Send + Sync,
{
    fn infer_output(
        &self,
        op: crate::ir::Op,
        operands: &[TensorInfo],
        attrs: &OpAttrs,
    ) -> Option<TensorInfo> {
        self(op, operands, attrs)
    }
}

/// Supplies metadata valid for every alternative in an e-class. Return `None`
/// when that cannot be established; choosing one arbitrary node is unsound.
pub trait TensorMetadata<N: Analysis<OpNode>>: Send + Sync {
    fn info(&self, egraph: &EGraph<OpNode, N>, eclass: Id) -> Option<TensorInfo>;
}

impl<N, F> TensorMetadata<N> for F
where
    N: Analysis<OpNode>,
    F: Fn(&EGraph<OpNode, N>, Id) -> Option<TensorInfo> + Send + Sync,
{
    fn info(&self, egraph: &EGraph<OpNode, N>, eclass: Id) -> Option<TensorInfo> {
        self(egraph, eclass)
    }
}

/// Read-only access to one structural match for generated semantic code.
pub struct MatchContext<'a, N: Analysis<OpNode>, M: TensorMetadata<N>> {
    egraph: &'a EGraph<OpNode, N>,
    matched: &'a TensorMatch,
    metadata: &'a M,
}

impl<'a, N: Analysis<OpNode>, M: TensorMetadata<N>> MatchContext<'a, N, M> {
    pub fn new(egraph: &'a EGraph<OpNode, N>, matched: &'a TensorMatch, metadata: &'a M) -> Self {
        Self {
            egraph,
            matched,
            metadata,
        }
    }

    pub fn tensor_id(&self, var: Var) -> Option<Id> {
        self.matched
            .tensors
            .get(var)
            .map(|id| self.egraph.find(*id))
    }

    pub fn tensor(&self, var: Var) -> Option<TensorInfo> {
        self.metadata.info(self.egraph, self.tensor_id(var)?)
    }

    pub fn attrs(&self, var: AttrVar) -> Option<&OpAttrs> {
        self.matched.attrs.get(&var)
    }
}

/// Host input identities with immutable types. Register inputs before building
/// the graph; conflicting registrations never overwrite existing metadata.
#[derive(Clone, Debug, Default)]
pub struct TensorBindings {
    entries: std::collections::HashMap<(Op, String), TensorInfo>,
}

impl TensorBindings {
    pub fn register_symbol(
        &mut self,
        name: impl Into<String>,
        info: TensorInfo,
    ) -> Result<(), String> {
        self.register(Op::TensorLang(tensor_lang::Op::Symbol), name.into(), info)
    }

    pub fn register_constant(
        &mut self,
        name: impl Into<String>,
        info: TensorInfo,
    ) -> Result<(), String> {
        self.register(Op::TensorLang(tensor_lang::Op::Constant), name.into(), info)
    }

    fn register(&mut self, op: Op, name: String, info: TensorInfo) -> Result<(), String> {
        let key = (op, name);
        if let Some(previous) = self.entries.get(&key) {
            if previous != &info {
                return Err(format!(
                    "conflicting tensor type for {} '{}'",
                    op.name(),
                    key.1
                ));
            }
            return Ok(());
        }
        self.entries.insert(key, info);
        Ok(())
    }

    pub fn info(&self, node: &OpNode) -> Option<&TensorInfo> {
        let name = match node.attrs() {
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Symbol { name })
            | OpAttrs::TensorLang(tensor_lang::OpAttrs::Constant { name }) => name,
            _ => return None,
        };
        self.entries.get(&(node.op(), name.clone()))
    }
}
