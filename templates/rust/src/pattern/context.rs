use ::egg::{Analysis, EGraph, Id, Var};

use super::super::{DType, Op, OpAttrs, OpNode};

use super::matcher::TensorMatch;
use super::pattern::AttrVar;

/// The concrete tensor description available to a rule's semantic functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<u64>,
    pub dtype: DType,
}

/// Infer a constructed operation's output separately from descriptor derivation.
/// Implementations can share the same operation semantics with e-class analysis.
pub trait OutputInference: Send + Sync {
    /// Resolve an unconstrained literal using host semantics and matched output context.
    /// The default uses the matched root's dtype; explicit annotations always win.
    fn infer_literal(
        &self,
        _value: &str,
        dtype: Option<DType>,
        expected: &TensorInfo,
    ) -> Option<DType> {
        dtype.or(Some(expected.dtype))
    }

    fn infer_output(&self, op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo>;
}

impl<F> OutputInference for F
where
    F: Fn(Op, &[TensorInfo], &OpAttrs) -> Option<TensorInfo> + Send + Sync,
{
    fn infer_output(&self, op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
        self(op, operands, attrs)
    }
}

/// Supplies metadata valid for every alternative in an e-class. Return `None`
/// when that cannot be established; choosing one arbitrary node is unsound.
/// Answers must remain stable during a read-only matching traversal.
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
