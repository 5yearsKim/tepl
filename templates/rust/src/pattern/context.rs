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

/// Connect an egg analysis to generated rewrites. The capability is fixed by the
/// analysis type; unavailable facts in a checked analysis never disable checks.
pub trait RewriteAnalysis: Analysis<OpNode> + Send + Sync + 'static {
    const HAS_TENSOR_INFO: bool;

    fn tensor_info(graph: &EGraph<OpNode, Self>, id: Id) -> Option<TensorInfo>;

    fn infer_output(&self, op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo>;

    fn infer_literal(
        &self,
        _value: &str,
        dtype: Option<DType>,
        expected: &TensorInfo,
    ) -> Option<DType> {
        dtype.or(Some(expected.dtype))
    }
}

impl RewriteAnalysis for () {
    const HAS_TENSOR_INFO: bool = false;
    fn tensor_info(_: &EGraph<OpNode, Self>, _: Id) -> Option<TensorInfo> {
        None
    }
    fn infer_output(&self, _: Op, _: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
        None
    }
}

// Readers remain an implementation detail of generated matchers.
pub(crate) struct AnalysisMetadata<N>(::std::marker::PhantomData<fn() -> N>);
impl<N> AnalysisMetadata<N> {
    pub(crate) fn new() -> Self {
        Self(::std::marker::PhantomData)
    }
}
impl<N: RewriteAnalysis> TensorMetadata<N> for AnalysisMetadata<N> {
    fn info(&self, graph: &EGraph<OpNode, N>, id: Id) -> Option<TensorInfo> {
        N::tensor_info(graph, id)
    }
}

pub(super) struct AnalysisInference<'a, N>(pub &'a N);
impl<N: RewriteAnalysis> OutputInference for AnalysisInference<'_, N> {
    fn infer_output(&self, op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
        self.0.infer_output(op, operands, attrs)
    }
    fn infer_literal(
        &self,
        value: &str,
        dtype: Option<DType>,
        expected: &TensorInfo,
    ) -> Option<DType> {
        self.0.infer_literal(value, dtype, expected)
    }
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
