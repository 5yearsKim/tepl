use egg::{Analysis, EGraph, Id, Var};

use crate::ir::{OpAttrs, TensorLang};

use super::matcher::TensorMatch;
use super::pattern::AttrVar;

/// The concrete tensor description available to a rule's semantic functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
}

/// Supplies metadata valid for every alternative in an e-class. Return `None`
/// when that cannot be established; choosing one arbitrary node is unsound.
pub trait TensorMetadata<N: Analysis<TensorLang>>: Send + Sync {
    fn info(&self, egraph: &EGraph<TensorLang, N>, eclass: Id) -> Option<TensorInfo>;
}

impl<N, F> TensorMetadata<N> for F
where
    N: Analysis<TensorLang>,
    F: Fn(&EGraph<TensorLang, N>, Id) -> Option<TensorInfo> + Send + Sync,
{
    fn info(&self, egraph: &EGraph<TensorLang, N>, eclass: Id) -> Option<TensorInfo> {
        self(egraph, eclass)
    }
}

/// Read-only access to one structural match for generated semantic code.
pub struct MatchContext<'a, N: Analysis<TensorLang>, M: TensorMetadata<N>> {
    egraph: &'a EGraph<TensorLang, N>,
    matched: &'a TensorMatch,
    metadata: &'a M,
}

impl<'a, N: Analysis<TensorLang>, M: TensorMetadata<N>> MatchContext<'a, N, M> {
    pub fn new(
        egraph: &'a EGraph<TensorLang, N>,
        matched: &'a TensorMatch,
        metadata: &'a M,
    ) -> Self {
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
