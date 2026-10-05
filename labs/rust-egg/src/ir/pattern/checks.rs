use ::std::collections::HashSet;

use ::egg::{Analysis, EGraph, Var};

use super::super::OpNode;
use super::constraints::TensorConstraints;
use super::context::TensorMetadata;
use super::matcher::TensorMatch;
use super::pattern::{AttrVar, TensorPattern};
use super::shape::MetadataBindings;

/// A binding needed before a condition can execute. Metadata is fetched only
/// inside the expression, preserving &&/|| short-circuit behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatchBinding {
    Tensor(Var),
    Attribute(AttrVar),
    DType(usize),
    Dimension(usize),
    Sequence(usize),
}

impl MatchBinding {
    fn is_bound(self, matched: &TensorMatch, shapes: &MetadataBindings) -> bool {
        match self {
            Self::Tensor(var) => matched.tensors.get(var).is_some(),
            Self::Attribute(var) => matched.attrs.contains_key(&var),
            Self::DType(id) => shapes.dtype(id).is_some(),
            Self::Dimension(id) => shapes.dimension(id).is_some(),
            Self::Sequence(id) => shapes.sequence(id).is_some(),
        }
    }
}

type ConditionEvaluator<N> = dyn Fn(usize, &EGraph<OpNode, N>, &TensorMatch, &MetadataBindings) -> Option<bool>
    + Send
    + Sync;

/// Tensor declarations bind facts; ordered conditions consume those facts.
/// Evaluators must be pure and stable during a read-only traversal. Some(false)
/// and None reject a branch. Missing bindings delay evaluation instead.
pub struct MatchChecks<N: Analysis<OpNode>> {
    tensors: TensorConstraints,
    conditions: Vec<Vec<MatchBinding>>,
    evaluate: Box<ConditionEvaluator<N>>,
}

impl<N: Analysis<OpNode>> MatchChecks<N> {
    pub fn new(
        tensors: TensorConstraints,
        conditions: Vec<Vec<MatchBinding>>,
        evaluate: impl Fn(usize, &EGraph<OpNode, N>, &TensorMatch, &MetadataBindings) -> Option<bool>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            tensors,
            conditions,
            evaluate: Box::new(evaluate),
        }
    }

    pub fn tensors(tensors: TensorConstraints) -> Self {
        Self::new(tensors, vec![], |_, _, _, _| Some(true))
    }

    pub fn tensor_constraints(&self) -> &TensorConstraints {
        &self.tensors
    }

    pub fn validate(&self, pattern: &TensorPattern) -> Result<(), String> {
        self.tensors.validate(pattern)?;
        let mut tensors = HashSet::new();
        let mut attrs = HashSet::new();
        pattern.vars(&mut tensors);
        pattern.attr_vars(&mut attrs);
        let mut available: HashSet<_> = tensors.into_iter().map(MatchBinding::Tensor).collect();
        available.extend(attrs.into_iter().map(MatchBinding::Attribute));
        available.extend(self.tensors.shape_symbols().map(|(id, sequence)| {
            if sequence {
                MatchBinding::Sequence(id)
            } else {
                MatchBinding::Dimension(id)
            }
        }));
        available.extend(self.tensors.dtype_symbols().map(MatchBinding::DType));
        for (index, dependencies) in self.conditions.iter().enumerate() {
            for dependency in dependencies {
                if !available.contains(dependency) {
                    return Err(format!(
                        "condition {index} refers to unavailable binding {dependency:?}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Advance one branch's cursor. Wait at the first unbound condition; never
    /// reorder conditions or evaluate a passed condition again in this branch.
    pub(super) fn advance(
        &self,
        graph: &EGraph<OpNode, N>,
        matched: &TensorMatch,
        shapes: &MetadataBindings,
        next: &mut usize,
    ) -> Option<()> {
        while let Some(dependencies) = self.conditions.get(*next) {
            if !dependencies
                .iter()
                .all(|binding| binding.is_bound(matched, shapes))
            {
                break;
            }
            if !(self.evaluate)(*next, graph, matched, shapes)? {
                return None;
            }
            *next += 1;
        }
        Some(())
    }

    pub(super) fn complete(&self, next: usize) -> bool {
        next == self.conditions.len()
    }

    /// Recover current dimension bindings and recheck all search conditions
    /// before application-time host calls or descriptor derivations.
    pub fn check_match(
        &self,
        graph: &EGraph<OpNode, N>,
        matched: &TensorMatch,
        metadata: &dyn TensorMetadata<N>,
    ) -> Option<MetadataBindings> {
        let shapes = self.tensors.check_match(graph, matched, metadata)?;
        let mut next = 0;
        self.advance(graph, matched, &shapes, &mut next)?;
        self.complete(next).then_some(shapes)
    }
}
