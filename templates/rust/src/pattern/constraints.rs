use ::std::collections::{HashMap, HashSet};

use ::egg::{Analysis, EGraph, Var};

use super::super::{DType, OpNode};
use super::context::{TensorInfo, TensorMetadata};
use super::matcher::TensorMatch;
use super::pattern::TensorPattern;
use super::shape::{MetadataBindings, ShapePart};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeConstraint {
    Exact(DType),
    Bind(usize),
}

#[derive(Clone, Debug)]
pub struct TensorConstraint {
    pub dtype: Option<DTypeConstraint>,
    pub shape: Vec<ShapePart>,
}

/// Ordered declarations, indexed by capture. Inherited restrictions are additive.
#[derive(Clone, Debug, Default)]
pub struct TensorConstraints {
    declarations: Vec<(Var, TensorConstraint)>,
    by_capture: HashMap<Var, Vec<usize>>,
}

impl TensorConstraints {
    pub fn new(declarations: impl IntoIterator<Item = (Var, TensorConstraint)>) -> Self {
        let declarations: Vec<_> = declarations.into_iter().collect();
        let mut by_capture: HashMap<Var, Vec<usize>> = HashMap::new();
        for (index, (var, _)) in declarations.iter().enumerate() {
            by_capture.entry(*var).or_default().push(index);
        }
        Self {
            declarations,
            by_capture,
        }
    }

    /// Validate once before traversal, including plans supplied by handwritten callers.
    pub fn validate(&self, pattern: &TensorPattern) -> Result<(), String> {
        let mut captures = HashSet::new();
        pattern.vars(&mut captures);
        let mut symbols = HashMap::new();
        for (var, constraint) in &self.declarations {
            if !captures.contains(var) {
                return Err(format!("constraint refers to uncaptured tensor {var}"));
            }
            let mut sequences = 0;
            for part in &constraint.shape {
                let (id, sequence) = match part {
                    ShapePart::Dimension(id) => (Some(*id), false),
                    ShapePart::Sequence(id) => {
                        sequences += 1;
                        (*id, true)
                    }
                    ShapePart::Wildcard => (None, false),
                };
                if let Some(id) = id {
                    if symbols
                        .insert(id, sequence)
                        .is_some_and(|old| old != sequence)
                    {
                        return Err(format!(
                            "shape symbol {id} is both a dimension and a sequence"
                        ));
                    }
                }
            }
            if sequences > 1 {
                return Err("a shape can contain at most one sequence".into());
            }
        }
        Ok(())
    }

    pub(super) fn shape_symbols(&self) -> impl Iterator<Item = (usize, bool)> + '_ {
        self.declarations
            .iter()
            .flat_map(|(_, constraint)| constraint.shape.iter())
            .filter_map(|part| match part {
                ShapePart::Dimension(id) => Some((*id, false)),
                ShapePart::Sequence(Some(id)) => Some((*id, true)),
                _ => None,
            })
    }

    pub(super) fn dtype_symbols(&self) -> impl Iterator<Item = usize> + '_ {
        self.declarations
            .iter()
            .filter_map(|(_, constraint)| match constraint.dtype {
                Some(DTypeConstraint::Bind(id)) => Some(id),
                _ => None,
            })
    }

    pub fn contains(&self, var: Var) -> bool {
        self.by_capture.contains_key(&var)
    }

    /// Use a validated plan. Discard `bindings` after failure; successful checks
    /// share symbols across captures. An unconstrained capture adds no restrictions.
    pub fn check_capture(
        &self,
        var: Var,
        info: &TensorInfo,
        bindings: &mut MetadataBindings,
    ) -> Option<()> {
        if let Some(indices) = self.by_capture.get(&var) {
            for &index in indices {
                let constraint = &self.declarations[index].1;
                match constraint.dtype {
                    Some(DTypeConstraint::Exact(dtype)) if dtype != info.dtype => return None,
                    Some(DTypeConstraint::Bind(id)) => bindings.bind_dtype(id, info.dtype)?,
                    _ => {}
                }
                bindings.check(&info.shape, &constraint.shape)?;
            }
        }
        Some(())
    }

    /// Recheck current e-class facts before application and recover dimension bindings.
    pub fn check_match<N: Analysis<OpNode>>(
        &self,
        graph: &EGraph<OpNode, N>,
        matched: &TensorMatch,
        metadata: &dyn TensorMetadata<N>,
    ) -> Option<MetadataBindings> {
        let mut bindings = MetadataBindings::default();
        let mut checked = HashSet::new();
        for (var, _) in &self.declarations {
            if checked.insert(*var) {
                let id = graph.find(*matched.tensors.get(*var)?);
                self.check_capture(*var, &metadata.info(graph, id)?, &mut bindings)?;
            }
        }
        Some(bindings)
    }
}
