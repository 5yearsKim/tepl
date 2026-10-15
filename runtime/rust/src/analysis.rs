use crate::values::{AttributeValue, shape};
use crate::{Budget, DType, EvalError, EvalResult, Node, NodeKind, Program, Value};
use egg::{Analysis, DidMerge, EGraph, Id};
use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    sync::Mutex,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Fact<T> {
    #[default]
    Unknown,
    Known(T),
    Conflict,
}
impl<T: Clone + Eq> Fact<T> {
    pub fn get(&self, name: &'static str) -> EvalResult<&T> {
        match self {
            Self::Known(x) => Ok(x),
            Self::Unknown => Err(EvalError::Unavailable(name)),
            Self::Conflict => Err(EvalError::Conflict(format!("conflicting {name}"))),
        }
    }
    pub fn merge(&mut self, from: Self) -> DidMerge {
        let old = self.clone();
        let other = from.clone();
        *self = match (&old, from) {
            (Self::Conflict, _) | (_, Self::Conflict) => Self::Conflict,
            (Self::Unknown, x) => x,
            (x, Self::Unknown) => x.clone(),
            (Self::Known(x), Self::Known(y)) if x == &y => Self::Known(y),
            _ => Self::Conflict,
        };
        DidMerge(*self != old, *self != other)
    }
    pub fn disagrees(&self, other: &Self) -> bool {
        matches!((self, other), (Self::Known(x), Self::Known(y)) if x != y)
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorFacts {
    pub shape: Fact<Vec<u64>>,
    pub dtype: Fact<DType>,
}
impl TensorFacts {
    pub fn known(shape: Vec<u64>, dtype: DType) -> Self {
        Self {
            shape: Fact::Known(shape),
            dtype: Fact::Known(dtype),
        }
    }
    pub fn shape_value(&self) -> EvalResult<Value> {
        Ok(shape(self.shape.get("shape")?))
    }
    pub fn dtype_value(&self) -> EvalResult<Value> {
        Ok(self.dtype.get("dtype")?.to_value())
    }
    pub fn conflicts(&self) -> bool {
        self.shape == Fact::Conflict || self.dtype == Fact::Conflict
    }
    pub fn disagrees(&self, other: &Self) -> bool {
        self.shape.disagrees(&other.shape) || self.dtype.disagrees(&other.dtype)
    }
}
pub struct TensorAnalysis<P: Program> {
    inputs: BTreeMap<String, TensorFacts>,
    inserted_inputs: BTreeSet<String>,
    diagnostic: Mutex<Option<EvalError>>,
    marker: PhantomData<P>,
}
impl<P: Program> Default for TensorAnalysis<P> {
    fn default() -> Self {
        Self {
            inputs: BTreeMap::new(),
            inserted_inputs: BTreeSet::new(),
            diagnostic: Mutex::new(None),
            marker: PhantomData,
        }
    }
}
impl<P: Program> TensorAnalysis<P> {
    pub fn new(inputs: impl IntoIterator<Item = (String, TensorFacts)>) -> EvalResult<Self> {
        let mut analysis = Self::default();
        for (name, facts) in inputs {
            analysis.register_input(name, facts)?;
        }
        Ok(analysis)
    }
    /// Register inputs before inserting their nodes into the e-graph.
    pub fn register_input(
        &mut self,
        name: impl Into<String>,
        facts: TensorFacts,
    ) -> EvalResult<()> {
        let name = name.into();
        if facts.conflicts() {
            return Err(EvalError::Conflict(format!(
                "conflicting input metadata for {name}"
            )));
        }
        let old = self.inputs.get(&name).cloned().unwrap_or_default();
        let mut entry = old.clone();
        entry.shape.merge(facts.shape);
        entry.dtype.merge(facts.dtype);
        if entry.conflicts() {
            return Err(EvalError::Conflict(format!(
                "conflicting input metadata for {name}"
            )));
        }
        crate::error::require(
            !self.inserted_inputs.contains(&name) || old == entry,
            "register input metadata before inserting its node",
        )?;
        self.inputs.insert(name, entry);
        Ok(())
    }
    pub fn report(&self, error: EvalError) {
        let mut diagnostic = self.diagnostic.lock().unwrap();
        if diagnostic.is_none() {
            *diagnostic = Some(error);
        }
    }
    pub fn check(&self) -> EvalResult<()> {
        match self.diagnostic.lock().unwrap().clone() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
    pub fn infer_node(&self, node: &Node<P>, operands: &[TensorFacts]) -> EvalResult<TensorFacts> {
        self.infer_node_with_budget(node, operands, &mut Budget::default())
    }
    pub fn infer_node_with_budget(
        &self,
        node: &Node<P>,
        operands: &[TensorFacts],
        budget: &mut Budget,
    ) -> EvalResult<TensorFacts> {
        match node.kind() {
            NodeKind::Input(name) => Ok(self.inputs.get(name).cloned().unwrap_or_default()),
            NodeKind::Literal { dtype, .. } => Ok(TensorFacts {
                shape: Fact::Known(vec![]),
                dtype: dtype.map_or(Fact::Unknown, Fact::Known),
            }),
            NodeKind::Operation(op) => {
                if operands.iter().any(TensorFacts::conflicts) {
                    return Err(EvalError::Conflict("conflicting operand metadata".into()));
                }
                let attrs = node.attribute_value();
                let shape = match P::infer_shape(*op, operands, &attrs, budget) {
                    Ok(x) => x.map_or(Fact::Unknown, Fact::Known),
                    Err(EvalError::Unavailable(_)) => Fact::Unknown,
                    Err(e) => return Err(e),
                };
                let dtype = match P::infer_dtype(*op, operands, &attrs, budget) {
                    Ok(x) => x.map_or(Fact::Unknown, Fact::Known),
                    Err(EvalError::Unavailable(_)) => Fact::Unknown,
                    Err(e) => return Err(e),
                };
                Ok(TensorFacts { shape, dtype })
            }
        }
    }
}
impl<P: Program> Analysis<Node<P>> for TensorAnalysis<P> {
    type Data = TensorFacts;
    fn make(graph: &mut EGraph<Node<P>, Self>, node: &Node<P>, _: Id) -> TensorFacts {
        if let NodeKind::Input(name) = node.kind() {
            graph.analysis.inserted_inputs.insert(name.clone());
        }
        let operands = node
            .operands()
            .iter()
            .map(|id| graph[*id].data.clone())
            .collect::<Vec<_>>();
        match graph.analysis.infer_node(node, &operands) {
            Ok(x) => x,
            Err(e) => {
                graph.analysis.report(e);
                TensorFacts {
                    shape: Fact::Conflict,
                    dtype: Fact::Conflict,
                }
            }
        }
    }
    fn merge(&mut self, to: &mut TensorFacts, from: TensorFacts) -> DidMerge {
        let a = to.shape.merge(from.shape);
        let b = to.dtype.merge(from.dtype);
        if to.conflicts() {
            self.report(EvalError::Conflict(
                "e-class shape or dtype conflict".into(),
            ));
        }
        DidMerge(a.0 || b.0, a.1 || b.1)
    }
}
