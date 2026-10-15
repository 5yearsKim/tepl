use crate::error::require;
use crate::values::AttributeValue;
use crate::{Budget, DType, EvalResult, TensorFacts, Value};
use egg::{Id, Language};
use std::{fmt, hash::Hash};

/// The only interface implemented by a generated program.
pub trait Program: Clone + fmt::Debug + Eq + Ord + Hash + Send + Sync + 'static {
    type Op: Clone + Copy + fmt::Debug + Eq + Ord + Hash + Send + Sync;
    type Attrs: Clone + fmt::Debug + Eq + Ord + Hash + Send + Sync + AttributeValue;
    fn operation(id: usize) -> Self::Op;
    fn operation_id(op: Self::Op) -> usize;
    fn operation_name(op: Self::Op) -> &'static str;
    fn arity(op: Self::Op) -> (usize, bool);
    fn accepts_attrs(op: Self::Op, attrs: Option<&Self::Attrs>) -> bool;
    fn infer_shape(
        op: Self::Op,
        operands: &[TensorFacts],
        attrs: &Value,
        budget: &mut Budget,
    ) -> EvalResult<Option<Vec<u64>>>;
    fn infer_dtype(
        op: Self::Op,
        operands: &[TensorFacts],
        attrs: &Value,
        budget: &mut Budget,
    ) -> EvalResult<Option<DType>>;
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKind<O> {
    Input(String),
    Literal { value: String, dtype: Option<DType> },
    Operation(O),
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<P: Program> {
    kind: NodeKind<P::Op>,
    attrs: Option<P::Attrs>,
    children: Vec<Id>,
}
impl<P: Program> Node<P> {
    pub fn operation(
        op: impl Into<P::Op>,
        attrs: Option<P::Attrs>,
        children: Vec<Id>,
    ) -> EvalResult<Self> {
        let op = op.into();
        let (n, variadic) = P::arity(op);
        require(
            if variadic {
                children.len() >= n
            } else {
                children.len() == n
            },
            "operation arity mismatch",
        )?;
        require(
            P::accepts_attrs(op, attrs.as_ref()),
            "operation attribute schema mismatch",
        )?;
        Ok(Self {
            kind: NodeKind::Operation(op),
            attrs,
            children,
        })
    }
    pub fn input(name: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Input(name.into()),
            attrs: None,
            children: vec![],
        }
    }
    pub fn literal(value: impl Into<String>, dtype: Option<DType>) -> EvalResult<Self> {
        let value = value.into();
        let decimal = value.contains('.');
        if decimal {
            crate::values::floating(&value)?;
        } else {
            crate::values::integer(&value)?;
        }
        require(
            dtype.is_none_or(|d| d.accepts_literal(&value)),
            "literal incompatible with dtype",
        )?;
        Ok(Self {
            kind: NodeKind::Literal { value, dtype },
            attrs: None,
            children: vec![],
        })
    }
    pub fn kind(&self) -> &NodeKind<P::Op> {
        &self.kind
    }
    pub fn attrs(&self) -> Option<&P::Attrs> {
        self.attrs.as_ref()
    }
    pub fn attribute_value(&self) -> Value {
        self.attrs
            .as_ref()
            .map_or(Value::None, AttributeValue::to_value)
    }
    pub fn operands(&self) -> &[Id] {
        &self.children
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Discriminant<O> {
    Input,
    Literal,
    Operation(O),
}
impl<P: Program> Language for Node<P> {
    type Discriminant = Discriminant<P::Op>;
    fn discriminant(&self) -> Self::Discriminant {
        match self.kind {
            NodeKind::Input(_) => Discriminant::Input,
            NodeKind::Literal { .. } => Discriminant::Literal,
            NodeKind::Operation(op) => Discriminant::Operation(op),
        }
    }
    fn matches(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.attrs == other.attrs
            && self.children.len() == other.children.len()
    }
    fn children(&self) -> &[Id] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut [Id] {
        &mut self.children
    }
}
impl<P: Program> fmt::Display for Node<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            NodeKind::Input(name) => f.write_str(name),
            NodeKind::Literal { value, .. } => f.write_str(value),
            NodeKind::Operation(op) => {
                f.write_str(P::operation_name(*op))?;
                if let Some(attrs) = &self.attrs {
                    write!(f, "[{attrs:?}]")?;
                }
                Ok(())
            }
        }
    }
}
