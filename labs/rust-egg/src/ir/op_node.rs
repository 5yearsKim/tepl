use super::DType;
use ::egg::{Id, Language};
use ::std::fmt;

// Generated sum types: one egg language, distinct dialect identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Literal,
    Input,
    Scalar(super::dialects::scalar::Op),
    TensorLang(super::dialects::tensor_lang::Op),
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpAttrs {
    None,
    Literal { value: String, dtype: Option<DType> },
    Input { name: String },
    Scalar(super::dialects::scalar::OpAttrs),
    TensorLang(super::dialects::tensor_lang::OpAttrs),
}
impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Self::Literal => "<literal>",
            Self::Input => "<input>",
            Self::Scalar(op) => op.name(),
            Self::TensorLang(op) => op.name(),
        }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        if name == "<literal>" {
            return Some(Self::Literal);
        }
        if name == "<input>" {
            return Some(Self::Input);
        }
        let (dialect, _op) = name.split_once('.')?;
        match dialect {
            "Scalar" => super::dialects::scalar::Op::from_name(_op).map(Self::Scalar),
            "TensorLang" => super::dialects::tensor_lang::Op::from_name(_op).map(Self::TensorLang),
            _ => None,
        }
    }
    pub fn arity(self) -> Arity {
        match self {
            Self::Literal | Self::Input => Arity::Exact(0),
            Self::Scalar(op) => op.arity(),
            Self::TensorLang(op) => op.arity(),
        }
    }
    pub fn accepts_attrs(self, attrs: &OpAttrs) -> bool {
        match (self, attrs) {
            (Self::Input, OpAttrs::Input { .. }) => true,
            (Self::Literal, OpAttrs::Literal { value, dtype }) => {
                valid_literal(value) && dtype.is_none_or(|d| d.accepts_literal(value))
            }
            (Self::Scalar(op), OpAttrs::Scalar(attrs)) => op.accepts_attrs(attrs),
            (Self::Scalar(op), OpAttrs::None) => {
                op.accepts_attrs(&super::dialects::scalar::OpAttrs::None)
            }
            (Self::TensorLang(op), OpAttrs::TensorLang(attrs)) => op.accepts_attrs(attrs),
            (Self::TensorLang(op), OpAttrs::None) => {
                op.accepts_attrs(&super::dialects::tensor_lang::OpAttrs::None)
            }
            _ => false,
        }
    }
}
impl OpAttrs {
    pub fn schema_id(&self) -> Option<usize> {
        match self {
            Self::None | Self::Literal { .. } | Self::Input { .. } => None,
            Self::Scalar(attrs) => attrs.schema_id(),
            Self::TensorLang(attrs) => attrs.schema_id(),
        }
    }
    pub fn checked_schema(self, schema: usize) -> Option<Self> {
        (self.schema_id() == Some(schema)).then_some(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    Exact(usize),
    AtLeast(usize),
}
impl Arity {
    pub fn accepts(self, actual: usize) -> bool {
        match self {
            Self::Exact(n) => actual == n,
            Self::AtLeast(n) => actual >= n,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeError(pub String);
impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl ::std::error::Error for NodeError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpNode {
    op: Op,
    children: Vec<Id>,
    attrs: OpAttrs,
}
pub trait DialectOp: Sized {
    type Attrs;
    fn into_node(self, attrs: Self::Attrs, children: Vec<Id>) -> Result<OpNode, NodeError>;
}
impl OpNode {
    pub fn new<O: DialectOp>(
        op: O,
        attrs: O::Attrs,
        children: impl Into<Vec<Id>>,
    ) -> Result<Self, NodeError> {
        op.into_node(attrs, children.into())
    }
    pub fn from_parts(
        op: Op,
        children: impl Into<Vec<Id>>,
        attrs: OpAttrs,
    ) -> Result<Self, NodeError> {
        let children = children.into();
        if !op.arity().accepts(children.len()) {
            return Err(NodeError(format!(
                "{} expects {:?} operands, got {}",
                op.name(),
                op.arity(),
                children.len()
            )));
        }
        if !op.accepts_attrs(&attrs) {
            return Err(NodeError(format!("invalid attributes for {}", op.name())));
        }
        Ok(Self {
            op,
            children,
            attrs,
        })
    }
    pub fn input(name: impl Into<String>) -> Self {
        Self {
            op: Op::Input,
            children: vec![],
            attrs: OpAttrs::Input { name: name.into() },
        }
    }
    pub fn literal(value: impl Into<String>, dtype: DType) -> Result<Self, NodeError> {
        Self::from_parts(
            Op::Literal,
            vec![],
            OpAttrs::Literal {
                value: value.into(),
                dtype: Some(dtype),
            },
        )
    }
    pub fn op(&self) -> Op {
        self.op
    }
    pub fn attrs(&self) -> &OpAttrs {
        &self.attrs
    }
}
impl Language for OpNode {
    type Discriminant = Op;
    fn discriminant(&self) -> Op {
        self.op
    }
    fn matches(&self, other: &Self) -> bool {
        self.op == other.op
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
impl fmt::Display for OpNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:?}", self.op.name(), self.attrs)
    }
}

pub(crate) fn valid_literal(value: &str) -> bool {
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    let digits = |part: &str| !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit());
    match unsigned.split_once('.') {
        Some((integer, fraction)) => digits(integer) && digits(fraction),
        None => digits(unsigned),
    }
}
