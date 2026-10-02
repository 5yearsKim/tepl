use super::DType;
use super::dialects::{scalar, tensor_lang};
use egg::{Id, Language};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    TeplLiteral,
    Scalar(scalar::Op),
    TensorLang(tensor_lang::Op),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpAttrs {
    None,
    TeplLiteral { value: String, dtype: Option<DType> },
    Scalar(scalar::OpAttrs),
    TensorLang(tensor_lang::OpAttrs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    Exact(usize),
    AtLeast(usize),
}

impl Arity {
    pub fn accepts(self, actual: usize) -> bool {
        match self {
            Self::Exact(expected) => actual == expected,
            Self::AtLeast(minimum) => actual >= minimum,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeError {
    Arity {
        op: Op,
        expected: Arity,
        actual: usize,
    },
    Attributes {
        op: Op,
    },
}

impl fmt::Display for NodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arity {
                op,
                expected,
                actual,
            } => write!(
                formatter,
                "{} expects {expected:?} operands, got {actual}",
                op.name()
            ),
            Self::Attributes { op } => {
                write!(formatter, "invalid attributes for {}", op.name())
            }
        }
    }
}

impl std::error::Error for NodeError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpNode {
    op: Op,
    children: Vec<Id>,
    attrs: OpAttrs,
}

impl OpNode {
    pub fn from_parts(
        op: Op,
        children: impl Into<Vec<Id>>,
        attrs: OpAttrs,
    ) -> Result<Self, NodeError> {
        let children = children.into();
        let expected = op.arity();
        if !expected.accepts(children.len()) {
            return Err(NodeError::Arity {
                op,
                expected,
                actual: children.len(),
            });
        }
        if !op.accepts_attrs(&attrs) {
            return Err(NodeError::Attributes { op });
        }
        Ok(Self {
            op,
            children,
            attrs,
        })
    }

    pub fn unary(op: Op, child: Id) -> Result<Self, NodeError> {
        Self::from_parts(op, vec![child], OpAttrs::None)
    }

    pub fn binary(op: Op, left: Id, right: Id) -> Result<Self, NodeError> {
        Self::from_parts(op, vec![left, right], OpAttrs::None)
    }

    pub fn reduce(kind: impl Into<String>, value: Id, axes: Vec<usize>) -> Self {
        Self::from_parts(
            Op::TensorLang(tensor_lang::Op::Reduce),
            vec![value],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Reduce {
                kind: kind.into(),
                axes,
            }),
        )
        .expect("a reduce node has valid arity and attributes")
    }

    pub fn symbol(name: impl Into<String>) -> Self {
        Self::from_parts(
            Op::TensorLang(tensor_lang::Op::Symbol),
            vec![],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Symbol { name: name.into() }),
        )
        .expect("a symbol node has valid arity and attributes")
    }

    pub fn constant(name: impl Into<String>) -> Self {
        Self::from_parts(
            Op::TensorLang(tensor_lang::Op::Constant),
            vec![],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Constant { name: name.into() }),
        )
        .expect("a constant node has valid arity and attributes")
    }

    /// A rank-zero numeric literal. Spelling is preserved, so `1` and `1.0`
    /// remain distinct within each dtype. Broadcasting and float rounding are
    /// host semantics; dtype is part of the node identity.
    pub fn literal(value: impl Into<String>, dtype: DType) -> Result<Self, NodeError> {
        Self::from_parts(
            Op::TensorLang(tensor_lang::Op::Literal),
            vec![],
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Literal {
                value: value.into(),
                dtype,
            }),
        )
    }

    pub fn op(&self) -> Op {
        self.op
    }

    pub fn attrs(&self) -> &OpAttrs {
        &self.attrs
    }

    pub fn symbol_name(&self) -> Option<&str> {
        match &self.attrs {
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Symbol { name })
                if self.op == Op::TensorLang(tensor_lang::Op::Symbol) =>
            {
                Some(name)
            }
            _ => None,
        }
    }

    pub fn reduction(&self) -> Option<(&str, &[usize], Id)> {
        match (&self.op, &self.attrs, self.children.as_slice()) {
            (
                Op::TensorLang(tensor_lang::Op::Reduce),
                OpAttrs::TensorLang(tensor_lang::OpAttrs::Reduce { kind, axes }),
                [input],
            ) => Some((kind, axes, *input)),
            _ => None,
        }
    }
}

impl fmt::Display for OpNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.attrs {
            OpAttrs::None => write!(formatter, "{}", self.op.display_name()),
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Symbol { name })
            | OpAttrs::TensorLang(tensor_lang::OpAttrs::Constant { name }) => {
                write!(formatter, "{name}")
            }
            OpAttrs::TensorLang(tensor_lang::OpAttrs::Literal { value, dtype }) => {
                write!(formatter, "{value}:{dtype}")
            }
            OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            }) => write!(
                formatter,
                "dot(lc={lhs_contracting:?},rc={rhs_contracting:?},lb={lhs_batch:?},rb={rhs_batch:?})"
            ),
            attrs => write!(formatter, "{} {attrs:?}", self.op.display_name()),
        }
    }
}

impl Language for OpNode {
    type Discriminant = Op;

    fn discriminant(&self) -> Self::Discriminant {
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

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Self::TeplLiteral => "<literal>",
            Self::Scalar(op) => op.name(),
            Self::TensorLang(op) => op.name(),
        }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        if name == "<literal>" {
            return Some(Self::TeplLiteral);
        }
        let (dialect, op) = name.split_once('.')?;
        match dialect {
            "Scalar" => scalar::Op::from_name(op).map(Self::Scalar),
            "TensorLang" => tensor_lang::Op::from_name(op).map(Self::TensorLang),
            _ => None,
        }
    }
    pub fn arity(self) -> Arity {
        match self {
            Self::TeplLiteral => Arity::Exact(0),
            Self::Scalar(op) => op.arity(),
            Self::TensorLang(op) => op.arity(),
        }
    }
    pub fn accepts_attrs(self, attrs: &OpAttrs) -> bool {
        match (self, attrs) {
            (Self::TeplLiteral, OpAttrs::TeplLiteral { value, dtype }) => {
                valid_literal(value) && dtype.is_none_or(|d| d.accepts_literal(value))
            }
            (Self::Scalar(op), OpAttrs::Scalar(attrs)) => op.accepts_attrs(attrs),
            (Self::Scalar(op), OpAttrs::None) => op.accepts_attrs(&scalar::OpAttrs::None),
            (Self::TensorLang(op), OpAttrs::TensorLang(attrs)) => op.accepts_attrs(attrs),
            (Self::TensorLang(op), OpAttrs::None) => op.accepts_attrs(&tensor_lang::OpAttrs::None),
            _ => false,
        }
    }
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
}
impl Op {
    pub fn alias(self) -> Option<&'static str> {
        match self {
            Self::TensorLang(op) => op.alias(),
            _ => None,
        }
    }
    pub fn display_name(self) -> &'static str {
        self.alias().unwrap_or(self.name())
    }
}
fn valid_literal(value: &str) -> bool {
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    let digits = |part: &str| !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit());
    match unsigned.split_once('.') {
        Some((a, b)) => digits(a) && digits(b),
        None => digits(unsigned),
    }
}
