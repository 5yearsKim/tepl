//! TensorLang dialect definitions and its `egg::Language` node implementation.
//!
//! Reference output for the future TEPL dialect generator. Mirrors
//! `examples/dialects/tensor.tepl` and is maintained by hand until generation exists.

use std::fmt;

use egg::{Id, Language};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpKind {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Negate,
    Exp,
    Log,
    Abs,
    Relu,
    Scale,
    Square,
    Sqrt,
    Rsqrt,
    Reshape,
    Transpose,
    Broadcast,
    Slice,
    Concatenate,
    Reduce,
    DotGeneral,
    Convolution,
    AllGather,
    AllReduce,
    ReduceScatter,
    AllToAll,
    Alias,
    Rmsnorm,
    VocabCrossEntropy,
    OnlineAttention,
    Constant,
    Symbol,
}

impl OpKind {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "add" => Self::Add,
            "subtract" => Self::Subtract,
            "multiply" | "mul" => Self::Multiply,
            "divide" => Self::Divide,
            "maximum" => Self::Maximum,
            "minimum" => Self::Minimum,
            "negate" => Self::Negate,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "abs" => Self::Abs,
            "relu" => Self::Relu,
            "scale" => Self::Scale,
            "square" => Self::Square,
            "sqrt" => Self::Sqrt,
            "rsqrt" => Self::Rsqrt,
            "reshape" => Self::Reshape,
            "transpose" => Self::Transpose,
            "broadcast" => Self::Broadcast,
            "slice" => Self::Slice,
            "concatenate" => Self::Concatenate,
            "reduce" => Self::Reduce,
            "dot_general" | "dot" => Self::DotGeneral,
            "convolution" => Self::Convolution,
            "all_gather" => Self::AllGather,
            "all_reduce" => Self::AllReduce,
            "reduce_scatter" => Self::ReduceScatter,
            "all_to_all" => Self::AllToAll,
            "alias" => Self::Alias,
            "rmsnorm" => Self::Rmsnorm,
            "vocab_cross_entropy" => Self::VocabCrossEntropy,
            "online_attention" => Self::OnlineAttention,
            "constant" => Self::Constant,
            "symbol" => Self::Symbol,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::Negate => "negate",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Abs => "abs",
            Self::Relu => "relu",
            Self::Scale => "scale",
            Self::Square => "square",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Reshape => "reshape",
            Self::Transpose => "transpose",
            Self::Broadcast => "broadcast",
            Self::Slice => "slice",
            Self::Concatenate => "concatenate",
            Self::Reduce => "reduce",
            Self::DotGeneral => "dot_general",
            Self::Convolution => "convolution",
            Self::AllGather => "all_gather",
            Self::AllReduce => "all_reduce",
            Self::ReduceScatter => "reduce_scatter",
            Self::AllToAll => "all_to_all",
            Self::Alias => "alias",
            Self::Rmsnorm => "rmsnorm",
            Self::VocabCrossEntropy => "vocab_cross_entropy",
            Self::OnlineAttention => "online_attention",
            Self::Constant => "constant",
            Self::Symbol => "symbol",
        }
    }

    pub fn alias(self) -> Option<&'static str> {
        match self {
            Self::Multiply => Some("mul"),
            Self::DotGeneral => Some("dot"),
            _ => None,
        }
    }

    pub fn display_name(self) -> &'static str {
        self.alias().unwrap_or(self.name())
    }

    pub fn arity(self) -> Arity {
        match self {
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Maximum
            | Self::Minimum
            | Self::Scale
            | Self::DotGeneral
            | Self::Convolution
            | Self::VocabCrossEntropy
            | Self::OnlineAttention => Arity::Exact(2),
            Self::Negate
            | Self::Exp
            | Self::Log
            | Self::Abs
            | Self::Relu
            | Self::Square
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Reshape
            | Self::Transpose
            | Self::Broadcast
            | Self::Slice
            | Self::Reduce
            | Self::AllGather
            | Self::AllReduce
            | Self::ReduceScatter
            | Self::AllToAll
            | Self::Alias
            | Self::Rmsnorm => Arity::Exact(1),
            Self::Concatenate => Arity::AtLeast(2),
            Self::Constant | Self::Symbol => Arity::Exact(0),
        }
    }

    pub(crate) fn accepts_attrs(self, attrs: &OpAttrs) -> bool {
        match self {
            Self::Reduce => matches!(attrs, OpAttrs::Reduce { .. }),
            Self::AllReduce | Self::ReduceScatter => {
                matches!(attrs, OpAttrs::CollectiveReduce { .. })
            }
            Self::Transpose => matches!(attrs, OpAttrs::Transpose { .. }),
            Self::Reshape => matches!(attrs, OpAttrs::Reshape { .. }),
            Self::Broadcast => matches!(attrs, OpAttrs::Broadcast { .. }),
            Self::Slice => matches!(attrs, OpAttrs::Slice { .. }),
            Self::Concatenate => matches!(attrs, OpAttrs::Concatenate { .. }),
            Self::DotGeneral => matches!(attrs, OpAttrs::DotGeneral { .. }),
            Self::Symbol => matches!(attrs, OpAttrs::Symbol { .. }),
            Self::Constant => matches!(attrs, OpAttrs::Constant { .. }),
            _ => matches!(attrs, OpAttrs::None),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpAttrs {
    None,
    // attrs CollectiveReduce is shared by all_reduce and reduce_scatter.
    CollectiveReduce {
        kind: String,
    },
    Reshape {
        shape: Vec<usize>,
    },
    Transpose {
        permutation: Vec<usize>,
    },
    Broadcast {
        dimensions: Vec<usize>,
        shape: Vec<usize>,
    },
    Slice {
        start: Vec<usize>,
        limit: Vec<usize>,
        strides: Vec<usize>,
    },
    Concatenate {
        axis: usize,
    },
    Reduce {
        kind: String,
        axes: Vec<usize>,
    },
    DotGeneral {
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
    },
    Constant {
        name: String,
    },
    Symbol {
        name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeError {
    Arity {
        op: OpKind,
        expected: Arity,
        actual: usize,
    },
    Attributes {
        op: OpKind,
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
pub struct TensorLang {
    op: OpKind,
    children: Vec<Id>,
    attrs: OpAttrs,
}

impl TensorLang {
    pub fn new(
        op: OpKind,
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

    pub fn unary(op: OpKind, child: Id) -> Result<Self, NodeError> {
        Self::new(op, vec![child], OpAttrs::None)
    }

    pub fn binary(op: OpKind, left: Id, right: Id) -> Result<Self, NodeError> {
        Self::new(op, vec![left, right], OpAttrs::None)
    }

    pub fn reduce(kind: impl Into<String>, value: Id, axes: Vec<usize>) -> Self {
        Self::new(
            OpKind::Reduce,
            vec![value],
            OpAttrs::Reduce {
                kind: kind.into(),
                axes,
            },
        )
        .expect("a reduce node has valid arity and attributes")
    }

    pub fn symbol(name: impl Into<String>) -> Self {
        Self::new(
            OpKind::Symbol,
            vec![],
            OpAttrs::Symbol { name: name.into() },
        )
        .expect("a symbol node has valid arity and attributes")
    }

    pub fn constant(name: impl Into<String>) -> Self {
        Self::new(
            OpKind::Constant,
            vec![],
            OpAttrs::Constant { name: name.into() },
        )
        .expect("a constant node has valid arity and attributes")
    }

    pub fn op(&self) -> OpKind {
        self.op
    }

    pub fn attrs(&self) -> &OpAttrs {
        &self.attrs
    }

    pub fn symbol_name(&self) -> Option<&str> {
        match &self.attrs {
            OpAttrs::Symbol { name } if self.op == OpKind::Symbol => Some(name),
            _ => None,
        }
    }

    pub fn reduction(&self) -> Option<(&str, &[usize], Id)> {
        match (&self.op, &self.attrs, self.children.as_slice()) {
            (OpKind::Reduce, OpAttrs::Reduce { kind, axes }, [input]) => Some((kind, axes, *input)),
            _ => None,
        }
    }
}

impl fmt::Display for TensorLang {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.attrs {
            OpAttrs::None => write!(formatter, "{}", self.op.display_name()),
            OpAttrs::Symbol { name } | OpAttrs::Constant { name } => {
                write!(formatter, "{name}")
            }
            OpAttrs::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => write!(
                formatter,
                "dot(lc={lhs_contracting:?},rc={rhs_contracting:?},lb={lhs_batch:?},rb={rhs_batch:?})"
            ),
            attrs => write!(formatter, "{} {attrs:?}", self.op.display_name()),
        }
    }
}

impl Language for TensorLang {
    type Discriminant = OpKind;

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
