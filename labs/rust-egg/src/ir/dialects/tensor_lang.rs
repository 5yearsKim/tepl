use crate::ir::{Arity, DType};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
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
    Literal,
    Symbol,
}

impl Op {
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
            "literal" => Self::Literal,
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
            Self::Literal => "literal",
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
            Self::Constant | Self::Literal | Self::Symbol => Arity::Exact(0),
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
            Self::Literal => {
                matches!(attrs, OpAttrs::Literal { value, dtype } if valid_literal(value) && dtype.accepts_literal(value))
            }
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
    Literal {
        value: String,
        dtype: DType,
    },
    Symbol {
        name: String,
    },
}

// Same syntax as TEPL graph numbers: an optional sign, digits, and an optional
// decimal fraction. Do not parse through f64: identity must not depend on
// rounding, machine integer range, or an assumed tensor element type.
fn valid_literal(value: &str) -> bool {
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    let digits = |part: &str| !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit());
    match unsigned.split_once('.') {
        Some((integer, fraction)) => digits(integer) && digits(fraction),
        None => digits(unsigned),
    }
}

use crate::ir::op_node::{DialectOp, NodeError, Op as AnyOp, OpAttrs as AnyAttrs, OpNode};
impl From<Op> for AnyOp {
    fn from(op: Op) -> Self {
        Self::TensorLang(op)
    }
}
impl From<OpAttrs> for AnyAttrs {
    fn from(attrs: OpAttrs) -> Self {
        match attrs {
            OpAttrs::None => Self::None,
            attrs => Self::TensorLang(attrs),
        }
    }
}
impl DialectOp for Op {
    type Attrs = OpAttrs;
    fn into_node(self, attrs: OpAttrs, children: Vec<egg::Id>) -> Result<OpNode, NodeError> {
        OpNode::from_parts(self.into(), children, attrs.into())
    }
}
