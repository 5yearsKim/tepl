//! Tensor e-nodes for `egg`. Attributes describe an operation; children are e-class IDs.

use std::fmt;

use egg::{Id, Language, Symbol};

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
    RmsNorm,
    VocabCrossEntropy,
    OnlineAttention,
    Constant,
    Symbol,
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

impl OpKind {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "add" => Self::Add,
            "subtract" => Self::Subtract,
            "multiply" => Self::Multiply,
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
            "dot_general" => Self::DotGeneral,
            "convolution" => Self::Convolution,
            "all_gather" => Self::AllGather,
            "all_reduce" => Self::AllReduce,
            "reduce_scatter" => Self::ReduceScatter,
            "all_to_all" => Self::AllToAll,
            "alias" => Self::Alias,
            "rmsnorm" => Self::RmsNorm,
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
            Self::RmsNorm => "rmsnorm",
            Self::VocabCrossEntropy => "vocab_cross_entropy",
            Self::OnlineAttention => "online_attention",
            Self::Constant => "constant",
            Self::Symbol => "symbol",
        }
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
            | Self::RmsNorm => Arity::Exact(1),
            Self::Concatenate => Arity::AtLeast(2),
            Self::Constant | Self::Symbol => Arity::Exact(0),
        }
    }

    fn accepts_attrs(self, attrs: &OpAttrs) -> bool {
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
            Self::Symbol => matches!(attrs, OpAttrs::Symbol(_)),
            Self::Constant => matches!(attrs, OpAttrs::NamedConstant(_)),
            _ => matches!(attrs, OpAttrs::None),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpAttrs {
    None,
    Reduce {
        kind: Symbol,
        axes: Vec<usize>,
    },
    CollectiveReduce {
        kind: Symbol,
    },
    Transpose {
        permutation: Vec<usize>,
    },
    Reshape {
        shape: Vec<usize>,
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
    DotGeneral {
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
    },
    Symbol(Symbol),
    NamedConstant(Symbol),
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

    pub fn reduce(kind: impl Into<Symbol>, value: Id, axes: Vec<usize>) -> Self {
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

    pub fn symbol(symbol: impl Into<Symbol>) -> Self {
        Self::new(OpKind::Symbol, vec![], OpAttrs::Symbol(symbol.into()))
            .expect("a symbol node has valid arity and attributes")
    }

    pub fn op(&self) -> OpKind {
        self.op
    }

    pub fn attrs(&self) -> &OpAttrs {
        &self.attrs
    }

    pub fn symbol_name(&self) -> Option<Symbol> {
        match &self.attrs {
            OpAttrs::Symbol(symbol) if self.op == OpKind::Symbol => Some(*symbol),
            _ => None,
        }
    }

    pub fn reduction(&self) -> Option<(Symbol, &[usize], Id)> {
        match (&self.op, &self.attrs, self.children.as_slice()) {
            (OpKind::Reduce, OpAttrs::Reduce { kind, axes }, [input]) => {
                Some((*kind, axes, *input))
            }
            _ => None,
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

#[cfg(test)]
mod tests {
    use egg::{EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, Searcher};

    use super::{Arity, NodeError, OpAttrs, OpKind, TensorLang};

    #[test]
    fn construction_rejects_bad_arity_and_attributes() {
        let x = Id::from(0);
        assert_eq!(
            TensorLang::new(OpKind::Add, vec![x], OpAttrs::None),
            Err(NodeError::Arity {
                op: OpKind::Add,
                expected: Arity::Exact(2),
                actual: 1,
            })
        );
        assert_eq!(
            TensorLang::new(OpKind::Reduce, vec![x], OpAttrs::None),
            Err(NodeError::Attributes { op: OpKind::Reduce })
        );
        assert_eq!(
            TensorLang::new(
                OpKind::Add,
                vec![x, x],
                OpAttrs::Reduce {
                    kind: "sum".into(),
                    axes: vec![0],
                }
            ),
            Err(NodeError::Attributes { op: OpKind::Add })
        );
    }

    #[test]
    fn reduction_pattern_matches_kind_and_axes() {
        let mut egraph = EGraph::<TensorLang, ()>::default();
        let x = egraph.add(TensorLang::symbol("x"));
        let sum_axis_0 = egraph.add(TensorLang::reduce("sum", x, vec![0]));
        egraph.add(TensorLang::reduce("sum", x, vec![1]));
        egraph.add(TensorLang::reduce("max", x, vec![0]));
        egraph.rebuild();

        let mut ast = PatternAst::default();
        ast.add(ENodeOrVar::Var("?x".parse().unwrap()));
        ast.add(ENodeOrVar::ENode(TensorLang::reduce(
            "sum",
            Id::from(0),
            vec![0],
        )));
        let pattern = Pattern::new(ast);
        let matches = pattern.search(&egraph);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].eclass, egraph.find(sum_axis_0));
        assert_eq!(
            TensorLang::reduce("sum", x, vec![0]).discriminant(),
            OpKind::Reduce
        );
    }
}
