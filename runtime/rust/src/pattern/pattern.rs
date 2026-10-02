use std::collections::HashSet;

use egg::{Symbol, Var};

use crate::ir::{DType, Op, OpAttrs};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttrVar(pub Symbol);

impl From<&str> for AttrVar {
    fn from(name: &str) -> Self {
        Self(name.into())
    }
}

#[derive(Clone, Debug)]
pub enum AttrPattern {
    Any,
    Literal { value: String, dtype: Option<DType> },
    Exact(OpAttrs),
    Bind(AttrVar),
}

#[derive(Clone, Debug)]
pub enum TensorPattern {
    Var(Var),
    /// Match `pattern` and bind its result e-class to `var`.
    Bind {
        var: Var,
        pattern: Box<TensorPattern>,
    },
    Op {
        op: Op,
        attrs: AttrPattern,
        children: Vec<TensorPattern>,
    },
}

impl TensorPattern {
    /// Match a numeric literal node with exactly this spelling and dtype.
    pub fn literal(value: impl Into<String>, dtype: Option<DType>) -> Self {
        Self::op(
            Op::TeplLiteral,
            AttrPattern::Literal {
                value: value.into(),
                dtype,
            },
            vec![],
        )
    }

    pub fn bind(var: Var, pattern: Self) -> Self {
        Self::Bind {
            var,
            pattern: Box::new(pattern),
        }
    }

    pub fn op(op: impl Into<Op>, attrs: AttrPattern, children: Vec<Self>) -> Self {
        Self::Op {
            op: op.into(),
            attrs,
            children,
        }
    }

    pub(super) fn vars(&self, found: &mut HashSet<Var>) {
        match self {
            Self::Var(var) => {
                found.insert(*var);
            }
            Self::Bind { var, pattern } => {
                found.insert(*var);
                pattern.vars(found);
            }
            Self::Op { children, .. } => {
                for child in children {
                    child.vars(found);
                }
            }
        }
    }

    pub(super) fn attr_vars(&self, found: &mut HashSet<AttrVar>) {
        match self {
            Self::Var(_) => {}
            Self::Bind { pattern, .. } => pattern.attr_vars(found),
            Self::Op {
                attrs, children, ..
            } => {
                if let AttrPattern::Bind(var) = attrs {
                    found.insert(*var);
                }
                for child in children {
                    child.attr_vars(found);
                }
            }
        }
    }

    pub(super) fn root_op(&self) -> Option<Op> {
        match self {
            Self::Var(_) => None,
            Self::Bind { pattern, .. } => pattern.root_op(),
            Self::Op { op, .. } => Some(*op),
        }
    }
}

#[derive(Clone, Debug)]
pub enum AttrExpr {
    Exact(OpAttrs),
    Captured(AttrVar),
    Derived(AttrVar),
}

#[derive(Clone, Debug)]
pub enum TensorExpr {
    Var(Var),
    Op {
        op: Op,
        attrs: AttrExpr,
        children: Vec<TensorExpr>,
    },
}

impl TensorExpr {
    /// Construct a numeric literal on the RHS.
    pub fn literal(value: impl Into<String>, dtype: Option<DType>) -> Self {
        Self::op(
            Op::TeplLiteral,
            AttrExpr::Exact(OpAttrs::TeplLiteral {
                value: value.into(),
                dtype,
            }),
            vec![],
        )
    }

    pub fn op(op: impl Into<Op>, attrs: AttrExpr, children: Vec<Self>) -> Self {
        Self::Op {
            op: op.into(),
            attrs,
            children,
        }
    }

    pub(super) fn vars(&self, found: &mut HashSet<Var>) {
        match self {
            Self::Var(var) => {
                found.insert(*var);
            }
            Self::Op { children, .. } => {
                for child in children {
                    child.vars(found);
                }
            }
        }
    }

    pub(super) fn captured_attrs(&self, found: &mut HashSet<AttrVar>) {
        if let Self::Op {
            attrs, children, ..
        } = self
        {
            if let AttrExpr::Captured(var) = attrs {
                found.insert(*var);
            }
            for child in children {
                child.captured_attrs(found);
            }
        }
    }
}
