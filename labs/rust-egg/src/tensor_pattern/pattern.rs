use std::collections::HashSet;

use egg::{Symbol, Var};

use crate::ir::{OpAttrs, OpKind};

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
    Exact(OpAttrs),
    Bind(AttrVar),
}

#[derive(Clone, Debug)]
pub enum TensorPattern {
    Var(Var),
    Op {
        op: OpKind,
        attrs: AttrPattern,
        children: Vec<TensorPattern>,
    },
}

impl TensorPattern {
    pub fn op(op: OpKind, attrs: AttrPattern, children: Vec<Self>) -> Self {
        Self::Op {
            op,
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

    pub(super) fn attr_vars(&self, found: &mut HashSet<AttrVar>) {
        if let Self::Op {
            attrs, children, ..
        } = self
        {
            if let AttrPattern::Bind(var) = attrs {
                found.insert(*var);
            }
            for child in children {
                child.attr_vars(found);
            }
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
        op: OpKind,
        attrs: AttrExpr,
        children: Vec<TensorExpr>,
    },
}

impl TensorExpr {
    pub fn op(op: OpKind, attrs: AttrExpr, children: Vec<Self>) -> Self {
        Self::Op {
            op,
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
