use crate::{Node, Program, TensorAnalysis};
use egg::{EGraph, Id};
use std::collections::BTreeMap;
pub struct BuiltGraph<P: Program> {
    pub graph: EGraph<Node<P>, TensorAnalysis<P>>,
    pub root: Id,
    pub bindings: BTreeMap<String, Id>,
}
