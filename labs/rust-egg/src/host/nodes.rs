use crate::ir::dialects::tensor_lang as t;
use crate::ir::{NodeError, Op, OpAttrs, OpNode};
use egg::Id;

pub fn symbol(name: impl Into<String>) -> OpNode {
    OpNode::new(
        t::Op::Symbol,
        t::OpAttrs::SymbolAttrs { name: name.into() },
        vec![],
    )
    .unwrap()
}
pub fn constant(name: impl Into<String>) -> OpNode {
    OpNode::new(
        t::Op::Constant,
        t::OpAttrs::ConstantAttrs { name: name.into() },
        vec![],
    )
    .unwrap()
}
pub fn unary(op: t::Op, child: Id) -> Result<OpNode, NodeError> {
    OpNode::new(op, t::OpAttrs::None, vec![child])
}
pub fn binary(op: t::Op, lhs: Id, rhs: Id) -> Result<OpNode, NodeError> {
    OpNode::new(op, t::OpAttrs::None, vec![lhs, rhs])
}
pub fn reduce(kind: impl Into<String>, input: Id, axes: Vec<u64>) -> OpNode {
    OpNode::new(
        t::Op::Reduce,
        t::OpAttrs::ReduceAttrs {
            kind: kind.into(),
            axes: axes,
        },
        vec![input],
    )
    .unwrap()
}
pub fn symbol_name(node: &OpNode) -> Option<&str> {
    match (node.op(), node.attrs()) {
        (Op::TensorLang(t::Op::Symbol), OpAttrs::TensorLang(t::OpAttrs::SymbolAttrs { name })) => {
            Some(name)
        }
        _ => None,
    }
}
pub fn reduction(node: &OpNode) -> Option<(&str, &[u64], Id)> {
    use egg::Language;
    match (node.op(), node.attrs(), node.children()) {
        (
            Op::TensorLang(t::Op::Reduce),
            OpAttrs::TensorLang(t::OpAttrs::ReduceAttrs { kind, axes }),
            [input],
        ) => Some((kind, axes, *input)),
        _ => None,
    }
}
