use crate::ir::dialects::tensor_lang as t;
use crate::ir::types::{Elements, Region};
use crate::ir::{NodeError, Op, OpAttrs, OpNode};
use egg::Id;

/// Host graph inputs are runtime leaves, outside the TensorLang dialect.
pub fn symbol(name: impl Into<String>) -> OpNode {
    OpNode::input(name)
}
pub fn constant(value: Elements) -> OpNode {
    OpNode::new(t::Op::Constant, t::OpAttrs::ConstantAttrs { value }, vec![]).unwrap()
}
pub fn unary(op: t::Op, child: Id) -> Result<OpNode, NodeError> {
    OpNode::new(op, t::OpAttrs::None, vec![child])
}
pub fn binary(op: t::Op, lhs: Id, rhs: Id) -> Result<OpNode, NodeError> {
    OpNode::new(op, t::OpAttrs::None, vec![lhs, rhs])
}
pub fn reduce(body: Region, input: Id, init_value: Id, dimensions: Vec<u64>) -> OpNode {
    OpNode::new(
        t::Op::Reduce,
        t::OpAttrs::ReduceAttrs { dimensions, body },
        vec![input, init_value],
    )
    .unwrap()
}
pub fn symbol_name(node: &OpNode) -> Option<&str> {
    match (node.op(), node.attrs()) {
        (Op::Input, OpAttrs::Input { name }) => Some(name),
        _ => None,
    }
}
pub fn reduction(node: &OpNode) -> Option<(&Region, &[u64], Id, Id)> {
    use egg::Language;
    match (node.op(), node.attrs(), node.children()) {
        (
            Op::TensorLang(t::Op::Reduce),
            OpAttrs::TensorLang(t::OpAttrs::ReduceAttrs { body, dimensions }),
            [input, init_value],
        ) => Some((body, dimensions, *input, *init_value)),
        _ => None,
    }
}
