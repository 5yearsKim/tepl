use super::{TensorBindings, infer_tensor_output};
use crate::ir::dialects::tensor_lang;
use crate::ir::pattern::TensorInfo;
use crate::ir::{Op, OpNode};
use egg::{Analysis, DidMerge, EGraph, Id, Language};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    Unknown,
    Known(TensorInfo),
    Invalid,
}

#[derive(Default)]
pub struct ShapeAnalysis {
    pub symbols: TensorBindings,
}

impl Analysis<OpNode> for ShapeAnalysis {
    type Data = Shape;

    fn make(egraph: &mut EGraph<OpNode, Self>, node: &OpNode, _id: Id) -> Shape {
        if matches!(
            node.op(),
            Op::TensorLang(tensor_lang::Op::Symbol) | Op::TensorLang(tensor_lang::Op::Constant)
        ) {
            return egraph
                .analysis
                .symbols
                .info(node)
                .cloned()
                .map(Shape::Known)
                .unwrap_or(Shape::Unknown);
        }

        let children: Vec<_> = node
            .children()
            .iter()
            .map(|id| &egraph[egraph.find(*id)].data)
            .collect();
        if children.iter().any(|shape| **shape == Shape::Invalid) {
            return Shape::Invalid;
        }
        if children.iter().any(|shape| **shape == Shape::Unknown) {
            return Shape::Unknown;
        }
        let known: Vec<TensorInfo> = children
            .iter()
            .map(|shape| match shape {
                Shape::Known(info) => info.clone(),
                _ => unreachable!(),
            })
            .collect();
        let result = infer_tensor_output(node.op(), &known, node.attrs());
        result.map(Shape::Known).unwrap_or(Shape::Invalid)
    }

    fn merge(&mut self, target: &mut Shape, incoming: Shape) -> DidMerge {
        let merged = match (&*target, &incoming) {
            (Shape::Invalid, _) | (_, Shape::Invalid) => Shape::Invalid,
            // A known alternative cannot establish the shape of an unknown one.
            (Shape::Unknown, _) | (_, Shape::Unknown) => Shape::Unknown,
            (Shape::Known(a), Shape::Known(b)) if a == b => Shape::Known(a.clone()),
            (Shape::Known(_), Shape::Known(_)) => Shape::Invalid,
        };
        let changed_target = *target != merged;
        let changed_incoming = incoming != merged;
        *target = merged;
        DidMerge(changed_target, changed_incoming)
    }
}

pub fn tensor_info(egraph: &EGraph<OpNode, ShapeAnalysis>, id: Id) -> Option<TensorInfo> {
    match &egraph[egraph.find(id)].data {
        Shape::Known(info) => Some(info.clone()),
        _ => None,
    }
}
