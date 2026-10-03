//! The lab's direct egg analysis: one configuration per graph, TensorFacts per e-class.

use egg::{Analysis, DidMerge, EGraph, Id, Language};

use super::{TensorBindings, TensorFacts, infer_tensor};
use crate::ir::analysis::Inference;
use crate::ir::pattern::TensorInfo;
use crate::ir::{Op, OpNode};

/// Input shapes and dtypes, registered before building the graph.
#[derive(Clone, Debug, Default)]
pub struct ShapeAnalysis {
    pub symbols: TensorBindings,
}

impl ShapeAnalysis {
    pub fn new(symbols: TensorBindings) -> Self {
        Self { symbols }
    }
}

impl Analysis<OpNode> for ShapeAnalysis {
    type Data = TensorFacts;

    fn make(egraph: &mut EGraph<OpNode, Self>, node: &OpNode, _id: Id) -> TensorFacts {
        // Inputs get their metadata from the bindings supplied by the application.
        if node.op() == Op::Input {
            let inferred = match egraph.analysis.symbols.info(node) {
                Some(info) => Inference::Known(info.clone()),
                None => Inference::Unknown,
            };
            return TensorFacts::from_inference(inferred);
        }

        // Gather the metadata already stored on the operand e-classes.
        let mut operands = Vec::new();
        let mut has_unknown = false;
        for child in node.children() {
            let facts = &egraph[egraph.find(*child)].data;
            if facts.is_invalid() {
                return TensorFacts::from_inference(Inference::Invalid("invalid operand metadata"));
            }
            match facts.info() {
                Some(info) => operands.push(info.clone()),
                // Keep checking: a later invalid operand must take priority.
                None => has_unknown = true,
            }
        }
        if has_unknown {
            return TensorFacts::from_inference(Inference::Unknown);
        }

        // Infer this operation, then store the result on its e-class.
        TensorFacts::from_inference(infer_tensor(node.op(), &operands, node.attrs()))
    }

    fn merge(&mut self, target: &mut TensorFacts, incoming: TensorFacts) -> DidMerge {
        target.merge(incoming)
    }
}

/// Read an e-class's agreed shape and dtype; also accepted by rewrite builders.
pub fn tensor_info(egraph: &EGraph<OpNode, ShapeAnalysis>, id: Id) -> Option<TensorInfo> {
    egraph[egraph.find(id)].data.info().cloned()
}
