//! Direct egg analysis: one configuration per graph, TensorAnalysisData per e-class.

use ::egg::{Analysis, DidMerge, EGraph, Id, Language};

use super::super::pattern::TensorInfo;
use super::super::{Op, OpNode};
use super::Inference;
use super::{TensorAnalysisData, TensorBindingTable, infer_tensor};

/// Input shapes and dtypes, registered before building the graph.
#[derive(Clone, Debug, Default)]
pub struct TensorAnalysis {
    pub symbols: TensorBindingTable,
}

impl TensorAnalysis {
    pub fn new(symbols: TensorBindingTable) -> Self {
        Self { symbols }
    }
}

impl Analysis<OpNode> for TensorAnalysis {
    type Data = TensorAnalysisData;

    fn make(egraph: &mut EGraph<OpNode, Self>, node: &OpNode, _id: Id) -> TensorAnalysisData {
        // Inputs get their metadata from the bindings supplied by the application.
        if node.op() == Op::Input {
            let inferred = match egraph.analysis.symbols.info(node) {
                Some(info) => Inference::Known(info.clone()),
                None => Inference::Unknown,
            };
            return TensorAnalysisData::from_inference(inferred);
        }

        // Gather the metadata already stored on the operand e-classes.
        let mut operands = Vec::new();
        let mut has_unknown = false;
        for child in node.children() {
            let facts = &egraph[egraph.find(*child)].data;
            if facts.is_invalid() {
                return TensorAnalysisData::from_inference(Inference::Invalid(
                    "invalid operand metadata",
                ));
            }
            match facts.info() {
                Some(info) => operands.push(info.clone()),
                // Keep checking: a later invalid operand must take priority.
                None => has_unknown = true,
            }
        }
        if has_unknown {
            return TensorAnalysisData::from_inference(Inference::Unknown);
        }

        // Infer this operation, then store the result on its e-class.
        TensorAnalysisData::from_inference(infer_tensor(node.op(), &operands, node.attrs()))
    }

    fn merge(&mut self, target: &mut TensorAnalysisData, incoming: TensorAnalysisData) -> DidMerge {
        target.merge(incoming)
    }
}

/// Read an e-class's agreed shape and dtype; also accepted by rewrite builders.
pub fn tensor_info(egraph: &EGraph<OpNode, TensorAnalysis>, id: Id) -> Option<TensorInfo> {
    egraph[egraph.find(id)].data.info().cloned()
}
