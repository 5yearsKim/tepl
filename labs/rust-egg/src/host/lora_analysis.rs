//! Explicit application policy for the LoRA demo's dot dtype.
//! The generated default analysis leaves undeclared dtype behavior unknown.
use egg::{Analysis, DidMerge, EGraph, Id, Language};

use crate::ir::analysis::{
    Inference, TensorAnalysisData, TensorBindingTable, TensorInfo, infer_shape, infer_tensor,
};
use crate::ir::dialects::tensor_lang as t;
use crate::ir::{DType, Op, OpAttrs, OpNode};

#[derive(Clone, Debug, Default)]
pub struct LoraAnalysis {
    pub symbols: TensorBindingTable,
}

impl LoraAnalysis {
    pub fn new(symbols: TensorBindingTable) -> Self {
        Self { symbols }
    }
}

fn infer_lora_tensor(op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Inference<TensorInfo> {
    if op != Op::TensorLang(t::Op::DotGeneral) {
        return infer_tensor(op, operands, attrs);
    }
    if !op.arity().accepts(operands.len()) || !op.accepts_attrs(attrs) {
        return Inference::Invalid("invalid dot signature");
    }
    let shapes: Vec<_> = operands.iter().map(|info| info.shape.as_slice()).collect();
    let shape = infer_shape(op, &shapes, attrs);
    if let Inference::Invalid(reason) = shape {
        return Inference::Invalid(reason);
    }
    let OpAttrs::TensorLang(t::OpAttrs::DotGeneralAttrs {
        precision_config,
        algorithm,
        ..
    }) = attrs
    else {
        unreachable!()
    };
    if algorithm.is_some()
        || precision_config.as_slice() != [crate::ir::types::Precision::Default; 2]
    {
        return Inference::Unknown;
    }
    let [lhs, rhs] = operands else { unreachable!() };
    if lhs.dtype != rhs.dtype || lhs.dtype == DType::Bool {
        return Inference::Invalid("demo dot requires matching numeric dtypes");
    }
    match shape {
        Inference::Known(shape) => Inference::Known(TensorInfo {
            shape,
            dtype: lhs.dtype,
        }),
        Inference::Unknown => Inference::Unknown,
        Inference::Invalid(reason) => Inference::Invalid(reason),
    }
}

/// Rewrite validation uses the same explicit policy as the demo's analysis.
pub fn infer_lora_tensor_output(
    op: Op,
    operands: &[TensorInfo],
    attrs: &OpAttrs,
) -> Option<TensorInfo> {
    infer_lora_tensor(op, operands, attrs).into_option()
}

pub fn lora_tensor_info(graph: &EGraph<OpNode, LoraAnalysis>, id: Id) -> Option<TensorInfo> {
    graph[graph.find(id)].data.info().cloned()
}

impl Analysis<OpNode> for LoraAnalysis {
    type Data = TensorAnalysisData;

    fn make(graph: &mut EGraph<OpNode, Self>, node: &OpNode, _id: Id) -> Self::Data {
        if node.op() == Op::Input {
            return Self::Data::from_inference(
                graph
                    .analysis
                    .symbols
                    .info(node)
                    .cloned()
                    .map(Inference::Known)
                    .unwrap_or(Inference::Unknown),
            );
        }
        let mut operands = Vec::new();
        let mut unknown = false;
        for child in node.children() {
            let facts = &graph[graph.find(*child)].data;
            if facts.is_invalid() {
                return Self::Data::from_inference(Inference::Invalid("invalid operand metadata"));
            }
            match facts.info() {
                Some(info) => operands.push(info.clone()),
                None => unknown = true,
            }
        }
        Self::Data::from_inference(if unknown {
            Inference::Unknown
        } else {
            infer_lora_tensor(node.op(), &operands, node.attrs())
        })
    }

    fn merge(&mut self, target: &mut Self::Data, incoming: Self::Data) -> DidMerge {
        target.merge(incoming)
    }
}
