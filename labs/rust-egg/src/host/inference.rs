use crate::ir::dialects::tensor_lang;
use crate::ir::pattern::TensorInfo;
use crate::ir::{DType, Op, OpAttrs};
/// Shared semantics for e-class analysis and pre-insertion RHS verification.
pub fn infer_tensor_output(op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
    if let (
        Op::Literal,
        [],
        OpAttrs::Literal {
            dtype: Some(dtype), ..
        },
    ) = (op, operands, attrs)
    {
        return Some(TensorInfo {
            shape: vec![],
            dtype: *dtype,
        });
    }
    if let (
        Op::TensorLang(tensor_lang::Op::Constant),
        [],
        OpAttrs::TensorLang(tensor_lang::OpAttrs::ConstantAttrs { value }),
    ) = (op, operands, attrs)
    {
        return Some(TensorInfo {
            shape: value.shape.clone(),
            dtype: value.element_type.parse().ok()?,
        });
    }
    let [lhs, rhs] = operands else {
        return None;
    };
    if lhs.dtype != rhs.dtype || lhs.dtype == DType::Bool {
        return None;
    }
    let shape = match (op, attrs) {
        (Op::TensorLang(tensor_lang::Op::Add | tensor_lang::Op::Multiply), OpAttrs::None) => {
            if lhs.shape != rhs.shape {
                return None;
            }
            lhs.shape.clone()
        }
        (Op::TensorLang(tensor_lang::Op::DotGeneral), _) => {
            batched_dot_shape(&lhs.shape, &rhs.shape, attrs)?
        }
        _ => return None,
    };
    Some(TensorInfo {
        shape,
        dtype: lhs.dtype,
    })
}

pub fn dot_attrs() -> OpAttrs {
    OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting_dimensions: vec![2],
        rhs_contracting_dimensions: vec![1],
        lhs_batching_dimensions: vec![0],
        rhs_batching_dimensions: vec![0],
        precision_config: vec![crate::ir::types::Precision::Default; 2],
        algorithm: None,
    })
}

pub fn batched_dot_shape(lhs: &[u64], rhs: &[u64], attrs: &OpAttrs) -> Option<Vec<u64>> {
    if lhs.len() != 3
        || rhs.len() != 3
        || *attrs != dot_attrs()
        || lhs[0] != rhs[0]
        || lhs[2] != rhs[1]
    {
        return None;
    }
    Some(vec![lhs[0], lhs[1], rhs[2]])
}

/// Infer metadata from every node alternative in an acyclic graph.
/// Missing metadata, cycles, and disagreeing alternatives return None.
pub fn infer_eclass<N: egg::Analysis<crate::ir::OpNode>>(
    graph: &egg::EGraph<crate::ir::OpNode, N>,
    id: egg::Id,
    input: &impl Fn(&crate::ir::OpNode) -> Option<TensorInfo>,
) -> Option<TensorInfo> {
    fn visit<N: egg::Analysis<crate::ir::OpNode>>(
        graph: &egg::EGraph<crate::ir::OpNode, N>,
        id: egg::Id,
        input: &impl Fn(&crate::ir::OpNode) -> Option<TensorInfo>,
        visiting: &mut std::collections::HashSet<egg::Id>,
    ) -> Option<TensorInfo> {
        use egg::Language;
        let id = graph.find(id);
        if !visiting.insert(id) {
            return None;
        }
        let mut result = None;
        for node in &graph[id].nodes {
            let info = if let Some(info) = input(node) {
                info
            } else {
                let operands = node
                    .children()
                    .iter()
                    .map(|&child| visit(graph, child, input, visiting))
                    .collect::<Option<Vec<_>>>()?;
                infer_tensor_output(node.op(), &operands, node.attrs())?
            };
            if result.as_ref().is_some_and(|previous| previous != &info) {
                return None;
            }
            result = Some(info);
        }
        visiting.remove(&id);
        result
    }
    visit(graph, id, input, &mut Default::default())
}
