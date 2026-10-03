//! Metadata lookup for fixtures using EGraph<OpNode, ()> without ShapeAnalysis.
//! Graphs with ShapeAnalysis should read the cached facts through tensor_info.
use super::infer_tensor_output;
use crate::ir::pattern::TensorInfo;

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
