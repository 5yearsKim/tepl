use super::super::analysis::TensorInfo;
use super::super::analysis::{
    Inference, TensorAnalysis, TensorBindingTable, infer_dtype, infer_shape,
};
use super::super::{DType, NodeError, Op, OpAttrs, OpNode};
use ::egg::{Analysis, EGraph, Id, Language};
use ::std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct InputSpec {
    pub node: usize,
    pub name: String,
    pub shape: Option<Vec<u64>>,
    pub dtype: Option<DType>,
}

/// IDs in prepared nodes are local indices, translated during insertion.
#[derive(Clone, Debug)]
pub struct GraphDefinition {
    nodes: Vec<OpNode>,
    inputs: Vec<InputSpec>,
    bindings: BTreeMap<String, usize>,
    root: usize,
}
#[derive(Clone, Debug)]
pub struct BuiltGraph {
    pub root: Id,
    bindings: BTreeMap<String, Id>,
}
impl BuiltGraph {
    /// Handles may become noncanonical after later unions; use graph.find(id).
    pub fn get(&self, name: &str) -> Option<Id> {
        self.bindings.get(name).copied()
    }
}
impl GraphDefinition {
    pub fn new(
        nodes: Vec<OpNode>,
        inputs: Vec<InputSpec>,
        bindings: BTreeMap<String, usize>,
        root: usize,
    ) -> Result<Self, NodeError> {
        let result = Self {
            nodes,
            inputs,
            bindings,
            root,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn nodes(&self) -> &[OpNode] {
        &self.nodes
    }
    pub fn inputs(&self) -> &[InputSpec] {
        &self.inputs
    }
    pub fn root(&self) -> usize {
        self.root
    }
    pub fn get(&self, name: &str) -> Option<usize> {
        self.bindings.get(name).copied()
    }

    /// Supply host metadata before constructing an e-graph. Declared facts must agree.
    pub fn with_input_info(mut self, name: &str, info: TensorInfo) -> Result<Self, NodeError> {
        let input = self
            .inputs
            .iter_mut()
            .find(|input| input.name == name)
            .ok_or_else(|| NodeError(format!("unknown graph input '{name}'")))?;
        if input
            .shape
            .as_ref()
            .is_some_and(|shape| *shape != info.shape)
            || input.dtype.is_some_and(|dtype| dtype != info.dtype)
        {
            return Err(NodeError(format!(
                "conflicting metadata for graph input '{name}'"
            )));
        }
        input.shape = Some(info.shape);
        input.dtype = Some(info.dtype);
        self.validate()?;
        Ok(self)
    }
    pub fn input_bindings(&self) -> Result<TensorBindingTable, NodeError> {
        let mut bindings = TensorBindingTable::default();
        for input in &self.inputs {
            if let (Some(shape), Some(dtype)) = (&input.shape, input.dtype) {
                bindings
                    .register_symbol(
                        &input.name,
                        TensorInfo {
                            shape: shape.clone(),
                            dtype,
                        },
                    )
                    .map_err(NodeError)?;
            }
        }
        Ok(bindings)
    }
    fn validate(&self) -> Result<(), NodeError> {
        if self.root >= self.nodes.len() || self.bindings.values().any(|id| *id >= self.nodes.len())
        {
            return Err(NodeError("invalid graph root or binding".into()));
        }
        let mut shapes: Vec<Option<Vec<u64>>> = vec![None; self.nodes.len()];
        let mut dtypes: Vec<Option<DType>> = vec![None; self.nodes.len()];
        let mut input_nodes = BTreeMap::new();
        let mut input_names = BTreeMap::new();
        for input in &self.inputs {
            if input.node >= self.nodes.len()
                || self.nodes[input.node] != OpNode::input(&input.name)
                || input_nodes.insert(input.node, input).is_some()
                || input_names.insert(&input.name, input.node).is_some()
            {
                return Err(NodeError("invalid or duplicate graph input".into()));
            }
            shapes[input.node] = input.shape.clone();
            dtypes[input.node] = input.dtype;
        }
        for (index, node) in self.nodes.iter().enumerate() {
            if node.children().iter().any(|id| usize::from(*id) >= index) {
                return Err(NodeError(
                    "graph operands must refer to earlier nodes".into(),
                ));
            }
            // Revalidate mutable Language children and externally prepared definitions.
            OpNode::from_parts(node.op(), node.children().to_vec(), node.attrs().clone())?;
            if node.op() == Op::Input {
                if !input_nodes.contains_key(&index) {
                    return Err(NodeError("undeclared graph input".into()));
                }
                continue;
            }
            if let OpAttrs::Literal { dtype, .. } = node.attrs() {
                shapes[index] = Some(vec![]);
                dtypes[index] = *dtype;
                continue;
            }
            if let Some(operands) = node
                .children()
                .iter()
                .map(|id| shapes[usize::from(*id)].as_deref())
                .collect::<Option<Vec<_>>>()
            {
                match infer_shape(node.op(), &operands, node.attrs()) {
                    Inference::Known(shape) => shapes[index] = Some(shape),
                    Inference::Unknown => {}
                    Inference::Invalid(reason) => {
                        return Err(NodeError(format!("graph node {index}: {reason}")));
                    }
                }
            }
            if let Some(operands) = node
                .children()
                .iter()
                .map(|id| dtypes[usize::from(*id)])
                .collect::<Option<Vec<_>>>()
            {
                match infer_dtype(node.op(), &operands, node.attrs()) {
                    Inference::Known(dtype) => dtypes[index] = Some(dtype),
                    Inference::Unknown => {}
                    Inference::Invalid(reason) => {
                        return Err(NodeError(format!("graph node {index}: {reason}")));
                    }
                }
            }
        }
        Ok(())
    }
    fn insert_nodes<N: Analysis<OpNode>>(
        &self,
        graph: &mut EGraph<OpNode, N>,
    ) -> Result<BuiltGraph, NodeError> {
        self.validate()?;
        let mut ids = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let children = node
                .children()
                .iter()
                .map(|id| ids[usize::from(*id)])
                .collect::<Vec<_>>();
            let node = OpNode::from_parts(node.op(), children, node.attrs().clone())?;
            ids.push(graph.add(node));
        }
        graph.rebuild();
        Ok(BuiltGraph {
            root: graph.find(ids[self.root]),
            bindings: self
                .bindings
                .iter()
                .map(|(name, id)| (name.clone(), graph.find(ids[*id])))
                .collect(),
        })
    }
    /// Insert a structural graph. Typed inputs require a fresh configured analysis.
    pub fn insert_into<N: Analysis<OpNode>>(
        &self,
        graph: &mut EGraph<OpNode, N>,
    ) -> Result<BuiltGraph, NodeError> {
        if self
            .inputs
            .iter()
            .any(|input| input.shape.is_some() || input.dtype.is_some())
        {
            return Err(NodeError(
                "graph input metadata requires into_egraph or into_egraph_with".into(),
            ));
        }
        self.insert_nodes(graph)
    }
    /// Register complete declared input types before constructing any nodes.
    pub fn into_egraph(self) -> Result<(EGraph<OpNode, TensorAnalysis>, BuiltGraph), NodeError> {
        self.into_egraph_with(TensorAnalysis::new)
    }
    /// A custom analysis factory receives all complete input types.
    pub fn into_egraph_with<N: Analysis<OpNode>>(
        self,
        factory: impl FnOnce(TensorBindingTable) -> N,
    ) -> Result<(EGraph<OpNode, N>, BuiltGraph), NodeError> {
        let mut graph = EGraph::new(factory(self.input_bindings()?));
        let built = self.insert_nodes(&mut graph)?;
        Ok((graph, built))
    }
}
