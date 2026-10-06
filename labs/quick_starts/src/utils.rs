use crate::tepl::{OpAttrs, OpNode, TensorAnalysisData, dialects::my_dialect as d};
use egg::{Analysis, EGraph, Id, Language};

pub trait ClassMetadata {
    fn summary(&self) -> Option<String>;
}

impl ClassMetadata for () {
    fn summary(&self) -> Option<String> {
        None
    }
}

impl ClassMetadata for TensorAnalysisData {
    fn summary(&self) -> Option<String> {
        Some(match self.info() {
            Some(info) => format!("{}{:?}", info.dtype, info.shape),
            None if self.is_invalid() => "invalid metadata".into(),
            None => "unknown metadata".into(),
        })
    }
}

pub fn print_egraph<N: Analysis<OpNode>>(graph: &EGraph<OpNode, N>, root: Id)
where
    N::Data: ClassMetadata,
{
    let mut classes: Vec<_> = graph.classes().collect();
    classes.sort_by_key(|class| class.id);
    println!("\nSaturated e-graph (nodes in each e-class are equivalent):");
    for class in classes {
        let metadata = class
            .data
            .summary()
            .map(|value| format!(": {value}"))
            .unwrap_or_default();
        let marker = if class.id == graph.find(root) {
            " (root)"
        } else {
            ""
        };
        println!("e{}{metadata}{marker}", class.id);
        let mut nodes: Vec<_> = class
            .nodes
            .iter()
            .map(|node| {
                if let OpAttrs::Input { name } = node.attrs() {
                    return format!("input {name}");
                }
                if let OpAttrs::Literal { value, dtype } = node.attrs() {
                    return match dtype {
                        Some(dtype) => format!("{value}:{dtype}"),
                        None => value.clone(),
                    };
                }
                let attrs = match node.attrs() {
                    OpAttrs::MyDialect(d::OpAttrs::DotAttrs { axis }) => format!("[axis={axis}]"),
                    _ => String::new(),
                };
                let children = node
                    .children()
                    .iter()
                    .map(|id| format!("e{}", graph.find(*id)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}{attrs}({children})", node.op().name())
            })
            .collect();
        nodes.sort();
        for node in nodes {
            println!("  {node}");
        }
    }
    println!();
}
