use tepl_runtime::egg::{EGraph, Id};
use tepl_runtime::{Node, Program, TensorAnalysis};
pub fn print_egraph<P: Program>(graph: &EGraph<Node<P>, TensorAnalysis<P>>, root: Id) {
    let mut classes = graph.classes().collect::<Vec<_>>();
    classes.sort_by_key(|c| c.id);
    for class in classes {
        let marker = if class.id == graph.find(root) {
            " (root)"
        } else {
            ""
        };
        println!("e{}: {:?}{marker}", class.id, class.data);
        for node in &class.nodes {
            println!("  {node} {:?}", node.operands());
        }
    }
}
