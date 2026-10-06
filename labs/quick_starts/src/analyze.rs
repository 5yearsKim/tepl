// Generated helpers include APIs beyond this small demo.
#[allow(dead_code, unused_imports, unused_variables)]
mod tepl_pattern_analyze;
mod utils;

use egg::{AstSize, EGraph, Extractor, Language, RecExpr, Runner, StopReason};
// Shared utilities access this binary's generated types through `crate::tepl`.
use tepl_pattern_analyze as tepl;
use tepl_pattern_analyze::analysis::tensor_info;
use tepl_pattern_analyze::dialects::my_dialect as d;
use tepl_pattern_analyze::graphs::add_dot_sample::graph_add_dot_sample;
use tepl_pattern_analyze::rules::add_dot::{
    rule_associate_add, rule_factor_small_dot, rule_swap_add, rule_swap_dot,
};
use tepl_pattern_analyze::{DType, OpAttrs, OpNode, TensorAnalysis};
use utils::print_egraph;

struct Host;

impl rule_swap_dot::HostFunctions for Host {
    fn supports_axis(&self, axis: u64) -> Option<bool> {
        // Only axis-0 vector dot products support swapping operands.
        Some(axis == 0)
    }
}

// Omit attributes from display: every dot contracts lhs axis 1 with rhs axis 0.
fn expression(expr: &RecExpr<OpNode>) -> String {
    let mut terms: Vec<String> = Vec::new();
    for node in expr.as_ref() {
        terms.push(match node.attrs() {
            OpAttrs::Input { name } => name.clone(),
            _ => format!(
                "{}({})",
                node.op().name(),
                node.children()
                    .iter()
                    .map(|id| terms[usize::from(*id)].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    terms.pop().unwrap()
}

fn main() {
    let graph_def = graph_add_dot_sample::build().unwrap();
    let bindings = graph_def.input_bindings().unwrap();
    let analysis = TensorAnalysis::new(bindings);
    let mut graph = EGraph::new(analysis);
    let built = graph_def.insert_nodes(&mut graph).unwrap();
    for (name, label) in [
        ("x", "x"),
        ("y", "y"),
        ("z", "z"),
        ("a", "dot(x, y)"),
        ("b", "dot(x, z)"),
    ] {
        let info = tensor_info(&graph, built.get(name).unwrap()).unwrap();
        println!("  {label}: {}{:?}", info.dtype, info.shape);
    }
    let before_info = tensor_info(&graph, built.root).unwrap();
    let (before_size, before) = Extractor::new(&graph, AstSize).find_best(built.root);
    let rules = [
        rule_associate_add::build(()).unwrap(),
        rule_swap_add::build(()).unwrap(),
        rule_swap_dot::build(Host).unwrap(),
        rule_factor_small_dot::build(()).unwrap(),
    ];
    let runner = Runner::default().with_egraph(graph).run(&rules);
    assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    println!("Stop reason: {:?}", runner.stop_reason);
    print_egraph(&runner.egraph, built.root);

    let after_info = tensor_info(&runner.egraph, built.root).unwrap();
    let (after_size, after) = Extractor::new(&runner.egraph, AstSize).find_best(built.root);
    let sum = OpNode::new(
        d::Op::Add,
        d::OpAttrs::None,
        vec![built.get("y").unwrap(), built.get("z").unwrap()],
    )
    .unwrap();
    let sum_id = runner.egraph.lookup(sum).unwrap();
    let info = tensor_info(&runner.egraph, sum_id).unwrap();
    println!("  Factored add(y, z): {}{:?}", info.dtype, info.shape);
    println!("Before: {} (AST size {before_size})", expression(&before));
    println!("After:  {} (AST size {after_size})", expression(&after));
    println!("Output: {}{:?}", after_info.dtype, after_info.shape);

    assert_eq!(info.shape, vec![3, 4]);
    assert_eq!(before_info, after_info);
    assert_eq!(after_info.dtype, DType::I32);
    assert_eq!(after_info.shape, vec![2, 4]);
    assert_eq!(before_size, 7);
    assert_eq!(after_size, 5);
}
