// Generated helpers may be unused.
#[allow(dead_code, unused_imports, unused_variables)]
mod tepl_pattern_basic;
mod utils;

use egg::{AstDepth, EGraph, Extractor, Runner, StopReason};
// Module alias for utils.rs.
use tepl_pattern_basic as tepl;
use tepl_pattern_basic::OpNode;
use tepl_pattern_basic::graphs::add_dot_sample::graph_add_dot_sample;
use tepl_pattern_basic::rules::add_dot::{rule_associate_add, rule_swap_add, rule_swap_dot};
use utils::print_egraph;

struct Host;

impl rule_swap_dot::HostFunctions for Host {
    // Implements TEPL's $supports_axis.
    fn supports_axis(&self, axis: u64) -> Option<bool> {
        Some(axis == 0)
    }
}

fn main() {
    // Build the starting graph.
    let graph_def = graph_add_dot_sample::build().unwrap();

    // Build rules; swap_dot uses the host callback.
    let rules = [
        rule_associate_add::build(()).unwrap(),
        rule_swap_add::build(()).unwrap(),
        rule_swap_dot::build(Host).unwrap(),
    ];

    // No tensor metadata analysis.
    let mut graph = EGraph::<OpNode, ()>::default();
    let built = graph_def.insert_nodes(&mut graph).unwrap();

    // Run rewrites to saturation.
    let runner = Runner::default().with_egraph(graph).run(&rules);

    println!("Stop reason: {:?}", runner.stop_reason);
    assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    print_egraph(&runner.egraph, built.root);

    // Extract a minimum-depth expression.
    let (depth, best) = Extractor::new(&runner.egraph, AstDepth).find_best(built.root);
    println!("\nMinimum AST depth: {depth}");
    println!("Best expression:\n{}", best.pretty(80));
}
