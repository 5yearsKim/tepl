// The generated module contains runtime helpers that this small demo doesn't use.
#[allow(dead_code, unused_imports, unused_variables)]
mod tepl_pattern_basic;
mod utils;

use egg::{AstDepth, EGraph, Extractor, Runner, StopReason};
// Shared utilities access this binary's generated types through `crate::tepl`.
use tepl_pattern_basic as tepl;
use tepl_pattern_basic::OpNode;
use tepl_pattern_basic::graphs::add_dot_sample::graph_add_dot_sample;
use tepl_pattern_basic::rules::add_dot::{rule_associate_add, rule_swap_add, rule_swap_dot};
use utils::print_egraph;

struct Host;

impl rule_swap_dot::HostFunctions for Host {
    // TEPL's $supports_axis(...) calls this Rust method.
    fn supports_axis(&self, axis: u64) -> Option<bool> {
        Some(axis == 0)
    }
}

fn main() {
    let graph_def = graph_add_dot_sample::build().unwrap();
    let rules = [
        rule_associate_add::build(()).unwrap(),
        rule_swap_add::build(()).unwrap(),
        rule_swap_dot::build(Host).unwrap(),
    ];

    let mut graph = EGraph::<OpNode, ()>::default();
    let built = graph_def.insert_nodes(&mut graph).unwrap();
    let runner = Runner::default().with_egraph(graph).run(&rules);

    println!("Stop reason: {:?}", runner.stop_reason);
    assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    print_egraph(&runner.egraph, built.root);

    let (depth, best) = Extractor::new(&runner.egraph, AstDepth).find_best(built.root);
    println!("\nMinimum AST depth: {depth}");
    println!("Best expression:\n{}", best.pretty(80));
}
