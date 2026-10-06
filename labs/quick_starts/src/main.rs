// The generated module contains runtime helpers that this small demo doesn't use.
#[allow(dead_code, unused_imports, unused_variables)]
mod generated;

use egg::{AstDepth, EGraph, Extractor, Runner, StopReason};
use generated::OpNode;
use generated::graphs::example::graph_example;
use generated::rules::your_rule::{rule_associate_add, rule_swap_add, rule_swap_dot};

struct Host;

impl rule_swap_dot::HostFunctions for Host {
    // TEPL's $supports_axis(...) calls this Rust method.
    fn supports_axis(&self, axis: u64) -> Option<bool> {
        Some(axis == 0)
    }
}

fn main() {
    let graph_def = graph_example::build().unwrap();
    let rules = [
        rule_associate_add::build(()).unwrap(),
        rule_swap_add::build(()).unwrap(),
        rule_swap_dot::build(Host).unwrap(),
    ];

    let mut graph = EGraph::<OpNode, ()>::default();
    let built = graph_def.insert_into(&mut graph).unwrap();
    let runner = Runner::default().with_egraph(graph).run(&rules);

    println!("Stop reason: {:?}", runner.stop_reason);
    assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    println!("\nSaturated e-graph:\n{:?}", runner.egraph.dump());

    let (depth, best) = Extractor::new(&runner.egraph, AstDepth).find_best(built.root);
    println!("\nMinimum AST depth: {depth}");
    println!("Best expression:\n{}", best.pretty(80));
}
