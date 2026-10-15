mod tepl_pattern_basic;
mod utils;
use tepl_pattern_basic as tepl;
use tepl_runtime::egg::{AstDepth, Extractor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let built = tepl::graphs::add_dot_sample::graph_add_dot_sample::build()?;
    let rules = tepl::rules::all()?;
    let result = tepl_runtime::runner::run(tepl_runtime::runner::runner(built.graph)?, &rules)?;
    println!("Stop reason: {:?}", result.stop_reason);
    utils::print_egraph(&result.egraph, built.root);
    let (depth, best) = Extractor::new(&result.egraph, AstDepth).find_best(built.root);
    println!("Minimum AST depth: {depth}\n{}", best.pretty(80));
    Ok(())
}
