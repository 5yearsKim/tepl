mod tepl_pattern_analyze;
mod utils;
use tepl_pattern_analyze as tepl;
use tepl_runtime::egg::{AstSize, Extractor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let built = tepl::graphs::add_dot_sample::graph_add_dot_sample::build()?;
    let before = built.graph[built.root].data.clone();
    let (before_size, _) = Extractor::new(&built.graph, AstSize).find_best(built.root);
    let rules = tepl::rules::all()?;
    let result = tepl_runtime::runner::run(tepl_runtime::runner::runner(built.graph)?, &rules)?;
    let (after_size, best) = Extractor::new(&result.egraph, AstSize).find_best(built.root);
    assert_eq!(before, result.egraph[built.root].data);
    assert!(after_size < before_size);
    println!("AST size: {before_size} -> {after_size}");
    utils::print_egraph(&result.egraph, built.root);
    println!("Best expression: {}", best.pretty(80));
    Ok(())
}
