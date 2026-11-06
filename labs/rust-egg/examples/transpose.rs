use rust_egg::ir;
use tepl_runtime::egg::{AstSize, Extractor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let built = ir::graphs::transpose::graph_example::build()?;
    let rules = ir::rules::all()?;
    let runner = tepl_runtime::runner::runner(built.graph)?.with_iter_limit(20);
    let result = tepl_runtime::runner::run(runner, &rules)?;
    let (cost, expression) = Extractor::new(&result.egraph, AstSize).find_best(built.root);
    println!("Best expression ({cost} nodes): {}", expression.pretty(80));
    println!("Output metadata: {:?}", result.egraph[built.root].data);
    for rejected in result.egraph.analysis.rejections() {
        eprintln!(
            "{}: rule {}: {}",
            rejected.source, rejected.rule, rejected.error
        );
    }
    Ok(())
}
