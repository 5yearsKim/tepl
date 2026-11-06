use crate::{EvalResult, Node, Program, TensorAnalysis};
use egg::{EGraph, Rewrite, Runner};

pub type CheckedRunner<P> = Runner<Node<P>, TensorAnalysis<P>>;
/// Installs error propagation for egg callbacks, which do not return Results.
pub fn runner<P: Program>(
    graph: EGraph<Node<P>, TensorAnalysis<P>>,
) -> EvalResult<CheckedRunner<P>> {
    graph.analysis.check()?;
    Ok(Runner::new(TensorAnalysis::default())
        .with_egraph(graph)
        .with_hook(|runner| runner.egraph.analysis.check().map_err(|e| e.to_string())))
}
pub fn run<P: Program>(
    runner: CheckedRunner<P>,
    rules: &[Rewrite<Node<P>, TensorAnalysis<P>>],
) -> EvalResult<CheckedRunner<P>> {
    runner.egraph.analysis.check()?;
    let result = runner.run(rules);
    result.egraph.analysis.check()?;
    Ok(result)
}
