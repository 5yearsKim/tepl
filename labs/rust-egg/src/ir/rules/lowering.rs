//! Reference lowering of examples/rules/lowering.tepl.
use crate::ir::dialects::{scalar as s, tensor_lang as t};
use crate::ir::pattern::*;
use crate::ir::{OpAttrs, OpNode};
use egg::{Analysis, EGraph, Rewrite, Var};
use std::sync::Arc;

pub mod rule_scalar_add {
    use super::*;
    pub fn build_rewrite<N, M, I>(
        metadata: M,
        inference: I,
        _: (),
    ) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
    {
        let x: Var = "?x".parse().unwrap();
        let y: Var = "?y".parse().unwrap();
        let metadata = Arc::new(metadata);
        let checker = metadata.clone();
        tensor_rewrite_checked(
            "lowering::scalar_add",
            TensorPattern::op(
                t::Op::Add,
                AttrPattern::Exact(OpAttrs::None),
                vec![TensorPattern::Var(x), TensorPattern::Var(y)],
            ),
            TensorExpr::op(
                s::Op::Add,
                AttrExpr::Exact(OpAttrs::None),
                vec![TensorExpr::Var(x), TensorExpr::Var(y)],
            ),
            move |graph: &EGraph<OpNode, N>, id| metadata.info(graph, id),
            inference,
            move |graph, matched| {
                let ctx = MatchContext::new(graph, matched, checker.as_ref());
                if !ctx.tensor(x)?.shape.is_empty() || !ctx.tensor(y)?.shape.is_empty() {
                    return None;
                }
                Some(Default::default())
            },
        )
    }
}
