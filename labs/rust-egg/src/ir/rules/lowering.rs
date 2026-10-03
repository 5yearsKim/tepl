// Reference generated rules: default TensorAnalysis API plus explicit runtime hooks.
// Bodies originate from TEPL; the compiler still emits the callback-only API.
#![allow(unused_imports, unused_variables, unused_mut, unused_parens)]
use super::super::analysis::{TensorAnalysis, infer_tensor_output, tensor_info};
use super::super::dialects::*;
use super::super::pattern::*;
use super::super::{DType, Op, OpAttrs, OpNode};
use egg::{Analysis, EGraph, Rewrite, Var};
use std::{collections::HashMap, sync::Arc};

pub mod rule_scalar_add {
    use super::*;
    pub trait Functions: Send + Sync {}
    impl Functions for () {}
    pub fn pattern() -> TensorPattern {
        TensorPattern::op(
            tensor_lang::Op::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::Var("?c0".parse::<Var>().expect("generated capture ID")),
                TensorPattern::Var("?c1".parse::<Var>().expect("generated capture ID")),
            ],
        )
    }
    pub fn expression() -> TensorExpr {
        TensorExpr::op(
            scalar::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::Var("?c0".parse::<Var>().expect("generated capture ID")),
                TensorExpr::Var("?c1".parse::<Var>().expect("generated capture ID")),
            ],
        )
    }
    /// Default builder uses the same inference and metadata as TensorAnalysis.
    pub fn build_rewrite<F: Functions + 'static>(
        functions: F,
    ) -> Result<Rewrite<OpNode, TensorAnalysis>, String> {
        build_rewrite_with(tensor_info, infer_tensor_output, functions)
    }

    /// Explicit hooks for custom analyses and runtime validation tests.
    pub fn build_rewrite_with<N, M, I, F>(
        metadata: M,
        inference: I,
        functions: F,
    ) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
        F: Functions + 'static,
    {
        let metadata = Arc::new(metadata);
        let checker_metadata = metadata.clone();
        tensor_rewrite_checked(
            "lowering::scalar_add",
            pattern(),
            expression(),
            move |graph: &EGraph<OpNode, N>, id| metadata.info(graph, id),
            inference,
            move |graph, matched| {
                let ctx = MatchContext::new(graph, matched, checker_metadata.as_ref());
                let capture_0 = ctx.tensor("?c0".parse::<Var>().expect("generated capture ID"))?;
                let capture_1 = ctx.tensor("?c1".parse::<Var>().expect("generated capture ID"))?;
                let mut dimensions = ShapeBindings::default();
                dimensions.check(&capture_0.shape, &[])?;
                dimensions.check(&capture_1.shape, &[])?;
                Some(Default::default())
            },
        )
    }
}
