//! Reference lowering of `examples/rules/basic.tepl`; all rules verify RHS types.

use crate::ir::dialects::tensor_lang;
use std::sync::Arc;

use egg::{Analysis, EGraph, Rewrite, Var};

use crate::ir::pattern::{
    AttrExpr, AttrPattern, MatchContext, OutputInference, TensorExpr, TensorInfo, TensorMetadata,
    TensorPattern, tensor_rewrite_checked,
};
use crate::ir::{DType, OpAttrs, OpNode};

fn swap<N, M, I, F>(
    name: &str,
    metadata: M,
    inference: I,
    check: F,
) -> Result<Rewrite<OpNode, N>, String>
where
    N: Analysis<OpNode>,
    M: TensorMetadata<N> + 'static,
    I: OutputInference + 'static,
    F: Fn(&TensorInfo, &TensorInfo) -> Option<bool> + Send + Sync + 'static,
{
    let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
    let metadata = Arc::new(metadata);
    let checker_metadata = metadata.clone();
    tensor_rewrite_checked(
        name,
        TensorPattern::op(
            tensor_lang::Op::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x), TensorPattern::Var(y)],
        ),
        TensorExpr::op(
            tensor_lang::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![TensorExpr::Var(y), TensorExpr::Var(x)],
        ),
        move |graph: &EGraph<OpNode, N>, id| metadata.info(graph, id),
        inference,
        move |graph, matched| {
            let ctx = MatchContext::new(graph, matched, checker_metadata.as_ref());
            check(&ctx.tensor(x)?, &ctx.tensor(y)?)?.then_some(Default::default())
        },
    )
}

pub mod rule_commute_f32 {
    use super::*;

    pub fn build_rewrite<N, M, I>(metadata: M, inference: I) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
    {
        swap("basic::commute_f32", metadata, inference, |x, y| {
            Some(
                x.dtype == DType::F32
                    && y.dtype == DType::F32
                    && x.shape.len() == 1
                    && x.shape == y.shape,
            )
        })
    }
}

pub mod rule_commute_same_dtype {
    use super::*;

    pub trait Functions: Send + Sync {
        fn same_dtype(&self, x: &TensorInfo, y: &TensorInfo) -> Option<bool>;
    }

    pub fn build_rewrite<N, M, I, F>(
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
        swap(
            "basic::commute_same_dtype",
            metadata,
            inference,
            move |x, y| {
                if x.shape.len() != 1 || x.shape != y.shape {
                    return None;
                }
                functions.same_dtype(x, y)
            },
        )
    }
}

pub mod rule_commute_scalar {
    use super::*;

    pub fn build_rewrite<N, M, I>(metadata: M, inference: I) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
    {
        swap("basic::commute_scalar", metadata, inference, |x, s| {
            Some(x.dtype == DType::F32 && s.dtype == DType::F32 && s.shape.is_empty())
        })
    }
}

fn literal_pattern(value: &str, dtype: DType) -> TensorPattern {
    let x = "?X".parse::<Var>().unwrap();
    TensorPattern::op(
        tensor_lang::Op::Add,
        AttrPattern::Exact(OpAttrs::None),
        vec![
            TensorPattern::Var(x),
            TensorPattern::literal(value, dtype).unwrap(),
        ],
    )
}

fn literal_rewrite<N, M, I>(
    name: &str,
    value: &str,
    dtype: DType,
    metadata: M,
    inference: I,
) -> Result<Rewrite<OpNode, N>, String>
where
    N: Analysis<OpNode>,
    M: TensorMetadata<N> + 'static,
    I: OutputInference + 'static,
{
    let x = "?X".parse::<Var>().unwrap();
    let metadata = Arc::new(metadata);
    let checker_metadata = metadata.clone();
    tensor_rewrite_checked(
        name,
        literal_pattern(value, dtype),
        TensorExpr::op(
            tensor_lang::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::literal(value, dtype).map_err(|error| error.to_string())?,
                TensorExpr::Var(x),
            ],
        ),
        move |graph: &EGraph<OpNode, N>, id| metadata.info(graph, id),
        inference,
        move |graph, matched| {
            let ctx = MatchContext::new(graph, matched, checker_metadata.as_ref());
            (ctx.tensor(x)?.dtype == dtype).then_some(Default::default())
        },
    )
}

pub mod rule_commute_integer_literal {
    use super::*;

    pub fn pattern() -> TensorPattern {
        literal_pattern("1", DType::I32)
    }

    pub fn build_rewrite<N, M, I>(metadata: M, inference: I) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
    {
        literal_rewrite(
            "basic::commute_integer_literal",
            "1",
            DType::I32,
            metadata,
            inference,
        )
    }
}

pub mod rule_commute_float_literal {
    use super::*;

    pub fn pattern() -> TensorPattern {
        literal_pattern("1.0", DType::F32)
    }

    pub fn build_rewrite<N, M, I>(metadata: M, inference: I) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
    {
        literal_rewrite(
            "basic::commute_float_literal",
            "1.0",
            DType::F32,
            metadata,
            inference,
        )
    }
}
