//! Reference lowering of `examples/rules/lora.tepl`.

use crate::ir::dialects::tensor_lang;
use std::collections::HashMap;

use egg::{Analysis, EGraph, Rewrite, Var};

use crate::ir::pattern::{
    AttrExpr, AttrPattern, AttrVar, MatchContext, OutputInference, TensorExpr, TensorInfo,
    TensorMetadata, TensorPattern, tensor_rewrite, tensor_rewrite_checked,
};
use crate::ir::{OpAttrs, OpNode};

pub mod rule_lora {
    use super::*;

    /// Functions required to construct and validate this rewrite.
    pub trait Functions: Send + Sync {
        fn broadcastable(&self, batch: &[usize], weight_batch: &[usize]) -> Option<bool>;
        fn reassociable(
            &self,
            x: &TensorInfo,
            a: &TensorInfo,
            b: &TensorInfo,
            outer: &OpAttrs,
            inner: &OpAttrs,
        ) -> Option<bool>;
        fn infer_dot(&self, lhs: &TensorInfo, rhs: &TensorInfo, attrs: &OpAttrs)
        -> Option<OpAttrs>;
        fn infer_lora_out(
            &self,
            x: &TensorInfo,
            a: &TensorInfo,
            b: &TensorInfo,
            outer: &OpAttrs,
            inner: &OpAttrs,
        ) -> Option<OpAttrs>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, w, a, b] = ["?X", "?W", "?A", "?B"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            tensor_lang::Op::DotGeneral,
            AttrPattern::Bind(AttrVar::from("outer")),
            vec![
                TensorPattern::Var(x),
                TensorPattern::op(
                    tensor_lang::Op::Add,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![
                        TensorPattern::Var(w),
                        TensorPattern::op(
                            tensor_lang::Op::DotGeneral,
                            AttrPattern::Bind(AttrVar::from("inner")),
                            vec![TensorPattern::Var(a), TensorPattern::Var(b)],
                        ),
                    ],
                ),
            ],
        )
    }

    pub fn build_rewrite<N, M, F>(metadata: M, functions: F) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        F: Functions + 'static,
    {
        let [x, w, a, b] = ["?X", "?W", "?A", "?B"].map(|name| name.parse::<Var>().unwrap());
        let [outer, inner, xw, xa, out] = ["outer", "inner", "xw", "xa", "out"].map(AttrVar::from);

        let rhs = expression();
        tensor_rewrite("lora::lora", pattern(), rhs, move |egraph, matched| {
            let ctx = MatchContext::new(egraph, matched, &metadata);
            check_and_derive(&ctx, &functions, [x, w, a, b], [outer, inner, xw, xa, out])
        })
    }

    /// Checked lowering: verifies every constructed operation and the output
    /// type before insertion, in addition to the rule's legality predicates.
    pub fn build_checked_rewrite<N, M, I, F>(
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
        let metadata = std::sync::Arc::new(metadata);
        let checker_metadata = metadata.clone();
        let inputs = ["?X", "?W", "?A", "?B"].map(|name| name.parse::<Var>().unwrap());
        let attrs = ["outer", "inner", "xw", "xa", "out"].map(AttrVar::from);
        tensor_rewrite_checked(
            "lora::lora",
            pattern(),
            expression(),
            move |graph: &EGraph<OpNode, N>, id| metadata.info(graph, id),
            inference,
            move |graph, matched| {
                let ctx = MatchContext::new(graph, matched, checker_metadata.as_ref());
                check_and_derive(&ctx, &functions, inputs, attrs)
            },
        )
    }

    fn expression() -> TensorExpr {
        let [x, w, a, b] = ["?X", "?W", "?A", "?B"].map(|name| name.parse::<Var>().unwrap());
        let [xw, xa, out] = ["xw", "xa", "out"].map(AttrVar::from);
        TensorExpr::op(
            tensor_lang::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    tensor_lang::Op::DotGeneral,
                    AttrExpr::Derived(xw),
                    vec![TensorExpr::Var(x), TensorExpr::Var(w)],
                ),
                TensorExpr::op(
                    tensor_lang::Op::DotGeneral,
                    AttrExpr::Derived(out),
                    vec![
                        TensorExpr::op(
                            tensor_lang::Op::DotGeneral,
                            AttrExpr::Derived(xa),
                            vec![TensorExpr::Var(x), TensorExpr::Var(a)],
                        ),
                        TensorExpr::Var(b),
                    ],
                ),
            ],
        )
    }

    fn matrix_shape(shape: &[usize]) -> Option<(&[usize], usize, usize)> {
        let rank = shape.len().checked_sub(2)?;
        Some((&shape[..rank], shape[rank], shape[rank + 1]))
    }

    fn check_and_derive<N, M, F>(
        ctx: &MatchContext<'_, N, M>,
        functions: &F,
        inputs: [Var; 4],
        attrs: [AttrVar; 5],
    ) -> Option<HashMap<AttrVar, OpAttrs>>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N>,
        F: Functions,
    {
        let [x, w, a, b] = inputs;
        let [outer, inner, xw_attr, xa_attr, out_attr] = attrs;
        let (x, w, a, b) = (
            ctx.tensor(x)?,
            ctx.tensor(w)?,
            ctx.tensor(a)?,
            ctx.tensor(b)?,
        );
        let (outer, inner) = (ctx.attrs(outer)?, ctx.attrs(inner)?);

        // Check the dimensions shared by the four TEPL declarations. Each batch
        // prefix can contain any number of axes.
        let (batch, _, k) = matrix_shape(&x.shape)?;
        let (weight_batch, wk, n) = matrix_shape(&w.shape)?;
        let (a_batch, ak, r) = matrix_shape(&a.shape)?;
        let (b_batch, br, bn) = matrix_shape(&b.shape)?;
        if weight_batch != a_batch
            || weight_batch != b_batch
            || k != wk
            || k != ak
            || r != br
            || n != bn
        {
            return None;
        }

        if !functions.broadcastable(batch, weight_batch)?
            || !functions.reassociable(&x, &a, &b, outer, inner)?
        {
            return None;
        }
        // Derivations read only matched LHS metadata and captured descriptors.
        let xw = functions.infer_dot(&x, &w, outer)?;
        let xa = functions.infer_dot(&x, &a, outer)?;
        let out = functions.infer_lora_out(&x, &a, &b, outer, inner)?;
        Some(HashMap::from([
            (xw_attr, xw),
            (xa_attr, xa),
            (out_attr, out),
        ]))
    }
}
