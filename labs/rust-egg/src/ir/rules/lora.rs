//! Reference lowering of `examples/lora.tepl`.

use std::collections::HashMap;

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{
    AttrExpr, AttrPattern, AttrVar, MatchContext, TensorExpr, TensorInfo, TensorMetadata,
    TensorPattern, tensor_rewrite,
};
use crate::ir::{OpAttrs, OpKind, TensorLang};

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
            OpKind::DotGeneral,
            AttrPattern::Bind(AttrVar::from("outer")),
            vec![
                TensorPattern::Var(x),
                TensorPattern::op(
                    OpKind::Add,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![
                        TensorPattern::Var(w),
                        TensorPattern::op(
                            OpKind::DotGeneral,
                            AttrPattern::Bind(AttrVar::from("inner")),
                            vec![TensorPattern::Var(a), TensorPattern::Var(b)],
                        ),
                    ],
                ),
            ],
        )
    }

    pub fn build_rewrite<N, M, F>(
        metadata: M,
        functions: F,
    ) -> Result<Rewrite<TensorLang, N>, String>
    where
        N: Analysis<TensorLang>,
        M: TensorMetadata<N> + 'static,
        F: Functions + 'static,
    {
        let [x, w, a, b] = ["?X", "?W", "?A", "?B"].map(|name| name.parse::<Var>().unwrap());
        let [outer, inner, xw, xa, out] = ["outer", "inner", "xw", "xa", "out"].map(AttrVar::from);

        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    OpKind::DotGeneral,
                    AttrExpr::Derived(xw),
                    vec![TensorExpr::Var(x), TensorExpr::Var(w)],
                ),
                TensorExpr::op(
                    OpKind::DotGeneral,
                    AttrExpr::Derived(out),
                    vec![
                        TensorExpr::op(
                            OpKind::DotGeneral,
                            AttrExpr::Derived(xa),
                            vec![TensorExpr::Var(x), TensorExpr::Var(a)],
                        ),
                        TensorExpr::Var(b),
                    ],
                ),
            ],
        );
        tensor_rewrite("lora", pattern(), rhs, move |egraph, matched| {
            let ctx = MatchContext::new(egraph, matched, &metadata);
            check_and_derive(&ctx, &functions, [x, w, a, b], [outer, inner, xw, xa, out])
        })
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
        N: Analysis<TensorLang>,
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
