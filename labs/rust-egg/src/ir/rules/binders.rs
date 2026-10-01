//! Reference lowering of `examples/binders.tepl`.

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{
    AttrExpr, AttrPattern, AttrVar, MatchContext, TensorExpr, TensorInfo, TensorMetadata,
    TensorPattern, tensor_rewrite,
};
use crate::ir::{OpAttrs, OpKind, TensorLang};

/// The `shared_expression` rule from `binders.tepl`.
pub mod shared_expression {
    use super::*;

    /// Functions required to validate this rewrite.
    pub trait Functions: Send + Sync {
        fn reusable(&self, tensor: &TensorInfo, attrs: &OpAttrs) -> Option<bool>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, w, z, y] = ["?X", "?W", "?Z", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::bind(
                    y,
                    TensorPattern::op(
                        OpKind::DotGeneral,
                        AttrPattern::Bind(AttrVar::from("d")),
                        vec![TensorPattern::Var(x), TensorPattern::Var(w)],
                    ),
                ),
                TensorPattern::op(
                    OpKind::Multiply,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![TensorPattern::Var(y), TensorPattern::Var(z)],
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
        let [y, z] = ["?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        let d = AttrVar::from("d");
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    OpKind::Multiply,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(y), TensorExpr::Var(z)],
                ),
                TensorExpr::Var(y),
            ],
        );
        tensor_rewrite(
            "shared_expression",
            pattern(),
            rhs,
            move |egraph, matched| {
                let ctx = MatchContext::new(egraph, matched, &metadata);
                functions
                    .reusable(&ctx.tensor(y)?, ctx.attrs(d)?)?
                    .then_some(Default::default())
            },
        )
    }
}

/// The `root_binding` rule from `binders.tepl`.
pub mod root_binding {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let x = "?X".parse::<Var>().unwrap();
        let y = "?Y".parse::<Var>().unwrap();
        TensorPattern::bind(
            y,
            TensorPattern::op(
                OpKind::Transpose,
                AttrPattern::Bind(AttrVar::from("t")),
                vec![TensorPattern::Var(x)],
            ),
        )
    }

    /// `let Y = (transpose[@t] X) => Y` is an identity rewrite.
    pub fn build_rewrite<N>() -> Result<Rewrite<TensorLang, N>, String>
    where
        N: Analysis<TensorLang>,
    {
        let y = "?Y".parse::<Var>().unwrap();
        tensor_rewrite("root_binding", pattern(), TensorExpr::Var(y), |_, _| {
            Some(Default::default())
        })
    }
}
