//! Reference lowering of `examples/rules/binders.tepl`.

use crate::ir::dialects::tensor_lang;
use egg::{Analysis, Rewrite, Var};

use crate::ir::pattern::{
    AttrExpr, AttrPattern, AttrVar, MatchContext, TensorExpr, TensorInfo, TensorMetadata,
    TensorPattern, tensor_rewrite,
};
use crate::ir::{OpAttrs, OpNode};

/// The `shared_expression` rule from `binders.tepl`.
pub mod rule_shared_expression {
    use super::*;

    /// Functions required to validate this rewrite.
    pub trait Functions: Send + Sync {
        fn reusable(&self, tensor: &TensorInfo, attrs: &OpAttrs) -> Option<bool>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, w, z, y] = ["?X", "?W", "?Z", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            tensor_lang::Op::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::bind(
                    y,
                    TensorPattern::op(
                        tensor_lang::Op::DotGeneral,
                        AttrPattern::Bind(AttrVar::from("d")),
                        vec![TensorPattern::Var(x), TensorPattern::Var(w)],
                    ),
                ),
                TensorPattern::op(
                    tensor_lang::Op::Multiply,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![TensorPattern::Var(y), TensorPattern::Var(z)],
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
        let [y, z] = ["?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        let d = AttrVar::from("d");
        let rhs = TensorExpr::op(
            tensor_lang::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    tensor_lang::Op::Multiply,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(y), TensorExpr::Var(z)],
                ),
                TensorExpr::Var(y),
            ],
        );
        tensor_rewrite(
            "binders::shared_expression",
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
pub mod rule_root_binding {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let x = "?X".parse::<Var>().unwrap();
        let y = "?Y".parse::<Var>().unwrap();
        TensorPattern::bind(
            y,
            TensorPattern::op(
                tensor_lang::Op::Transpose,
                AttrPattern::Bind(AttrVar::from("t")),
                vec![TensorPattern::Var(x)],
            ),
        )
    }

    /// `let Y = (transpose[@t] X) => Y` is an identity rewrite.
    pub fn build_rewrite<N>() -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
    {
        let y = "?Y".parse::<Var>().unwrap();
        tensor_rewrite(
            "binders::root_binding",
            pattern(),
            TensorExpr::Var(y),
            |_, _| Some(Default::default()),
        )
    }
}
