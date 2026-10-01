//! Reference lowering of `examples/binders.tepl`.

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{
    AttrExpr, AttrPattern, AttrVar, MatchContext, TensorExpr, TensorInfo, TensorMetadata,
    TensorPattern, tensor_rewrite,
};
use crate::ir::{OpAttrs, OpKind, TensorLang};

/// Host functions called by the rules in `binders.tepl`.
pub trait HostFunctions: Send + Sync {
    fn reusable(&self, tensor: &TensorInfo, attrs: &OpAttrs) -> Option<bool>;
}

pub fn shared_expression_pattern() -> TensorPattern {
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

pub fn rule_shared_expression<N, M, H>(
    metadata: M,
    host: H,
) -> Result<Rewrite<TensorLang, N>, String>
where
    N: Analysis<TensorLang>,
    M: TensorMetadata<N> + 'static,
    H: HostFunctions + 'static,
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
        shared_expression_pattern(),
        rhs,
        move |egraph, matched| {
            let ctx = MatchContext::new(egraph, matched, &metadata);
            host.reusable(&ctx.tensor(y)?, ctx.attrs(d)?)?
                .then_some(Default::default())
        },
    )
}

pub fn root_binding_pattern() -> TensorPattern {
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
pub fn rule_root_binding<N: Analysis<TensorLang>>() -> Result<Rewrite<TensorLang, N>, String> {
    let y = "?Y".parse::<Var>().unwrap();
    tensor_rewrite(
        "root_binding",
        root_binding_pattern(),
        TensorExpr::Var(y),
        |_, _| Some(Default::default()),
    )
}
