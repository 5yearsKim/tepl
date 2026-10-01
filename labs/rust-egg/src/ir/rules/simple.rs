//! Reference lowering of `examples/simple.tepl`.

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{AttrExpr, AttrPattern, TensorExpr, TensorPattern, tensor_rewrite};
use crate::ir::{OpAttrs, OpKind, TensorLang};

/// The `commute_add` rule from `simple.tepl`.
pub mod commute_add {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x), TensorPattern::Var(y)],
        )
    }

    /// `(add X Y) => (add Y X)`.
    pub fn build_rewrite<N>() -> Result<Rewrite<TensorLang, N>, String>
    where
        N: Analysis<TensorLang>,
    {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![TensorExpr::Var(y), TensorExpr::Var(x)],
        );
        tensor_rewrite("commute_add", pattern(), rhs, |_, _| {
            Some(Default::default())
        })
    }
}
