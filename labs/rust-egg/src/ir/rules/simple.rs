//! Reference lowering of `examples/rules/simple.tepl`.

use crate::ir::dialects::tensor_lang;
use egg::{Analysis, Rewrite, Var};

use crate::ir::pattern::{AttrExpr, AttrPattern, TensorExpr, TensorPattern, tensor_rewrite};
use crate::ir::{OpAttrs, OpNode};

/// The `commute_add` rule from `simple.tepl`.
pub mod rule_commute_add {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            tensor_lang::Op::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x), TensorPattern::Var(y)],
        )
    }

    /// `(add X Y) => (add Y X)`.
    pub fn build_rewrite<N>() -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
    {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            tensor_lang::Op::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![TensorExpr::Var(y), TensorExpr::Var(x)],
        );
        tensor_rewrite("simple::commute_add", pattern(), rhs, |_, _| {
            Some(Default::default())
        })
    }
}
