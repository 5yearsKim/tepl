//! Reference lowering of `examples/simple.tepl`.

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{AttrExpr, AttrPattern, TensorExpr, TensorPattern, tensor_rewrite};
use crate::ir::{OpAttrs, OpKind, TensorLang};

/// `(add X Y) => (add Y X)`.
pub fn rule_commute_add<N: Analysis<TensorLang>>() -> Result<Rewrite<TensorLang, N>, String> {
    let x = "?X".parse::<Var>().unwrap();
    let y = "?Y".parse::<Var>().unwrap();
    let lhs = TensorPattern::op(
        OpKind::Add,
        AttrPattern::Exact(OpAttrs::None),
        vec![TensorPattern::Var(x), TensorPattern::Var(y)],
    );
    let rhs = TensorExpr::op(
        OpKind::Add,
        AttrExpr::Exact(OpAttrs::None),
        vec![TensorExpr::Var(y), TensorExpr::Var(x)],
    );
    tensor_rewrite("commute_add", lhs, rhs, |_, _| Some(Default::default()))
}
