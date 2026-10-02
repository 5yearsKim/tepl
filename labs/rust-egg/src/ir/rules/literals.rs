//! Reference lowering of `examples/literals.tepl`.

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{AttrExpr, AttrPattern, TensorExpr, TensorPattern, tensor_rewrite};
use crate::ir::{OpAttrs, OpKind, TensorLang};

fn pattern(value: &str) -> TensorPattern {
    let x = "?X".parse::<Var>().unwrap();
    TensorPattern::op(
        OpKind::Add,
        AttrPattern::Exact(OpAttrs::None),
        vec![
            TensorPattern::Var(x),
            TensorPattern::literal(value).unwrap(),
        ],
    )
}

fn build_rewrite<N: Analysis<TensorLang>>(
    name: &str,
    value: &str,
) -> Result<Rewrite<TensorLang, N>, String> {
    let x = "?X".parse::<Var>().unwrap();
    let rhs = TensorExpr::op(
        OpKind::Add,
        AttrExpr::Exact(OpAttrs::None),
        vec![TensorExpr::literal(value).unwrap(), TensorExpr::Var(x)],
    );
    tensor_rewrite(name, pattern(value), rhs, |_, _| Some(Default::default()))
}

pub mod rule_commute_integer_literal {
    use super::*;

    pub fn pattern() -> TensorPattern {
        super::pattern("1")
    }

    pub fn build_rewrite<N: Analysis<TensorLang>>() -> Result<Rewrite<TensorLang, N>, String> {
        super::build_rewrite("commute_integer_literal", "1")
    }
}

pub mod rule_commute_float_literal {
    use super::*;

    pub fn pattern() -> TensorPattern {
        super::pattern("1.0")
    }

    pub fn build_rewrite<N: Analysis<TensorLang>>() -> Result<Rewrite<TensorLang, N>, String> {
        super::build_rewrite("commute_float_literal", "1.0")
    }
}

pub mod rule_commute_negative_literal {
    use super::*;

    pub fn pattern() -> TensorPattern {
        super::pattern("-0.5")
    }

    pub fn build_rewrite<N: Analysis<TensorLang>>() -> Result<Rewrite<TensorLang, N>, String> {
        super::build_rewrite("commute_negative_literal", "-0.5")
    }
}
