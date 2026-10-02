//! Reference lowering of examples/rules/scalar.tepl.
use crate::ir::dialects::scalar;
use crate::ir::pattern::*;
use crate::ir::{OpAttrs, OpNode};
use egg::{Analysis, Rewrite, Var};

pub mod rule_commute_add {
    use super::*;
    pub fn build_rewrite<N, M, I>(
        metadata: M,
        inference: I,
        _: (),
    ) -> Result<Rewrite<OpNode, N>, String>
    where
        N: Analysis<OpNode>,
        M: TensorMetadata<N> + 'static,
        I: OutputInference + 'static,
    {
        let x: Var = "?x".parse().unwrap();
        let y: Var = "?y".parse().unwrap();
        tensor_rewrite_checked(
            "scalar::commute_add",
            TensorPattern::op(
                scalar::Op::Add,
                AttrPattern::Exact(OpAttrs::None),
                vec![TensorPattern::Var(x), TensorPattern::Var(y)],
            ),
            TensorExpr::op(
                scalar::Op::Add,
                AttrExpr::Exact(OpAttrs::None),
                vec![TensorExpr::Var(y), TensorExpr::Var(x)],
            ),
            metadata,
            inference,
            |_, _| Some(Default::default()),
        )
    }
}
