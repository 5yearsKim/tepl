//! Reference lowering of `examples/inherited.tepl` after compile-time expansion.
//!
//! Abstract rules from `examples/abstract.tepl` are specialized here into
//! concrete patterns and host calls. No abstract parameters remain at runtime.

use egg::{Analysis, Rewrite, Var};

use crate::ir::patterns::{
    AttrExpr, AttrPattern, MatchContext, TensorExpr, TensorInfo, TensorMetadata, TensorPattern,
    tensor_rewrite,
};
use crate::ir::{OpAttrs, OpKind, TensorLang};

/// `commute(F = t.add)`.
pub mod rule_commute_add {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x), TensorPattern::Var(y)],
        )
    }

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

/// `commute(F = t.multiply)`.
pub mod rule_commute_mul {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Multiply,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x), TensorPattern::Var(y)],
        )
    }

    pub fn build_rewrite<N>() -> Result<Rewrite<TensorLang, N>, String>
    where
        N: Analysis<TensorLang>,
    {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Multiply,
            AttrExpr::Exact(OpAttrs::None),
            vec![TensorExpr::Var(y), TensorExpr::Var(x)],
        );
        tensor_rewrite("commute_mul", pattern(), rhs, |_, _| {
            Some(Default::default())
        })
    }
}

/// `commute(F = t.add)` restricted to equal vectors of length at most 1024.
pub mod rule_commute_small_vectors {
    use super::*;

    pub fn pattern() -> TensorPattern {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![TensorPattern::Var(x), TensorPattern::Var(y)],
        )
    }

    pub fn build_rewrite<N, M>(metadata: M) -> Result<Rewrite<TensorLang, N>, String>
    where
        N: Analysis<TensorLang>,
        M: TensorMetadata<N> + 'static,
    {
        let [x, y] = ["?X", "?Y"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![TensorExpr::Var(y), TensorExpr::Var(x)],
        );
        tensor_rewrite(
            "commute_small_vectors",
            pattern(),
            rhs,
            move |egraph, matched| {
                let ctx = MatchContext::new(egraph, matched, &metadata);
                let (x, y) = (ctx.tensor(x)?, ctx.tensor(y)?);
                let [n] = x.shape.as_slice() else {
                    return None;
                };
                if y.shape.as_slice() != [*n] || *n > 1024 {
                    return None;
                }
                Some(Default::default())
            },
        )
    }
}

/// `associate_right(F = t.add, allowed = can_reassociate_add)`.
pub mod rule_associate_add_right {
    use super::*;

    pub trait Functions: Send + Sync {
        fn can_reassociate_add(
            &self,
            x: &TensorInfo,
            y: &TensorInfo,
            z: &TensorInfo,
        ) -> Option<bool>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::op(
                    OpKind::Add,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![TensorPattern::Var(x), TensorPattern::Var(y)],
                ),
                TensorPattern::Var(z),
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
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::Var(x),
                TensorExpr::op(
                    OpKind::Add,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(y), TensorExpr::Var(z)],
                ),
            ],
        );
        tensor_rewrite(
            "associate_add_right",
            pattern(),
            rhs,
            move |egraph, matched| {
                let ctx = MatchContext::new(egraph, matched, &metadata);
                functions
                    .can_reassociate_add(&ctx.tensor(x)?, &ctx.tensor(y)?, &ctx.tensor(z)?)?
                    .then_some(Default::default())
            },
        )
    }
}

/// `associate_right(F = t.multiply, allowed = can_reassociate_mul)`.
pub mod rule_associate_mul_right {
    use super::*;

    pub trait Functions: Send + Sync {
        fn can_reassociate_mul(
            &self,
            x: &TensorInfo,
            y: &TensorInfo,
            z: &TensorInfo,
        ) -> Option<bool>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Multiply,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::op(
                    OpKind::Multiply,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![TensorPattern::Var(x), TensorPattern::Var(y)],
                ),
                TensorPattern::Var(z),
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
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Multiply,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::Var(x),
                TensorExpr::op(
                    OpKind::Multiply,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(y), TensorExpr::Var(z)],
                ),
            ],
        );
        tensor_rewrite(
            "associate_mul_right",
            pattern(),
            rhs,
            move |egraph, matched| {
                let ctx = MatchContext::new(egraph, matched, &metadata);
                functions
                    .can_reassociate_mul(&ctx.tensor(x)?, &ctx.tensor(y)?, &ctx.tensor(z)?)?
                    .then_some(Default::default())
            },
        )
    }
}

/// Addition reassociation with both the inherited host check and child constraints.
pub mod rule_associate_small_vectors {
    use super::*;

    pub trait Functions: Send + Sync {
        fn can_reassociate_add(
            &self,
            x: &TensorInfo,
            y: &TensorInfo,
            z: &TensorInfo,
        ) -> Option<bool>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::op(
                    OpKind::Add,
                    AttrPattern::Exact(OpAttrs::None),
                    vec![TensorPattern::Var(x), TensorPattern::Var(y)],
                ),
                TensorPattern::Var(z),
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
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::Var(x),
                TensorExpr::op(
                    OpKind::Add,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(y), TensorExpr::Var(z)],
                ),
            ],
        );
        tensor_rewrite(
            "associate_small_vectors",
            pattern(),
            rhs,
            move |egraph, matched| {
                let ctx = MatchContext::new(egraph, matched, &metadata);
                let (x, y, z) = (ctx.tensor(x)?, ctx.tensor(y)?, ctx.tensor(z)?);
                let [n] = x.shape.as_slice() else {
                    return None;
                };
                if y.shape.as_slice() != [*n] || z.shape.as_slice() != [*n] {
                    return None;
                }
                if !functions.can_reassociate_add(&x, &y, &z)? || *n > 1024 {
                    return None;
                }
                Some(Default::default())
            },
        )
    }
}

/// `distribute_left(product = t.multiply, sum = t.add, allowed = can_distribute_mul_over_add)`.
pub mod rule_distribute_mul_over_add {
    use super::*;

    pub trait Functions: Send + Sync {
        fn can_distribute_mul_over_add(
            &self,
            x: &TensorInfo,
            y: &TensorInfo,
            z: &TensorInfo,
        ) -> Option<bool>;
    }

    pub fn pattern() -> TensorPattern {
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        TensorPattern::op(
            OpKind::Multiply,
            AttrPattern::Exact(OpAttrs::None),
            vec![
                TensorPattern::Var(x),
                TensorPattern::op(
                    OpKind::Add,
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
        let [x, y, z] = ["?X", "?Y", "?Z"].map(|name| name.parse::<Var>().unwrap());
        let rhs = TensorExpr::op(
            OpKind::Add,
            AttrExpr::Exact(OpAttrs::None),
            vec![
                TensorExpr::op(
                    OpKind::Multiply,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(x), TensorExpr::Var(y)],
                ),
                TensorExpr::op(
                    OpKind::Multiply,
                    AttrExpr::Exact(OpAttrs::None),
                    vec![TensorExpr::Var(x), TensorExpr::Var(z)],
                ),
            ],
        );
        tensor_rewrite(
            "distribute_mul_over_add",
            pattern(),
            rhs,
            move |egraph, matched| {
                let ctx = MatchContext::new(egraph, matched, &metadata);
                functions
                    .can_distribute_mul_over_add(&ctx.tensor(x)?, &ctx.tensor(y)?, &ctx.tensor(z)?)?
                    .then_some(Default::default())
            },
        )
    }
}
