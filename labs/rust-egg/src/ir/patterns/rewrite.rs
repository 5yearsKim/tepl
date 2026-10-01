use std::collections::HashSet;

use egg::{
    Analysis, Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var,
};

use crate::ir::{OpAttrs, OpKind, TensorLang};

use super::matcher::{TensorMatch, matches_at};
use super::pattern::{AttrExpr, AttrVar, TensorExpr, TensorPattern};

pub type DerivedAttrs = std::collections::HashMap<AttrVar, OpAttrs>;

/// Build a reusable egg rewrite from a structural pattern, a RHS expression,
/// and a host function that checks semantics and derives RHS attributes.
/// Return `None` from the host function when a match is not proven legal.
pub fn tensor_rewrite<N, F>(
    name: impl Into<Symbol>,
    lhs: TensorPattern,
    rhs: TensorExpr,
    check_and_derive: F,
) -> Result<Rewrite<TensorLang, N>, String>
where
    N: Analysis<TensorLang>,
    F: Fn(&EGraph<TensorLang, N>, &TensorMatch) -> Option<DerivedAttrs> + Send + Sync + 'static,
{
    let mut lhs_attrs = HashSet::new();
    lhs.attr_vars(&mut lhs_attrs);
    let mut rhs_captures = HashSet::new();
    rhs.captured_attrs(&mut rhs_captures);
    if !rhs_captures.is_subset(&lhs_attrs) {
        return Err("RHS refers to an attribute not captured on the LHS".into());
    }

    Rewrite::new(
        name,
        TensorSearcher {
            pattern: lhs.clone(),
        },
        TensorApplier {
            lhs,
            rhs,
            check_and_derive,
        },
    )
}

struct TensorSearcher {
    pattern: TensorPattern,
}

impl<N: Analysis<TensorLang>> Searcher<TensorLang, N> for TensorSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<TensorLang, N>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, TensorLang>> {
        if limit == 0 {
            return None;
        }
        let mut seen = HashSet::new();
        let mut substs = Vec::new();
        for matched in matches_at(egraph, eclass, &self.pattern) {
            if seen.insert(matched.tensors.clone()) {
                substs.push(matched.tensors);
                if substs.len() == limit {
                    break;
                }
            }
        }
        (!substs.is_empty()).then_some(SearchMatches {
            eclass,
            substs,
            ast: None,
        })
    }

    fn search_with_limit(
        &self,
        egraph: &EGraph<TensorLang, N>,
        mut limit: usize,
    ) -> Vec<SearchMatches<'_, TensorLang>> {
        let classes: Vec<_> = match self.pattern.root_op() {
            Some(op) => egraph
                .classes_for_op(&op)
                .map(|ids| ids.collect())
                .unwrap_or_default(),
            None => egraph.classes().map(|class| class.id).collect(),
        };
        let mut results = Vec::new();
        for eclass in classes {
            if limit == 0 {
                break;
            }
            if let Some(found) = self.search_eclass_with_limit(egraph, eclass, limit) {
                limit -= found.substs.len();
                results.push(found);
            }
        }
        results
    }

    fn vars(&self) -> Vec<Var> {
        let mut found = HashSet::new();
        self.pattern.vars(&mut found);
        found.into_iter().collect()
    }
}

struct TensorApplier<F> {
    lhs: TensorPattern,
    rhs: TensorExpr,
    check_and_derive: F,
}

impl<N, F> Applier<TensorLang, N> for TensorApplier<F>
where
    N: Analysis<TensorLang>,
    F: Fn(&EGraph<TensorLang, N>, &TensorMatch) -> Option<DerivedAttrs> + Send + Sync,
{
    fn apply_one(
        &self,
        egraph: &mut EGraph<TensorLang, N>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<TensorLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        // egg does not pass attribute values in Subst. Rematch, retaining only
        // the structural witnesses with this substitution.
        let mut lhs_vars = HashSet::new();
        self.lhs.vars(&mut lhs_vars);
        let lhs_vars: Vec<_> = lhs_vars.into_iter().collect();
        let valid: Vec<_> = matches_at(egraph, eclass, &self.lhs)
            .into_iter()
            .filter(|candidate| same_subst(egraph, &candidate.tensors, subst, &lhs_vars))
            .filter_map(|candidate| {
                let derived = (self.check_and_derive)(egraph, &candidate)?;
                plan_rhs(&self.rhs, &candidate, &derived)
            })
            .collect();

        let mut changed = Vec::new();
        for plan in valid {
            let result = insert_plan(egraph, plan);
            if egraph.union_trusted(eclass, result, rule_name) {
                changed.push(result);
            }
        }
        changed
    }

    fn vars(&self) -> Vec<Var> {
        let mut found = HashSet::new();
        self.rhs.vars(&mut found);
        found.into_iter().collect()
    }
}

fn same_subst<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    candidate: &Subst,
    expected: &Subst,
    vars: &[Var],
) -> bool {
    vars.iter()
        .all(|var| match (candidate.get(*var), expected.get(*var)) {
            (Some(a), Some(b)) => egraph.find(*a) == egraph.find(*b),
            _ => false,
        })
}

enum PlannedExpr {
    Input(Id),
    Node {
        op: OpKind,
        attrs: OpAttrs,
        children: Vec<PlannedExpr>,
    },
}

fn plan_rhs(
    expr: &TensorExpr,
    matched: &TensorMatch,
    derived: &DerivedAttrs,
) -> Option<PlannedExpr> {
    match expr {
        TensorExpr::Var(var) => Some(PlannedExpr::Input(*matched.tensors.get(*var)?)),
        TensorExpr::Op {
            op,
            attrs,
            children,
        } => {
            let attrs = match attrs {
                AttrExpr::Exact(attrs) => attrs.clone(),
                AttrExpr::Captured(var) => matched.attrs.get(var)?.clone(),
                AttrExpr::Derived(var) => derived.get(var)?.clone(),
            };
            // Check operator, attributes, and arity before inserting anything.
            TensorLang::new(*op, vec![Id::from(0); children.len()], attrs.clone()).ok()?;
            let children = children
                .iter()
                .map(|child| plan_rhs(child, matched, derived))
                .collect::<Option<Vec<_>>>()?;
            Some(PlannedExpr::Node {
                op: *op,
                attrs,
                children,
            })
        }
    }
}

fn insert_plan<N: Analysis<TensorLang>>(
    egraph: &mut EGraph<TensorLang, N>,
    plan: PlannedExpr,
) -> Id {
    match plan {
        PlannedExpr::Input(id) => egraph.find(id),
        PlannedExpr::Node {
            op,
            attrs,
            children,
        } => {
            let children = children
                .into_iter()
                .map(|child| insert_plan(egraph, child))
                .collect::<Vec<_>>();
            egraph.add(
                TensorLang::new(op, children, attrs).expect("RHS was checked before insertion"),
            )
        }
    }
}
