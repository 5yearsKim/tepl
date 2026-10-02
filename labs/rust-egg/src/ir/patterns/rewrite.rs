use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use egg::{
    Analysis, Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var,
};

use crate::ir::{OpAttrs, TensorLang};

use super::context::{OutputInference, TensorInfo, TensorMetadata};
use super::matcher::{TensorMatch, for_each_match_at};
use super::pattern::{AttrExpr, AttrVar, TensorExpr, TensorPattern};

pub type DerivedAttrs = std::collections::HashMap<AttrVar, OpAttrs>;

/// Build a reusable egg rewrite from a structural pattern, a RHS expression,
/// and a host function that checks semantics and derives RHS attributes.
/// The caller must prove semantic validity and output compatibility for the
/// entire RHS. Return `None` when that cannot be established. Prefer
/// `tensor_rewrite_checked` when metadata and operation inference are available.
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
    build_tensor_rewrite(name, lhs, rhs, move |graph, _, matched| {
        check_and_derive(graph, matched)
    })
}

/// Validate every RHS operation and require shape/dtype equality with the
/// matched root before inserting any nodes. Numerical legality remains the
/// responsibility of `check_and_derive` and the host's operation semantics.
pub fn tensor_rewrite_checked<N, M, I, F>(
    name: impl Into<Symbol>,
    lhs: TensorPattern,
    rhs: TensorExpr,
    metadata: M,
    inference: I,
    check_and_derive: F,
) -> Result<Rewrite<TensorLang, N>, String>
where
    N: Analysis<TensorLang>,
    M: TensorMetadata<N> + 'static,
    I: OutputInference + 'static,
    F: Fn(&EGraph<TensorLang, N>, &TensorMatch) -> Option<DerivedAttrs> + Send + Sync + 'static,
{
    let expression = rhs.clone();
    build_tensor_rewrite(name, lhs, rhs, move |graph, root, matched| {
        let derived = check_and_derive(graph, matched)?;
        validate_rhs(&expression, matched, &derived)?;
        let expected = metadata.info(graph, graph.find(root))?;
        let actual = infer_rhs(graph, &expression, matched, &derived, &metadata, &inference)?;
        (actual == expected).then_some(derived)
    })
}

fn infer_rhs<N: Analysis<TensorLang>, M: TensorMetadata<N>, I: OutputInference>(
    graph: &EGraph<TensorLang, N>,
    expr: &TensorExpr,
    matched: &TensorMatch,
    derived: &DerivedAttrs,
    metadata: &M,
    inference: &I,
) -> Option<TensorInfo> {
    match expr {
        TensorExpr::Var(var) => metadata.info(graph, graph.find(*matched.tensors.get(*var)?)),
        TensorExpr::Op {
            op,
            attrs,
            children,
        } => {
            let attrs = resolve_attrs(attrs, matched, derived)?;
            let operands = children
                .iter()
                .map(|child| infer_rhs(graph, child, matched, derived, metadata, inference))
                .collect::<Option<Vec<_>>>()?;
            let output = inference.infer_output(*op, &operands, &attrs)?;
            // The host may reject unsupported literal formats or values, but
            // cannot reinterpret an explicitly typed rank-zero literal.
            if let OpAttrs::Literal { dtype, .. } = &attrs {
                if output.dtype != *dtype || !output.shape.is_empty() {
                    return None;
                }
            }
            Some(output)
        }
    }
}

fn build_tensor_rewrite<N, F>(
    name: impl Into<Symbol>,
    lhs: TensorPattern,
    rhs: TensorExpr,
    check_and_derive: F,
) -> Result<Rewrite<TensorLang, N>, String>
where
    N: Analysis<TensorLang>,
    F: Fn(&EGraph<TensorLang, N>, Id, &TensorMatch) -> Option<DerivedAttrs> + Send + Sync + 'static,
{
    validate_rhs_definition(&rhs)?;
    let mut lhs_attrs = HashSet::new();
    lhs.attr_vars(&mut lhs_attrs);
    let mut rhs_captures = HashSet::new();
    rhs.captured_attrs(&mut rhs_captures);
    if !rhs_captures.is_subset(&lhs_attrs) {
        return Err("RHS refers to an attribute not captured on the LHS".into());
    }
    let mut lhs_vars = HashSet::new();
    lhs.vars(&mut lhs_vars);
    let mut lhs_vars: Vec<_> = lhs_vars.into_iter().collect();
    lhs_vars.sort_unstable();

    Rewrite::new(
        name,
        TensorSearcher {
            pattern: lhs.clone(),
        },
        TensorApplier {
            lhs,
            lhs_vars,
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
        for_each_match_at(egraph, eclass, &self.pattern, |matched| {
            if seen.insert(matched.tensors.clone()) {
                substs.push(matched.tensors);
                if substs.len() == limit {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        });
        (!substs.is_empty()).then_some(SearchMatches {
            eclass,
            substs,
            ast: None,
        })
    }

    fn search_with_limit(
        &self,
        egraph: &EGraph<TensorLang, N>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, TensorLang>> {
        match self.pattern.root_op() {
            Some(op) => egraph
                .classes_for_op(&op)
                .map(|ids| search_classes(self, egraph, ids, limit))
                .unwrap_or_default(),
            None => search_classes(self, egraph, egraph.classes().map(|class| class.id), limit),
        }
    }

    fn vars(&self) -> Vec<Var> {
        let mut found = HashSet::new();
        self.pattern.vars(&mut found);
        found.into_iter().collect()
    }
}

fn search_classes<'a, N: Analysis<TensorLang>>(
    searcher: &'a TensorSearcher,
    egraph: &EGraph<TensorLang, N>,
    classes: impl Iterator<Item = Id>,
    mut limit: usize,
) -> Vec<SearchMatches<'a, TensorLang>> {
    let mut results = Vec::new();
    for eclass in classes {
        if limit == 0 {
            break;
        }
        if let Some(found) = searcher.search_eclass_with_limit(egraph, eclass, limit) {
            limit -= found.substs.len();
            results.push(found);
        }
    }
    results
}

struct TensorApplier<F> {
    lhs: TensorPattern,
    lhs_vars: Vec<Var>,
    rhs: TensorExpr,
    check_and_derive: F,
}

impl<F> TensorApplier<F> {
    fn apply_candidates<N: Analysis<TensorLang>>(
        &self,
        egraph: &mut EGraph<TensorLang, N>,
        eclass: Id,
        candidates: &[TensorMatch],
        rule_name: Symbol,
    ) -> Vec<Id>
    where
        F: Fn(&EGraph<TensorLang, N>, Id, &TensorMatch) -> Option<DerivedAttrs>,
    {
        // Finish all semantic checks before changing this e-class. A rejected
        // RHS must not leave any of its intermediate nodes in the graph.
        let valid: Vec<_> = candidates
            .iter()
            .filter_map(|candidate| {
                let derived = (self.check_and_derive)(egraph, eclass, candidate)?;
                validate_rhs(&self.rhs, candidate, &derived)?;
                Some((candidate, derived))
            })
            .collect();

        let mut changed = Vec::new();
        for (matched, derived) in valid {
            let result = insert_rhs(egraph, &self.rhs, matched, &derived);
            if egraph.union_trusted(eclass, result, rule_name) {
                changed.push(result);
            }
        }
        changed
    }
}

impl<N, F> Applier<TensorLang, N> for TensorApplier<F>
where
    N: Analysis<TensorLang>,
    F: Fn(&EGraph<TensorLang, N>, Id, &TensorMatch) -> Option<DerivedAttrs> + Send + Sync,
{
    fn apply_matches(
        &self,
        egraph: &mut EGraph<TensorLang, N>,
        matches: &[SearchMatches<TensorLang>],
        rule_name: Symbol,
    ) -> Vec<Id> {
        let mut changed = Vec::new();
        for mat in matches {
            // Rematch once per e-class so different attribute witnesses stay
            // available, even when they share the same tensor substitution.
            let keys: Vec<_> = mat
                .substs
                .iter()
                .map(|subst| subst_key(egraph, subst, &self.lhs_vars))
                .collect();
            let mut groups: HashMap<Vec<Id>, Vec<TensorMatch>> = HashMap::new();
            for key in keys.iter().flatten() {
                groups.entry(key.clone()).or_default();
            }
            if groups.is_empty() {
                continue;
            }
            for_each_match_at(egraph, mat.eclass, &self.lhs, |candidate| {
                if let Some(key) = subst_key(egraph, &candidate.tensors, &self.lhs_vars) {
                    if let Some(group) = groups.get_mut(&key) {
                        group.push(candidate);
                    }
                }
                ControlFlow::Continue(())
            });
            for key in keys.into_iter().flatten() {
                if let Some(candidates) = groups.get(&key) {
                    changed
                        .extend(self.apply_candidates(egraph, mat.eclass, candidates, rule_name));
                }
            }
        }
        changed
    }

    fn apply_one(
        &self,
        egraph: &mut EGraph<TensorLang, N>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<TensorLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let key = match subst_key(egraph, subst, &self.lhs_vars) {
            Some(key) => key,
            None => return Vec::new(),
        };
        let mut candidates = Vec::new();
        for_each_match_at(egraph, eclass, &self.lhs, |candidate| {
            if subst_key(egraph, &candidate.tensors, &self.lhs_vars).as_ref() == Some(&key) {
                candidates.push(candidate);
            }
            ControlFlow::Continue(())
        });
        self.apply_candidates(egraph, eclass, &candidates, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        let mut found = HashSet::new();
        self.rhs.vars(&mut found);
        found.into_iter().collect()
    }
}

fn subst_key<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    subst: &Subst,
    vars: &[Var],
) -> Option<Vec<Id>> {
    vars.iter()
        .map(|var| subst.get(*var).map(|id| egraph.find(*id)))
        .collect()
}

// Validate static construction errors once when the rewrite is built.
fn validate_rhs_definition(expr: &TensorExpr) -> Result<(), String> {
    if let TensorExpr::Op {
        op,
        attrs,
        children,
    } = expr
    {
        if !op.arity().accepts(children.len()) {
            return Err(format!("invalid RHS arity for {}", op.name()));
        }
        if let AttrExpr::Exact(attrs) = attrs {
            TensorLang::new(*op, vec![Id::from(0); children.len()], attrs.clone())
                .map_err(|error| error.to_string())?;
        }
        for child in children {
            validate_rhs_definition(child)?;
        }
    }
    Ok(())
}

fn resolve_attrs(
    expr: &AttrExpr,
    matched: &TensorMatch,
    derived: &DerivedAttrs,
) -> Option<OpAttrs> {
    Some(match expr {
        AttrExpr::Exact(attrs) => attrs.clone(),
        AttrExpr::Captured(var) => matched.attrs.get(var)?.clone(),
        AttrExpr::Derived(var) => derived.get(var)?.clone(),
    })
}

// Check the entire RHS before adding nodes; no intermediate metadata is needed.
fn validate_rhs(expr: &TensorExpr, matched: &TensorMatch, derived: &DerivedAttrs) -> Option<()> {
    match expr {
        TensorExpr::Var(var) => {
            matched.tensors.get(*var)?;
        }
        TensorExpr::Op {
            op,
            attrs,
            children,
        } => {
            let attrs = resolve_attrs(attrs, matched, derived)?;
            TensorLang::new(*op, vec![Id::from(0); children.len()], attrs).ok()?;
            for child in children {
                validate_rhs(child, matched, derived)?;
            }
        }
    }
    Some(())
}

fn insert_rhs<N: Analysis<TensorLang>>(
    egraph: &mut EGraph<TensorLang, N>,
    expr: &TensorExpr,
    matched: &TensorMatch,
    derived: &DerivedAttrs,
) -> Id {
    match expr {
        TensorExpr::Var(var) => egraph.find(matched.tensors[*var]),
        TensorExpr::Op {
            op,
            attrs,
            children,
        } => {
            let attrs =
                resolve_attrs(attrs, matched, derived).expect("RHS descriptors were validated");
            let children = children
                .iter()
                .map(|child| insert_rhs(egraph, child, matched, derived))
                .collect::<Vec<_>>();
            egraph
                .add(TensorLang::new(*op, children, attrs).expect("RHS construction was validated"))
        }
    }
}
