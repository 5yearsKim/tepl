use ::std::collections::{HashMap, HashSet};
use ::std::ops::ControlFlow;
use ::std::sync::Arc;

use ::egg::{
    Analysis, Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var,
};

use super::super::{OpAttrs, OpNode};

use super::checks::MatchChecks;
use super::constraints::TensorConstraints;
use super::context::{OutputInference, TensorInfo, TensorMetadata};
use super::matcher::{TensorMatch, for_each_match_at};
use super::pattern::{AttrExpr, AttrVar, TensorExpr, TensorPattern};
use super::shape::ShapeBindings;

pub type DerivedAttrs = ::std::collections::HashMap<AttrVar, OpAttrs>;

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
) -> Result<Rewrite<OpNode, N>, String>
where
    N: Analysis<OpNode> + 'static,
    M: TensorMetadata<N> + 'static,
    I: OutputInference + 'static,
    F: Fn(&EGraph<OpNode, N>, &TensorMatch) -> Option<DerivedAttrs> + Send + Sync + 'static,
{
    tensor_rewrite_checked_with_constraints(
        name,
        lhs,
        rhs,
        TensorConstraints::default(),
        metadata,
        inference,
        move |graph, matched, _| check_and_derive(graph, matched),
    )
}

/// Prune declared shape/dtype mismatches while matching. Recheck declarations
/// before calling semantic functions with the current dimension bindings.
pub fn tensor_rewrite_checked_with_constraints<N, M, I, F>(
    name: impl Into<Symbol>,
    lhs: TensorPattern,
    rhs: TensorExpr,
    constraints: TensorConstraints,
    metadata: M,
    inference: I,
    check_and_derive: F,
) -> Result<Rewrite<OpNode, N>, String>
where
    N: Analysis<OpNode> + 'static,
    M: TensorMetadata<N> + 'static,
    I: OutputInference + 'static,
    F: Fn(&EGraph<OpNode, N>, &TensorMatch, &ShapeBindings) -> Option<DerivedAttrs>
        + Send
        + Sync
        + 'static,
{
    tensor_rewrite_checked_with_checks(
        name,
        lhs,
        rhs,
        MatchChecks::tensors(constraints),
        metadata,
        inference,
        check_and_derive,
    )
}

/// Shape declarations and ordered pure conditions prune search branches and are
/// rechecked before the application-time callback. Host calls belong in that callback.
pub fn tensor_rewrite_checked_with_checks<N, M, I, F>(
    name: impl Into<Symbol>,
    lhs: TensorPattern,
    rhs: TensorExpr,
    checks: MatchChecks<N>,
    metadata: M,
    inference: I,
    check_and_derive: F,
) -> Result<Rewrite<OpNode, N>, String>
where
    N: Analysis<OpNode> + 'static,
    M: TensorMetadata<N> + 'static,
    I: OutputInference + 'static,
    F: Fn(&EGraph<OpNode, N>, &TensorMatch, &ShapeBindings) -> Option<DerivedAttrs>
        + Send
        + Sync
        + 'static,
{
    checks.validate(&lhs)?;
    let checks = Arc::new(checks);
    let metadata = Arc::new(metadata);
    let checker_checks = checks.clone();
    let checker_metadata = metadata.clone();
    let expression = rhs.clone();
    build_tensor_rewrite(
        name,
        lhs,
        rhs,
        checks,
        metadata,
        move |graph, root, matched| {
            let dimensions =
                checker_checks.check_match(graph, matched, checker_metadata.as_ref())?;
            let derived = check_and_derive(graph, matched, &dimensions)?;
            validate_rhs(&expression, matched, &derived)?;
            let expected = checker_metadata.info(graph, graph.find(root))?;
            let (resolved, actual) = infer_rhs(
                graph,
                &expression,
                matched,
                &derived,
                checker_metadata.as_ref(),
                &inference,
                &expected,
            )?;
            (actual == expected).then_some((derived, resolved))
        },
    )
}

fn infer_rhs<N: Analysis<OpNode>, M: TensorMetadata<N> + ?Sized, I: OutputInference>(
    graph: &EGraph<OpNode, N>,
    expr: &TensorExpr,
    matched: &TensorMatch,
    derived: &DerivedAttrs,
    metadata: &M,
    inference: &I,
    expected: &TensorInfo,
) -> Option<(TensorExpr, TensorInfo)> {
    match expr {
        TensorExpr::Var(var) => Some((
            expr.clone(),
            metadata.info(graph, graph.find(*matched.tensors.get(*var)?))?,
        )),
        TensorExpr::Op {
            op,
            attrs,
            children,
        } => {
            let mut attrs = resolve_attrs(attrs, matched, derived)?;
            let mut resolved = Vec::new();
            let mut operands = Vec::new();
            for child in children {
                let (expression, info) = infer_rhs(
                    graph, child, matched, derived, metadata, inference, expected,
                )?;
                resolved.push(expression);
                operands.push(info);
            }
            let output = if let OpAttrs::Literal { value, dtype } = &mut attrs {
                let concrete = inference.infer_literal(value, *dtype, expected)?;
                if dtype.is_some_and(|annotation| annotation != concrete) {
                    return None;
                }
                *dtype = Some(concrete);
                let output = inference.infer_output(*op, &operands, &attrs)?;
                if output.dtype != concrete || !output.shape.is_empty() {
                    return None;
                }
                output
            } else {
                inference.infer_output(*op, &operands, &attrs)?
            };
            OpNode::from_parts(*op, vec![Id::from(0); resolved.len()], attrs.clone()).ok()?;
            Some((
                TensorExpr::op(*op, AttrExpr::Exact(attrs), resolved),
                output,
            ))
        }
    }
}

fn build_tensor_rewrite<N, M, F>(
    name: impl Into<Symbol>,
    lhs: TensorPattern,
    rhs: TensorExpr,
    checks: Arc<MatchChecks<N>>,
    metadata: Arc<M>,
    check_and_derive: F,
) -> Result<Rewrite<OpNode, N>, String>
where
    N: Analysis<OpNode> + 'static,
    M: TensorMetadata<N> + 'static,
    F: Fn(&EGraph<OpNode, N>, Id, &TensorMatch) -> Option<(DerivedAttrs, TensorExpr)>
        + Send
        + Sync
        + 'static,
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
            checks: checks.clone(),
            metadata: metadata.clone(),
        },
        TensorApplier {
            lhs,
            lhs_vars,
            rhs,
            checks,
            metadata,
            check_and_derive,
        },
    )
}

struct TensorSearcher<N: Analysis<OpNode>, M> {
    pattern: TensorPattern,
    checks: Arc<MatchChecks<N>>,
    metadata: Arc<M>,
}

impl<N: Analysis<OpNode>, M: TensorMetadata<N>> Searcher<OpNode, N> for TensorSearcher<N, M> {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<OpNode, N>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, OpNode>> {
        if limit == 0 {
            return None;
        }
        let mut seen = HashSet::new();
        let mut substs = Vec::new();
        for_each_match_at(
            egraph,
            eclass,
            &self.pattern,
            &self.checks,
            self.metadata.as_ref(),
            |matched| {
                if seen.insert(matched.tensors.clone()) {
                    substs.push(matched.tensors);
                    if substs.len() == limit {
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            },
        );
        (!substs.is_empty()).then_some(SearchMatches {
            eclass,
            substs,
            ast: None,
        })
    }

    fn search_with_limit(
        &self,
        egraph: &EGraph<OpNode, N>,
        limit: usize,
    ) -> Vec<SearchMatches<'_, OpNode>> {
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

fn search_classes<'a, N: Analysis<OpNode>, M: TensorMetadata<N>>(
    searcher: &'a TensorSearcher<N, M>,
    egraph: &EGraph<OpNode, N>,
    classes: impl Iterator<Item = Id>,
    mut limit: usize,
) -> Vec<SearchMatches<'a, OpNode>> {
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

struct TensorApplier<N: Analysis<OpNode>, M, F> {
    lhs: TensorPattern,
    lhs_vars: Vec<Var>,
    rhs: TensorExpr,
    checks: Arc<MatchChecks<N>>,
    metadata: Arc<M>,
    check_and_derive: F,
}

impl<N: Analysis<OpNode>, M, F> TensorApplier<N, M, F> {
    fn apply_candidates(
        &self,
        egraph: &mut EGraph<OpNode, N>,
        eclass: Id,
        candidates: &[TensorMatch],
        rule_name: Symbol,
    ) -> Vec<Id>
    where
        M: TensorMetadata<N>,
        F: Fn(&EGraph<OpNode, N>, Id, &TensorMatch) -> Option<(DerivedAttrs, TensorExpr)>,
    {
        // Finish all semantic checks before changing this e-class. A rejected
        // RHS must not leave any of its intermediate nodes in the graph.
        let valid: Vec<_> = candidates
            .iter()
            .filter_map(|candidate| {
                let (derived, resolved) = (self.check_and_derive)(egraph, eclass, candidate)?;
                validate_rhs(&resolved, candidate, &derived)?;
                Some((candidate, derived, resolved))
            })
            .collect();

        let mut changed = Vec::new();
        for (matched, derived, resolved) in valid {
            let result = insert_rhs(egraph, &resolved, matched, &derived);
            if egraph.union_trusted(eclass, result, rule_name) {
                changed.push(result);
            }
        }
        changed
    }
}

impl<N, M, F> Applier<OpNode, N> for TensorApplier<N, M, F>
where
    N: Analysis<OpNode> + 'static,
    M: TensorMetadata<N>,
    F: Fn(&EGraph<OpNode, N>, Id, &TensorMatch) -> Option<(DerivedAttrs, TensorExpr)> + Send + Sync,
{
    fn apply_matches(
        &self,
        egraph: &mut EGraph<OpNode, N>,
        matches: &[SearchMatches<OpNode>],
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
            for_each_match_at(
                egraph,
                mat.eclass,
                &self.lhs,
                &self.checks,
                self.metadata.as_ref(),
                |candidate| {
                    if let Some(key) = subst_key(egraph, &candidate.tensors, &self.lhs_vars) {
                        if let Some(group) = groups.get_mut(&key) {
                            group.push(candidate);
                        }
                    }
                    ControlFlow::Continue(())
                },
            );
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
        egraph: &mut EGraph<OpNode, N>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<OpNode>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let key = match subst_key(egraph, subst, &self.lhs_vars) {
            Some(key) => key,
            None => return Vec::new(),
        };
        let mut candidates = Vec::new();
        for_each_match_at(
            egraph,
            eclass,
            &self.lhs,
            &self.checks,
            self.metadata.as_ref(),
            |candidate| {
                if subst_key(egraph, &candidate.tensors, &self.lhs_vars).as_ref() == Some(&key) {
                    candidates.push(candidate);
                }
                ControlFlow::Continue(())
            },
        );
        self.apply_candidates(egraph, eclass, &candidates, rule_name)
    }

    fn vars(&self) -> Vec<Var> {
        let mut found = HashSet::new();
        self.rhs.vars(&mut found);
        found.into_iter().collect()
    }
}

fn subst_key<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
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
            OpNode::from_parts(*op, vec![Id::from(0); children.len()], attrs.clone())
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
            OpNode::from_parts(*op, vec![Id::from(0); children.len()], attrs).ok()?;
            for child in children {
                validate_rhs(child, matched, derived)?;
            }
        }
    }
    Some(())
}

fn insert_rhs<N: Analysis<OpNode>>(
    egraph: &mut EGraph<OpNode, N>,
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
            egraph.add(
                OpNode::from_parts(*op, children, attrs).expect("RHS construction was validated"),
            )
        }
    }
}
