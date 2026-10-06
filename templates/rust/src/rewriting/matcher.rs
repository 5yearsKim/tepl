use ::std::collections::HashMap;
use ::std::ops::ControlFlow;

use ::egg::{Analysis, EGraph, Id, Language, Subst, Var};

use super::super::{OpAttrs, OpNode};

use super::checks::MatchChecks;
use super::constraints::TensorConstraints;
use super::context::TensorMetadata;
use super::pattern::{AttrPattern, AttrVar, TensorPattern};
use super::shape::MetadataBindings;

/// One exact structural match. `tensors` holds e-class IDs; `attrs` holds
/// values read from the particular e-nodes chosen while matching.
#[derive(Clone, Debug)]
pub struct TensorMatch {
    pub root: Id,
    pub tensors: Subst,
    pub attrs: HashMap<AttrVar, OpAttrs>,
}

/// Enumerate all pattern matches rooted in an e-class, including distinct
/// attribute captures that share the same tensor bindings.
pub fn matches_at<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    eclass: Id,
    pattern: &TensorPattern,
) -> Vec<TensorMatch> {
    let mut matches = Vec::new();
    for_each_match_at(
        egraph,
        eclass,
        pattern,
        &MatchChecks::tensors(TensorConstraints::default()),
        &|_: &EGraph<OpNode, N>, _: Id| None,
        |matched| {
            matches.push(matched);
            ControlFlow::Continue(())
        },
    );
    matches
}

/// Match with declared tensor restrictions; malformed plans return a construction error.
pub fn matches_at_with_constraints<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    eclass: Id,
    pattern: &TensorPattern,
    constraints: &TensorConstraints,
    metadata: &dyn TensorMetadata<N>,
) -> Result<Vec<TensorMatch>, String> {
    matches_at_with_checks(
        egraph,
        eclass,
        pattern,
        &MatchChecks::tensors(constraints.clone()),
        metadata,
    )
}

/// Match with shape declarations and an ordered plan of pure conditions.
pub fn matches_at_with_checks<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    eclass: Id,
    pattern: &TensorPattern,
    checks: &MatchChecks<N>,
    metadata: &dyn TensorMetadata<N>,
) -> Result<Vec<TensorMatch>, String> {
    checks.validate(pattern)?;
    let mut matches = Vec::new();
    for_each_match_at(egraph, eclass, pattern, checks, metadata, |matched| {
        matches.push(matched);
        ControlFlow::Continue(())
    });
    Ok(matches)
}

#[derive(Clone)]
struct SearchState {
    matched: TensorMatch,
    shapes: MetadataBindings,
    next_condition: usize,
}

/// Visit witnesses as they are found. Returning `Break` stops traversal.
/// Callers validate the immutable constraint plan before entering this traversal.
pub(super) fn for_each_match_at<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    eclass: Id,
    pattern: &TensorPattern,
    checks: &MatchChecks<N>,
    metadata: &dyn TensorMetadata<N>,
    mut visit: impl FnMut(TensorMatch) -> ControlFlow<()>,
) {
    let root = egraph.find(eclass);
    let _ = search(
        egraph,
        vec![(root, pattern)],
        SearchState {
            matched: TensorMatch {
                root,
                tensors: Subst::default(),
                attrs: HashMap::new(),
            },
            shapes: MetadataBindings::default(),
            next_condition: 0,
        },
        checks,
        metadata,
        &mut visit,
    );
}

/// Search one branch depth-first. `pending` is a stack of checks still needed
/// for a complete match; `matched` contains this branch's bindings so far.
/// Only a complete match reaches `visit`. A failed check discards the branch.
fn search<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    mut pending: Vec<(Id, &TensorPattern)>,
    mut state: SearchState,
    checks: &MatchChecks<N>,
    metadata: &dyn TensorMetadata<N>,
    visit: &mut dyn FnMut(TensorMatch) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if checks
        .advance(
            egraph,
            &state.matched,
            &state.shapes,
            &mut state.next_condition,
        )
        .is_none()
    {
        return ControlFlow::Continue(());
    }
    while let Some((eclass, pattern)) = pending.pop() {
        let eclass = egraph.find(eclass);
        match pattern {
            TensorPattern::Var(var) => {
                if !bind_tensor(egraph, &mut state, *var, eclass, checks, metadata) {
                    return ControlFlow::Continue(());
                }
            }
            TensorPattern::Bind { var, pattern } => {
                if !bind_tensor(egraph, &mut state, *var, eclass, checks, metadata) {
                    return ControlFlow::Continue(());
                }
                pending.push((eclass, pattern));
            }
            TensorPattern::Op {
                op,
                attrs,
                children,
            } => {
                // The memo index may be stale after a union until rebuild.
                if egraph.clean {
                    match lookup_bound_pattern(egraph, pattern, &state.matched) {
                        BoundLookup::Found(id) if id == eclass => continue,
                        BoundLookup::Found(_) | BoundLookup::Missing => {
                            return ControlFlow::Continue(());
                        }
                        BoundLookup::NeedsSearch => {}
                    }
                }

                for node in &egraph[eclass].nodes {
                    if node.op() != *op || node.children().len() != children.len() {
                        continue;
                    }
                    if !attrs_match(attrs, node.attrs(), &state.matched) {
                        continue;
                    }

                    let mut branch = state.clone();
                    if let AttrPattern::Bind(var) = attrs {
                        branch
                            .matched
                            .attrs
                            .entry(*var)
                            .or_insert_with(|| node.attrs().clone());
                    }
                    let mut remaining = pending.clone();
                    // The stack visits the left child first.
                    remaining.extend(node.children().iter().copied().zip(children).rev());
                    search(egraph, remaining, branch, checks, metadata, visit)?;
                }
                // Recursion already checked the remaining siblings.
                return ControlFlow::Continue(());
            }
        }
    }
    if checks.complete(state.next_condition) {
        visit(state.matched)
    } else {
        ControlFlow::Continue(())
    }
}

/// Capture a tensor variable on first use; require e-class equality thereafter.
fn bind_tensor<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    state: &mut SearchState,
    var: Var,
    eclass: Id,
    checks: &MatchChecks<N>,
    metadata: &dyn TensorMetadata<N>,
) -> bool {
    if let Some(bound) = state.matched.tensors.get(var) {
        // Existing bindings already passed all restrictions in this branch.
        return egraph.find(*bound) == eclass;
    }
    let constraints = checks.tensor_constraints();
    if constraints.contains(var) {
        let Some(info) = metadata.info(egraph, eclass) else {
            return false;
        };
        if constraints
            .check_capture(var, &info, &mut state.shapes)
            .is_none()
        {
            // Partial shape bindings are discarded with this entire branch.
            return false;
        }
    }
    state.matched.tensors.insert(var, eclass);
    checks
        .advance(
            egraph,
            &state.matched,
            &state.shapes,
            &mut state.next_condition,
        )
        .is_some()
}

fn attrs_match(pattern: &AttrPattern, attrs: &OpAttrs, matched: &TensorMatch) -> bool {
    match pattern {
        AttrPattern::Any => true,
        AttrPattern::Literal { value, dtype } => match attrs {
            OpAttrs::Literal {
                value: actual,
                dtype: actual_dtype,
            } => value == actual && dtype.is_none_or(|expected| Some(expected) == *actual_dtype),
            _ => false,
        },
        AttrPattern::Exact(expected) => attrs == expected,
        AttrPattern::Bind(var) => matched
            .attrs
            .get(var)
            .is_none_or(|previous| previous == attrs),
    }
}

enum BoundLookup {
    NeedsSearch,
    Missing,
    Found(Id),
}

/// Resolve a subtree using existing bindings and the e-graph's memo index.
/// No new bindings are introduced here. Wildcards and uncaptured attributes
/// must still enumerate witnesses, even if all tensor operands are known.
fn lookup_bound_pattern<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    pattern: &TensorPattern,
    matched: &TensorMatch,
) -> BoundLookup {
    match pattern {
        TensorPattern::Var(var) => match matched.tensors.get(*var) {
            Some(id) => BoundLookup::Found(egraph.find(*id)),
            None => BoundLookup::NeedsSearch,
        },
        TensorPattern::Bind { var, pattern } => {
            let Some(bound) = matched.tensors.get(*var) else {
                return BoundLookup::NeedsSearch;
            };
            match lookup_bound_pattern(egraph, pattern, matched) {
                BoundLookup::Found(id) if id != egraph.find(*bound) => BoundLookup::Missing,
                result => result,
            }
        }
        TensorPattern::Op {
            op,
            attrs,
            children,
        } => {
            let attrs = match attrs {
                AttrPattern::Exact(attrs) => attrs,
                AttrPattern::Bind(var) => match matched.attrs.get(var) {
                    Some(attrs) => attrs,
                    None => return BoundLookup::NeedsSearch,
                },
                AttrPattern::Any | AttrPattern::Literal { .. } => return BoundLookup::NeedsSearch,
            };
            let mut operands = Vec::new();
            for child in children {
                match lookup_bound_pattern(egraph, child, matched) {
                    BoundLookup::Found(id) => operands.push(id),
                    result => return result,
                }
            }
            let Ok(node) = OpNode::from_parts(*op, operands, attrs.clone()) else {
                return BoundLookup::Missing;
            };
            match egraph.lookup(node) {
                Some(id) => BoundLookup::Found(id),
                None => BoundLookup::Missing,
            }
        }
    }
}
