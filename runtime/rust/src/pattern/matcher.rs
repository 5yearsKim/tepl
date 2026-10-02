use std::collections::HashMap;
use std::ops::ControlFlow;

use egg::{Analysis, EGraph, Id, Language, Subst, Var};

use crate::ir::{OpAttrs, OpNode};

use super::pattern::{AttrPattern, AttrVar, TensorPattern};

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
    for_each_match_at(egraph, eclass, pattern, |matched| {
        matches.push(matched);
        ControlFlow::Continue(())
    });
    matches
}

/// Visit witnesses as they are found. Returning `Break` stops traversal,
/// including any remaining alternatives in parent e-classes.
pub(super) fn for_each_match_at<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    eclass: Id,
    pattern: &TensorPattern,
    mut visit: impl FnMut(TensorMatch) -> ControlFlow<()>,
) {
    let root = egraph.find(eclass);
    let _ = search(
        egraph,
        vec![(root, pattern)],
        TensorMatch {
            root,
            tensors: Subst::default(),
            attrs: HashMap::new(),
        },
        &mut visit,
    );
}

/// Search one branch depth-first. `pending` is a stack of checks still needed
/// for a complete match; `matched` contains this branch's bindings so far.
/// Only a complete match reaches `visit`. A failed check discards the branch.
fn search<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    mut pending: Vec<(Id, &TensorPattern)>,
    mut matched: TensorMatch,
    visit: &mut dyn FnMut(TensorMatch) -> ControlFlow<()>,
) -> ControlFlow<()> {
    while let Some((eclass, pattern)) = pending.pop() {
        let eclass = egraph.find(eclass);
        match pattern {
            TensorPattern::Var(var) => {
                if !bind_tensor(egraph, &mut matched, *var, eclass) {
                    return ControlFlow::Continue(());
                }
            }
            TensorPattern::Bind { var, pattern } => {
                // Capture this e-class, then check the inner pattern.
                if !bind_tensor(egraph, &mut matched, *var, eclass) {
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
                    match lookup_bound_pattern(egraph, pattern, &matched) {
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
                    if !attrs_match(attrs, node.attrs(), &matched) {
                        continue;
                    }

                    // Keep each candidate's bindings independent.
                    let mut branch = matched.clone();
                    if let AttrPattern::Bind(var) = attrs {
                        branch
                            .attrs
                            .entry(*var)
                            .or_insert_with(|| node.attrs().clone());
                    }
                    let mut remaining = pending.clone();
                    // Reverse puts the left child first; recurse with this stack.
                    remaining.extend(node.children().iter().copied().zip(children).rev());
                    search(egraph, remaining, branch, visit)?;
                }
                // Recursion already checked the remaining siblings.
                return ControlFlow::Continue(());
            }
        }
    }
    visit(matched)
}

/// Capture a tensor variable on first use; require e-class equality thereafter.
fn bind_tensor<N: Analysis<OpNode>>(
    egraph: &EGraph<OpNode, N>,
    matched: &mut TensorMatch,
    var: Var,
    eclass: Id,
) -> bool {
    if let Some(bound) = matched.tensors.get(var) {
        egraph.find(*bound) == eclass
    } else {
        matched.tensors.insert(var, eclass);
        true
    }
}

fn attrs_match(pattern: &AttrPattern, attrs: &OpAttrs, matched: &TensorMatch) -> bool {
    match pattern {
        AttrPattern::Any => true,
        AttrPattern::Literal { value, dtype } => match attrs {
            OpAttrs::TeplLiteral {
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
            // Lookup returns E3 without changing bindings or the stack.
            match egraph.lookup(node) {
                Some(id) => BoundLookup::Found(id),
                None => BoundLookup::Missing,
            }
        }
    }
}
