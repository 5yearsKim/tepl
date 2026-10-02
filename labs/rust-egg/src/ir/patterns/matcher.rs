//! Steps follow `(mul (add ?x ?y) (add ?x ?y))` matched at E4:
//! E1 = x, E2 = y, E3 = Add(E1, E2), E4 = Mul(E3, E3).
//! The Mul pattern has two Add children, each containing [Var(?x), Var(?y)].
//! Assume a rebuilt e-graph and Exact(None) attributes on all operators.
//! `pending` / `remaining` are stacks of (e-class, subpattern) checks;
//! the rightmost item runs next. Numbers follow execution, including recursion.

use std::collections::HashMap;
use std::ops::ControlFlow;

use egg::{Analysis, EGraph, Id, Language, Subst, Var};

use crate::ir::{OpAttrs, TensorLang};

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
pub fn matches_at<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
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
pub(super) fn for_each_match_at<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    eclass: Id,
    pattern: &TensorPattern,
    mut visit: impl FnMut(TensorMatch) -> ControlFlow<()>,
) {
    // Step 1: Start at E4: pending = [(E4, Mul)], bindings = {}.
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
fn search<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    mut pending: Vec<(Id, &TensorPattern)>,
    mut matched: TensorMatch,
    visit: &mut dyn FnMut(TensorMatch) -> ControlFlow<()>,
) -> ControlFlow<()> {
    // Step 2: Pop Mul; pending = [].
    // Step 6: Pop left Add; pending = [(E3, right Add)].
    // Step 11: After binding both variables, pop right Add; pending = [].
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
                        // Step 12: Right Add resolves to E3; skip scanning its nodes.
                        BoundLookup::Found(id) if id == eclass => continue,
                        BoundLookup::Found(_) | BoundLookup::Missing => {
                            return ControlFlow::Continue(());
                        }
                        BoundLookup::NeedsSearch => {}
                    }
                }

                // Step 4: Scan E4 for Mul(E3, E3).
                // Step 7: Left Add is still unbound; scan E3 for Add(E1, E2).
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
                    // Step 5: Mul -> remaining = [(E3, right Add), (E3, left Add)].
                    // Step 8: Left Add -> [(E3, right Add), (E2, ?y), (E1, ?x)].
                    // Reverse puts the left child first; recurse with this stack.
                    remaining.extend(node.children().iter().copied().zip(children).rev());
                    search(egraph, remaining, branch, visit)?;
                }
                // Recursion already checked the remaining siblings.
                return ControlFlow::Continue(());
            }
        }
    }
    // Step 13: Stack empty: emit {root: E4, tensors: {?x: E1, ?y: E2}, attrs: {}}.
    visit(matched)
}

/// Capture a tensor variable on first use; require e-class equality thereafter.
fn bind_tensor<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    matched: &mut TensorMatch,
    var: Var,
    eclass: Id,
) -> bool {
    if let Some(bound) = matched.tensors.get(var) {
        egraph.find(*bound) == eclass
    } else {
        // Step 9: Pop/bind ?x -> E1; pending = [(E3, right Add), (E2, ?y)].
        // Step 10: Pop/bind ?y -> E2; pending = [(E3, right Add)].
        matched.tensors.insert(var, eclass);
        true
    }
}

fn attrs_match(pattern: &AttrPattern, attrs: &OpAttrs, matched: &TensorMatch) -> bool {
    match pattern {
        AttrPattern::Any => true,
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
fn lookup_bound_pattern<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    pattern: &TensorPattern,
    matched: &TensorMatch,
) -> BoundLookup {
    match pattern {
        TensorPattern::Var(var) => match matched.tensors.get(*var) {
            Some(id) => BoundLookup::Found(egraph.find(*id)),
            // Step 3: Mul's lookup reaches unbound ?x, so normal search is needed.
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
                AttrPattern::Any => return BoundLookup::NeedsSearch,
            };
            let mut operands = Vec::new();
            for child in children {
                match lookup_bound_pattern(egraph, child, matched) {
                    BoundLookup::Found(id) => operands.push(id),
                    result => return result,
                }
            }
            // Step 12: Bound [?x, ?y] becomes [E1, E2]; build Add(E1, E2).
            let Ok(node) = TensorLang::new(*op, operands, attrs.clone()) else {
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

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use egg::{EGraph, Language, Var};

    use super::{for_each_match_at, matches_at};
    use crate::ir::patterns::{AttrPattern, AttrVar, TensorPattern};
    use crate::ir::{OpAttrs, OpKind, TensorLang};

    #[test]
    fn sibling_alternatives_are_independent_ordered_and_can_stop_early() {
        let mut egraph = EGraph::<TensorLang, ()>::default();
        let x = egraph.add(TensorLang::symbol("x"));
        let y = egraph.add(TensorLang::symbol("y"));
        let xy = egraph.add(TensorLang::binary(OpKind::Add, x, y).unwrap());
        let yx = egraph.add(TensorLang::binary(OpKind::Add, y, x).unwrap());
        egraph.union(xy, yx);
        let root = egraph.add(TensorLang::binary(OpKind::Add, xy, xy).unwrap());
        egraph.rebuild();

        let [a, b, c, d] = ["?a", "?b", "?c", "?d"].map(|s| s.parse::<Var>().unwrap());
        let add =
            |children| TensorPattern::op(OpKind::Add, AttrPattern::Exact(OpAttrs::None), children);
        let pattern = add(vec![
            add(vec![TensorPattern::Var(a), TensorPattern::Var(b)]),
            add(vec![TensorPattern::Var(c), TensorPattern::Var(d)]),
        ]);
        let bindings = |m: &super::TensorMatch| [a, b, c, d].map(|var| m.tensors[var]);
        // Preserve the e-graph's node order, with the left child's alternatives
        // as the outer loop and the right child's alternatives as the inner loop.
        let nodes = &egraph[egraph.find(xy)].nodes;
        let expected: Vec<_> = nodes
            .iter()
            .flat_map(|left| {
                nodes.iter().map(move |right| {
                    [
                        left.children()[0],
                        left.children()[1],
                        right.children()[0],
                        right.children()[1],
                    ]
                })
            })
            .collect();
        let all = matches_at(&egraph, root, &pattern);
        assert_eq!(all.len(), 4);
        assert_eq!(all.iter().map(bindings).collect::<Vec<_>>(), expected);

        let mut visited = Vec::new();
        for_each_match_at(&egraph, root, &pattern, |matched| {
            visited.push(bindings(&matched));
            ControlFlow::Break(())
        });
        assert_eq!(visited, expected[..1]);
    }

    #[test]
    fn repeated_attribute_binding_preserves_each_witness() {
        let mut egraph = EGraph::<TensorLang, ()>::default();
        let x = egraph.add(TensorLang::symbol("x"));
        let y = egraph.add(TensorLang::symbol("y"));
        let first = OpAttrs::DotGeneral {
            lhs_contracting: vec![0],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let second = OpAttrs::DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![1],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let first_id =
            egraph.add(TensorLang::new(OpKind::DotGeneral, vec![x, y], first.clone()).unwrap());
        let second_id =
            egraph.add(TensorLang::new(OpKind::DotGeneral, vec![x, y], second.clone()).unwrap());
        egraph.union(first_id, second_id);
        let root = egraph.add(TensorLang::binary(OpKind::Add, first_id, first_id).unwrap());
        egraph.rebuild();

        let [x_var, y_var] = ["?x", "?y"].map(|name| name.parse::<Var>().unwrap());
        let attr_var = AttrVar::from("dot");
        let dot = || {
            TensorPattern::op(
                OpKind::DotGeneral,
                AttrPattern::Bind(attr_var),
                vec![TensorPattern::Var(x_var), TensorPattern::Var(y_var)],
            )
        };
        let lhs = TensorPattern::op(
            OpKind::Add,
            AttrPattern::Exact(OpAttrs::None),
            vec![dot(), dot()],
        );
        let found = matches_at(&egraph, root, &lhs);
        assert_eq!(found.len(), 2);
        assert!(found.iter().any(|m| m.attrs[&attr_var] == first));
        assert!(found.iter().any(|m| m.attrs[&attr_var] == second));
        assert!(found.iter().all(|m| m.tensors[x_var] == x));
    }
}
