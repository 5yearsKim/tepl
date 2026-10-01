use std::collections::HashMap;
use std::ops::ControlFlow;

use egg::{Analysis, EGraph, Id, Language, Subst};

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
    let root = egraph.find(eclass);
    let _ = match_pattern(
        egraph,
        root,
        pattern,
        TensorMatch {
            root,
            tensors: Subst::default(),
            attrs: HashMap::new(),
        },
        &mut visit,
    );
}

fn match_pattern<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    eclass: Id,
    pattern: &TensorPattern,
    matched: TensorMatch,
    visit: &mut dyn FnMut(TensorMatch) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let eclass = egraph.find(eclass);
    match pattern {
        TensorPattern::Var(var) => {
            if let Some(bound) = matched.tensors.get(*var) {
                if egraph.find(*bound) == eclass {
                    visit(matched)
                } else {
                    ControlFlow::Continue(())
                }
            } else {
                let mut matched = matched;
                matched.tensors.insert(*var, eclass);
                visit(matched)
            }
        }
        TensorPattern::Bind { var, pattern } => {
            let mut matched = matched;
            if let Some(bound) = matched.tensors.get(*var) {
                if egraph.find(*bound) != eclass {
                    return ControlFlow::Continue(());
                }
            } else {
                matched.tensors.insert(*var, eclass);
            }
            match_pattern(egraph, eclass, pattern, matched, visit)
        }
        TensorPattern::Op {
            op,
            attrs,
            children,
        } => {
            // The memo index can be stale after a union until rebuild. Search
            // may use it, but application also calls this matcher on dirty graphs.
            if egraph.clean {
                match lookup_bound_pattern(egraph, pattern, &matched) {
                    BoundLookup::Found(id) if id == eclass => return visit(matched),
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
                let valid_attrs = match attrs {
                    AttrPattern::Any => true,
                    AttrPattern::Exact(expected) => node.attrs() == expected,
                    AttrPattern::Bind(var) => match matched.attrs.get(var) {
                        Some(previous) => previous == node.attrs(),
                        None => true,
                    },
                };
                if !valid_attrs {
                    continue;
                }

                let mut captured = matched.clone();
                if let AttrPattern::Bind(var) = attrs {
                    captured
                        .attrs
                        .entry(*var)
                        .or_insert_with(|| node.attrs().clone());
                }
                match_children(egraph, children, node.children(), captured, visit)?;
            }
            ControlFlow::Continue(())
        }
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
            let Ok(node) = TensorLang::new(*op, operands, attrs.clone()) else {
                return BoundLookup::Missing;
            };
            match egraph.lookup(node) {
                Some(id) => BoundLookup::Found(id),
                None => BoundLookup::Missing,
            }
        }
    }
}

fn match_children<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    patterns: &[TensorPattern],
    eclasses: &[Id],
    matched: TensorMatch,
    visit: &mut dyn FnMut(TensorMatch) -> ControlFlow<()>,
) -> ControlFlow<()> {
    match (patterns.split_first(), eclasses.split_first()) {
        (Some((pattern, rest)), Some((eclass, remaining))) => {
            match_pattern(egraph, *eclass, pattern, matched, &mut |branch| {
                match_children(egraph, rest, remaining, branch, visit)
            })
        }
        (None, None) => visit(matched),
        _ => unreachable!("pattern and e-node arities were checked"),
    }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, Var};

    use super::matches_at;
    use crate::ir::patterns::{AttrPattern, AttrVar, TensorPattern};
    use crate::ir::{OpAttrs, OpKind, TensorLang};

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
