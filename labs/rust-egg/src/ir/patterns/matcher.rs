use std::collections::HashMap;

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
    let root = egraph.find(eclass);
    match_pattern(
        egraph,
        root,
        pattern,
        TensorMatch {
            root,
            tensors: Subst::default(),
            attrs: HashMap::new(),
        },
    )
}

fn match_pattern<N: Analysis<TensorLang>>(
    egraph: &EGraph<TensorLang, N>,
    eclass: Id,
    pattern: &TensorPattern,
    matched: TensorMatch,
) -> Vec<TensorMatch> {
    let eclass = egraph.find(eclass);
    match pattern {
        TensorPattern::Var(var) => {
            if let Some(bound) = matched.tensors.get(*var) {
                if egraph.find(*bound) == eclass {
                    vec![matched]
                } else {
                    vec![]
                }
            } else {
                let mut matched = matched;
                matched.tensors.insert(*var, eclass);
                vec![matched]
            }
        }
        TensorPattern::Bind { var, pattern } => {
            let mut matched = matched;
            if let Some(bound) = matched.tensors.get(*var) {
                if egraph.find(*bound) != eclass {
                    return vec![];
                }
            } else {
                matched.tensors.insert(*var, eclass);
            }
            match_pattern(egraph, eclass, pattern, matched)
        }
        TensorPattern::Op {
            op,
            attrs,
            children,
        } => {
            let mut results = Vec::new();
            for node in &egraph[eclass].nodes {
                if node.op() != *op || node.children().len() != children.len() {
                    continue;
                }
                let mut captured = matched.clone();
                let valid_attrs = match attrs {
                    AttrPattern::Any => true,
                    AttrPattern::Exact(expected) => node.attrs() == expected,
                    AttrPattern::Bind(var) => match captured.attrs.get(var) {
                        Some(previous) => previous == node.attrs(),
                        None => {
                            captured.attrs.insert(*var, node.attrs().clone());
                            true
                        }
                    },
                };
                if !valid_attrs {
                    continue;
                }

                let mut branches = vec![captured];
                for (child_pattern, child_id) in children.iter().zip(node.children()) {
                    branches = branches
                        .into_iter()
                        .flat_map(|branch| match_pattern(egraph, *child_id, child_pattern, branch))
                        .collect();
                    if branches.is_empty() {
                        break;
                    }
                }
                results.extend(branches);
            }
            results
        }
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
