use crate::matching::{Bindings, Constraint, Pattern, constraints, matches};
use crate::values::{AttributeValue, Axis, invalid};
use crate::{Budget, EvalError, EvalResult, Node, Program, TensorAnalysis, TensorFacts, Value};
use egg::{Applier, EGraph, Id, PatternAst, Rewrite, SearchMatches, Searcher, Subst, Symbol, Var};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub struct RuleContext<'a, P: Program> {
    graph: &'a EGraph<Node<P>, TensorAnalysis<P>>,
    pub bindings: Bindings,
    pub budget: Budget,
    axis_cache: BTreeMap<(usize, usize), Value>,
    active_axes: BTreeSet<(usize, usize)>,
    pub root: Id,
}
impl<'a, P: Program> RuleContext<'a, P> {
    pub fn new(
        graph: &'a EGraph<Node<P>, TensorAnalysis<P>>,
        root: Id,
        bindings: Bindings,
    ) -> Self {
        Self {
            graph,
            bindings,
            root,
            budget: Budget::default(),
            axis_cache: BTreeMap::new(),
            active_axes: BTreeSet::new(),
        }
    }
    pub fn capture(&self, id: usize) -> EvalResult<Value> {
        self.bindings
            .captures
            .get(&id)
            .copied()
            .map(Value::Tensor)
            .ok_or_else(|| invalid("missing capture"))
    }
    pub fn descriptor(&self, id: usize) -> EvalResult<Value> {
        self.bindings
            .descriptors
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("missing attributes"))
    }
    pub fn dimension(&self, id: usize) -> EvalResult<Value> {
        self.bindings
            .dimensions
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("missing dimension"))
    }
    pub fn dtype_variable(&self, id: usize) -> EvalResult<Value> {
        self.bindings
            .dtypes
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("missing dtype variable"))
    }
    pub fn local(&self, id: usize) -> EvalResult<Value> {
        self.bindings
            .locals
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("missing local"))
    }
    pub fn shape(&self, id: usize) -> EvalResult<Value> {
        self.facts(id)?.shape_value()
    }
    pub fn dtype(&self, id: usize) -> EvalResult<Value> {
        self.facts(id)?.dtype_value()
    }
    pub fn facts(&self, id: usize) -> EvalResult<&TensorFacts> {
        let id = self
            .bindings
            .captures
            .get(&id)
            .ok_or_else(|| invalid("missing capture"))?;
        Ok(&self.graph[*id].data)
    }
    pub fn cached_axis(&self, owner: usize, group: usize) -> Option<Value> {
        self.axis_cache.get(&(owner, group)).cloned()
    }
    pub fn enter_axis(&mut self, owner: usize, group: usize) -> EvalResult<()> {
        crate::error::require(
            self.active_axes.insert((owner, group)),
            "cyclic axis definition",
        )
    }
    pub fn finish_axis(&mut self, owner: usize, group: usize, extents: Value) -> EvalResult<Value> {
        let xs = extents.dimensions()?;
        let axes = Value::List(
            xs.into_iter()
                .enumerate()
                .map(|(member, extent)| {
                    Value::Axis(Axis {
                        owner,
                        group,
                        member,
                        extent: extent.into(),
                    })
                })
                .collect(),
        );
        self.active_axes.remove(&(owner, group));
        self.axis_cache.insert((owner, group), axes.clone());
        Ok(axes)
    }
    pub fn captured_rhs(&self, capture: usize) -> EvalResult<PreparedReplacement<P>> {
        let id = *self
            .bindings
            .captures
            .get(&capture)
            .ok_or_else(|| invalid("missing capture"))?;
        Ok(PreparedReplacement::Existing {
            id,
            facts: self.graph[id].data.clone(),
        })
    }
    pub fn operation_rhs(
        &mut self,
        operation: usize,
        attrs: Value,
        children: Vec<PreparedReplacement<P>>,
    ) -> EvalResult<PreparedReplacement<P>> {
        let attrs = if attrs == Value::None {
            None
        } else {
            Some(P::Attrs::from_value(&attrs)?)
        };
        let node = Node::operation(
            P::operation(operation),
            attrs,
            vec![Id::from(0); children.len()],
        )?;
        self.prepared(node, children)
    }
    pub fn literal_rhs(
        &mut self,
        value: &str,
        dtype: Option<crate::DType>,
    ) -> EvalResult<PreparedReplacement<P>> {
        let dtype = dtype.or_else(|| match self.graph[self.root].data.dtype {
            crate::Fact::Known(d) => Some(d),
            _ => None,
        });
        self.prepared(Node::literal(value, dtype)?, vec![])
    }
    fn prepared(
        &mut self,
        node: Node<P>,
        children: Vec<PreparedReplacement<P>>,
    ) -> EvalResult<PreparedReplacement<P>> {
        let operands = children
            .iter()
            .map(|c| c.facts().clone())
            .collect::<Vec<_>>();
        let facts =
            self.graph
                .analysis
                .infer_node_with_budget(&node, &operands, &mut self.budget)?;
        Ok(PreparedReplacement::New {
            node,
            children,
            facts,
        })
    }
}

#[derive(Clone, Debug)]
pub enum PreparedReplacement<P: Program> {
    Existing {
        id: Id,
        facts: TensorFacts,
    },
    New {
        node: Node<P>,
        children: Vec<Self>,
        facts: TensorFacts,
    },
}
impl<P: Program> PreparedReplacement<P> {
    pub fn facts(&self) -> &TensorFacts {
        match self {
            Self::Existing { facts, .. } | Self::New { facts, .. } => facts,
        }
    }
    fn insert(self, graph: &mut EGraph<Node<P>, TensorAnalysis<P>>) -> Id {
        match self {
            Self::Existing { id, .. } => graph.find(id),
            Self::New {
                mut node, children, ..
            } => {
                let ids = children
                    .into_iter()
                    .map(|c| c.insert(graph))
                    .collect::<Vec<_>>();
                egg::Language::children_mut(&mut node).copy_from_slice(&ids);
                graph.add(node)
            }
        }
    }
}
pub type Prepare<P> = fn(&mut RuleContext<'_, P>) -> EvalResult<Option<PreparedReplacement<P>>>;
#[derive(Clone)]
pub struct RuleDefinition<P: Program> {
    pub name: &'static str,
    pub source: &'static str,
    pub lhs: Pattern,
    pub constraints: Vec<Constraint>,
    pub prepare: Prepare<P>,
}
impl<P: Program> RuleDefinition<P> {
    pub fn build(self) -> EvalResult<Rewrite<Node<P>, TensorAnalysis<P>>> {
        let adapter = Adapter(Arc::new(self));
        Rewrite::new(adapter.0.name, adapter.clone(), adapter).map_err(EvalError::Invalid)
    }
}
#[derive(Clone)]
struct Adapter<P: Program>(Arc<RuleDefinition<P>>);
fn variable(capture: usize) -> Var {
    format!("?c{capture}").parse().unwrap()
}
fn captures(pattern: &Pattern, out: &mut BTreeSet<usize>) {
    match pattern {
        Pattern::Capture(c) => {
            out.insert(*c);
        }
        Pattern::Bind(c, inner) => {
            out.insert(*c);
            captures(inner, out);
        }
        Pattern::Operation { operands, .. } => {
            for child in operands {
                captures(child, out);
            }
        }
        _ => {}
    }
}
impl<P: Program> Searcher<Node<P>, TensorAnalysis<P>> for Adapter<P> {
    fn search_eclass_with_limit(
        &self,
        graph: &EGraph<Node<P>, TensorAnalysis<P>>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, Node<P>>> {
        if limit == 0 || graph.analysis.check().is_err() {
            return None;
        }
        let witnesses = match matches(&self.0.lhs, eclass, graph, &mut Budget::default()) {
            Ok(xs) => xs,
            Err(e) => {
                self.report(graph, e);
                return None;
            }
        };
        let mut substs = vec![];
        let mut seen = BTreeSet::new();
        for witness in witnesses {
            let key = witness
                .captures
                .iter()
                .map(|(c, id)| (*c, graph.find(*id)))
                .collect::<Vec<_>>();
            if !seen.insert(key.clone()) {
                continue;
            }
            let mut subst = Subst::default();
            for (capture, id) in key {
                subst.insert(variable(capture), id);
            }
            substs.push(subst);
            if substs.len() == limit {
                break;
            }
        }
        if substs.is_empty() {
            None
        } else {
            Some(SearchMatches {
                eclass,
                substs,
                ast: None,
            })
        }
    }
    fn vars(&self) -> Vec<Var> {
        let mut ids = BTreeSet::new();
        captures(&self.0.lhs, &mut ids);
        ids.into_iter().map(variable).collect()
    }
}
impl<P: Program> Adapter<P> {
    fn report(&self, graph: &EGraph<Node<P>, TensorAnalysis<P>>, error: EvalError) {
        graph.analysis.report(match error {
            EvalError::Invalid(message) => EvalError::Invalid(format!(
                "{}: rule {}: {message}",
                self.0.source, self.0.name
            )),
            other => other,
        });
    }
    fn apply_root(
        &self,
        graph: &mut EGraph<Node<P>, TensorAnalysis<P>>,
        root: Id,
        substs: &[Subst],
        name: Symbol,
    ) -> Vec<Id> {
        if graph.analysis.check().is_err() {
            return vec![];
        }
        let prepare = || -> EvalResult<Vec<PreparedReplacement<P>>> {
            let witnesses = matches(&self.0.lhs, root, graph, &mut Budget::default())?;
            let mut prepared = vec![];
            for mut bindings in witnesses {
                if !substs.iter().any(|subst| {
                    bindings.captures.iter().all(|(c, id)| {
                        subst
                            .get(variable(*c))
                            .is_some_and(|v| graph.find(*v) == graph.find(*id))
                    })
                }) {
                    continue;
                }
                match constraints(&self.0.constraints, &mut bindings, graph) {
                    Ok(true) => {}
                    Ok(false) | Err(EvalError::Unavailable(_)) => continue,
                    Err(e) => return Err(e),
                }
                let mut ctx = RuleContext::new(graph, root, bindings);
                match (self.0.prepare)(&mut ctx) {
                    Ok(Some(rhs)) => {
                        if rhs.facts().conflicts() {
                            return Err(EvalError::Conflict(
                                "conflicting replacement metadata".into(),
                            ));
                        }
                        if !rhs.facts().disagrees(&graph[root].data) {
                            prepared.push(rhs);
                        }
                    }
                    Ok(None) | Err(EvalError::Unavailable(_)) => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(prepared)
        };
        let prepared = match prepare() {
            Ok(xs) => xs,
            Err(e) => {
                self.report(graph, e);
                return vec![];
            }
        };
        let mut added = vec![];
        for rhs in prepared {
            // An earlier candidate may have refined root metadata.
            if rhs.facts().disagrees(&graph[root].data) {
                continue;
            }
            let id = rhs.insert(graph);
            if graph.analysis.check().is_err() {
                break;
            }
            if graph.union_trusted(root, id, name) {
                added.push(id);
            }
            if graph.analysis.check().is_err() {
                break;
            }
        }
        added
    }
}
impl<P: Program> Applier<Node<P>, TensorAnalysis<P>> for Adapter<P> {
    fn apply_one(
        &self,
        graph: &mut EGraph<Node<P>, TensorAnalysis<P>>,
        eclass: Id,
        subst: &Subst,
        _: Option<&PatternAst<Node<P>>>,
        name: Symbol,
    ) -> Vec<Id> {
        self.apply_root(graph, eclass, std::slice::from_ref(subst), name)
    }
    fn apply_matches(
        &self,
        graph: &mut EGraph<Node<P>, TensorAnalysis<P>>,
        matches: &[SearchMatches<Node<P>>],
        name: Symbol,
    ) -> Vec<Id> {
        let mut out = vec![];
        for m in matches {
            out.extend(self.apply_root(graph, m.eclass, &m.substs, name));
            if graph.analysis.check().is_err() {
                break;
            }
        }
        out
    }
}
