use crate::{Budget, DType, EvalResult, Node, NodeKind, Program, TensorAnalysis, Value};
use egg::{EGraph, Id};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum Pattern {
    Capture(usize),
    Bind(usize, Box<Pattern>),
    Literal {
        value: String,
        dtype: Option<DType>,
    },
    Operation {
        operation: usize,
        descriptor: Option<usize>,
        operands: Vec<Pattern>,
    },
}
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    pub captures: BTreeMap<usize, Id>,
    pub descriptors: BTreeMap<usize, Value>,
    pub dimensions: BTreeMap<usize, Value>,
    pub dtypes: BTreeMap<usize, Value>,
    pub locals: BTreeMap<usize, Value>,
}
fn bind<T: PartialEq>(map: &mut BTreeMap<usize, T>, key: usize, value: T) -> bool {
    match map.get(&key) {
        Some(old) => *old == value,
        None => {
            map.insert(key, value);
            true
        }
    }
}

/// Enumerates concrete node witnesses, preserving tensor/attribute pairs.
pub fn matches<P: Program>(
    pattern: &Pattern,
    root: Id,
    graph: &EGraph<Node<P>, TensorAnalysis<P>>,
    budget: &mut Budget,
) -> EvalResult<Vec<Bindings>> {
    fn visit<P: Program>(
        pattern: &Pattern,
        id: Id,
        graph: &EGraph<Node<P>, TensorAnalysis<P>>,
        bindings: Bindings,
        budget: &mut Budget,
    ) -> EvalResult<Vec<Bindings>> {
        budget.tick()?;
        let id = graph.find(id);
        let mut bindings = bindings;
        match pattern {
            Pattern::Capture(c) => Ok(if bind(&mut bindings.captures, *c, id) {
                vec![bindings]
            } else {
                vec![]
            }),
            Pattern::Bind(c, inner) => {
                if bind(&mut bindings.captures, *c, id) {
                    visit(inner, id, graph, bindings, budget)
                } else {
                    Ok(vec![])
                }
            }
            Pattern::Literal { value, dtype } => {
                for node in &graph[id].nodes {
                    budget.tick()?;
                    if let NodeKind::Literal {
                        value: other,
                        dtype: actual,
                    } = node.kind()
                    {
                        if dtype.is_some() && dtype != actual {
                            continue;
                        }
                        let decimal = value.contains('.');
                        if decimal != other.contains('.') {
                            continue;
                        }
                        let equal = if decimal {
                            crate::values::floating(value)? == crate::values::floating(other)?
                        } else {
                            crate::values::integer(value)? == crate::values::integer(other)?
                        };
                        if equal {
                            return Ok(vec![bindings]);
                        }
                    }
                }
                Ok(vec![])
            }
            Pattern::Operation {
                operation,
                descriptor,
                operands,
            } => {
                let mut out = vec![];
                for node in &graph[id].nodes {
                    budget.tick()?;
                    if !matches!(node.kind(), NodeKind::Operation(op) if P::operation_id(*op) == *operation)
                        || operands.len() != node.operands().len()
                    {
                        continue;
                    }
                    let mut b = bindings.clone();
                    if let Some(d) = descriptor
                        && !bind(&mut b.descriptors, *d, node.attribute_value())
                    {
                        continue;
                    }
                    let mut candidates = vec![b];
                    for (pat, child) in operands.iter().zip(node.operands()) {
                        let mut next = vec![];
                        for candidate in candidates {
                            next.extend(visit(pat, *child, graph, candidate, budget)?);
                        }
                        budget.size(next.len())?;
                        candidates = next;
                        if candidates.is_empty() {
                            break;
                        }
                    }
                    budget.size(
                        out.len()
                            .checked_add(candidates.len())
                            .ok_or(crate::EvalError::Limit("match limit"))?,
                    )?;
                    out.extend(candidates);
                }
                Ok(out)
            }
        }
    }
    visit(pattern, root, graph, Bindings::default(), budget)
}

#[derive(Clone, Debug)]
pub enum ShapeElement {
    Dimension(Option<usize>),
    Sequence(Option<usize>),
}
#[derive(Clone, Debug)]
pub enum DTypeConstraint {
    Exact(DType),
    Variable(usize),
}
#[derive(Clone, Debug)]
pub struct Constraint {
    pub capture: usize,
    pub shape: Vec<ShapeElement>,
    pub dtype: Option<DTypeConstraint>,
}
pub fn constraints<P: Program>(
    constraints: &[Constraint],
    bindings: &mut Bindings,
    graph: &EGraph<Node<P>, TensorAnalysis<P>>,
) -> EvalResult<bool> {
    for c in constraints {
        let facts = &graph[bindings.captures[&c.capture]].data;
        let shape = facts.shape.get("shape")?;
        let fixed = c
            .shape
            .iter()
            .filter(|e| matches!(e, ShapeElement::Dimension(_)))
            .count();
        let sequence = c
            .shape
            .iter()
            .any(|e| matches!(e, ShapeElement::Sequence(_)));
        if if sequence {
            shape.len() < fixed
        } else {
            shape.len() != fixed
        } {
            return Ok(false);
        }
        let mut at = 0;
        for element in &c.shape {
            let (symbol, value) = match element {
                ShapeElement::Dimension(symbol) => {
                    let value = Value::Integer(shape[at].into());
                    at += 1;
                    (symbol, value)
                }
                ShapeElement::Sequence(symbol) => {
                    let n = shape.len() - fixed;
                    let value = crate::values::shape(&shape[at..at + n]);
                    at += n;
                    (symbol, value)
                }
            };
            if let Some(symbol) = symbol
                && !bind(&mut bindings.dimensions, *symbol, value)
            {
                return Ok(false);
            }
        }
        if let Some(c) = &c.dtype {
            let dtype = facts.dtype.get("dtype")?;
            match c {
                DTypeConstraint::Exact(expected) => {
                    if dtype != expected {
                        return Ok(false);
                    }
                }
                DTypeConstraint::Variable(id) => {
                    if !bind(
                        &mut bindings.dtypes,
                        *id,
                        Value::String(dtype.name().into()),
                    ) {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}
