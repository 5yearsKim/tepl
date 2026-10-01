//! Shared shape, cost, and evaluation support for the LoRA example and tests.

use egg::{Analysis, CostFunction, DidMerge, EGraph, Id, Language, RecExpr, StopReason};
use rust_egg::ir::patterns::{InferredTensor, TensorInfo};
use rust_egg::ir::rules::lora;
use rust_egg::ir::{OpAttrs, OpKind, TensorLang};
use std::collections::HashMap;

pub type Shapes = HashMap<String, Vec<usize>>;

pub fn example_shapes() -> Shapes {
    HashMap::from([
        ("X".into(), vec![2, 4, 64]),
        ("W".into(), vec![2, 64, 32]),
        ("A".into(), vec![2, 64, 4]),
        ("B".into(), vec![2, 4, 32]),
    ])
}

pub fn dot_attrs() -> OpAttrs {
    OpAttrs::DotGeneral {
        lhs_contracting: vec![2],
        rhs_contracting: vec![1],
        lhs_batch: vec![0],
        rhs_batch: vec![0],
    }
}

pub fn batched_dot_shape(lhs: &[usize], rhs: &[usize], attrs: &OpAttrs) -> Option<Vec<usize>> {
    if lhs.len() != 3
        || rhs.len() != 3
        || *attrs != dot_attrs()
        || lhs[0] != rhs[0]
        || lhs[2] != rhs[1]
    {
        return None;
    }
    Some(vec![lhs[0], lhs[1], rhs[2]])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    Unknown,
    Known(Vec<usize>),
    Invalid,
}

#[derive(Default)]
pub struct ShapeAnalysis {
    pub symbols: Shapes,
}

impl Analysis<TensorLang> for ShapeAnalysis {
    type Data = Shape;

    fn make(egraph: &mut EGraph<TensorLang, Self>, node: &TensorLang, _id: Id) -> Shape {
        if let Some(name) = node.symbol_name() {
            return egraph
                .analysis
                .symbols
                .get(name)
                .cloned()
                .map(Shape::Known)
                .unwrap_or(Shape::Unknown);
        }

        let children: Vec<_> = node
            .children()
            .iter()
            .map(|id| &egraph[egraph.find(*id)].data)
            .collect();
        if children.iter().any(|shape| **shape == Shape::Invalid) {
            return Shape::Invalid;
        }
        if children.iter().any(|shape| **shape == Shape::Unknown) {
            return Shape::Unknown;
        }
        let known: Vec<&[usize]> = children
            .iter()
            .map(|shape| match shape {
                Shape::Known(dims) => dims.as_slice(),
                _ => unreachable!(),
            })
            .collect();
        let result = match (node.op(), known.as_slice()) {
            (OpKind::Add, [lhs, rhs]) if lhs == rhs => Some(lhs.to_vec()),
            (OpKind::DotGeneral, [lhs, rhs]) => batched_dot_shape(lhs, rhs, node.attrs()),
            _ => None,
        };
        result.map(Shape::Known).unwrap_or(Shape::Invalid)
    }

    fn merge(&mut self, target: &mut Shape, incoming: Shape) -> DidMerge {
        let merged = match (&*target, &incoming) {
            (Shape::Invalid, _) | (_, Shape::Invalid) => Shape::Invalid,
            // A known alternative cannot establish the shape of an unknown one.
            (Shape::Unknown, _) | (_, Shape::Unknown) => Shape::Unknown,
            (Shape::Known(a), Shape::Known(b)) if a == b => Shape::Known(a.clone()),
            (Shape::Known(_), Shape::Known(_)) => Shape::Invalid,
        };
        let changed_target = *target != merged;
        let changed_incoming = incoming != merged;
        *target = merged;
        DidMerge(changed_target, changed_incoming)
    }
}

pub fn tensor_info(egraph: &EGraph<TensorLang, ShapeAnalysis>, id: Id) -> Option<TensorInfo> {
    match &egraph[egraph.find(id)].data {
        Shape::Known(shape) => Some(TensorInfo {
            shape: shape.clone(),
        }),
        _ => None,
    }
}

pub struct DemoLoraFunctions {
    pub allow_reassociation: bool,
}

impl lora::lora::Functions for DemoLoraFunctions {
    fn broadcastable(&self, batch: &[usize], weight_batch: &[usize]) -> Option<bool> {
        Some(batch == weight_batch)
    }

    fn reassociable(
        &self,
        _x: &TensorInfo,
        _a: &TensorInfo,
        _b: &TensorInfo,
        outer: &OpAttrs,
        inner: &OpAttrs,
    ) -> Option<bool> {
        Some(self.allow_reassociation && *outer == dot_attrs() && *inner == dot_attrs())
    }

    fn infer_dot(
        &self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        attrs: &OpAttrs,
    ) -> Option<InferredTensor> {
        Some(InferredTensor {
            attrs: attrs.clone(),
            output: TensorInfo {
                shape: batched_dot_shape(&lhs.shape, &rhs.shape, attrs)?,
            },
        })
    }
}

fn dot(expr: &mut RecExpr<TensorLang>, lhs: Id, rhs: Id, attrs: OpAttrs) -> Id {
    expr.add(TensorLang::new(OpKind::DotGeneral, vec![lhs, rhs], attrs).unwrap())
}

pub fn original_expr(swapped_add: bool, inner_attrs: OpAttrs) -> RecExpr<TensorLang> {
    let mut expr = RecExpr::default();
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| expr.add(TensorLang::symbol(name)));
    let ab = dot(&mut expr, a, b, inner_attrs);
    let (left, right) = if swapped_add { (ab, w) } else { (w, ab) };
    let sum = expr.add(TensorLang::binary(OpKind::Add, left, right).unwrap());
    dot(&mut expr, x, sum, dot_attrs());
    expr
}

pub fn expected_expr() -> RecExpr<TensorLang> {
    let mut expr = RecExpr::default();
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| expr.add(TensorLang::symbol(name)));
    let xw = dot(&mut expr, x, w, dot_attrs());
    let xa = dot(&mut expr, x, a, dot_attrs());
    let xab = dot(&mut expr, xa, b, dot_attrs());
    expr.add(TensorLang::binary(OpKind::Add, xw, xab).unwrap());
    expr
}

pub fn input_graph(
    shapes: Shapes,
    swapped_add: bool,
    inner_attrs: OpAttrs,
) -> (EGraph<TensorLang, ShapeAnalysis>, Id, RecExpr<TensorLang>) {
    let expr = original_expr(swapped_add, inner_attrs);
    let mut egraph = EGraph::new(ShapeAnalysis { symbols: shapes });
    let root = egraph.add_expr(&expr);
    egraph.rebuild();
    (egraph, root, expr)
}

pub struct ArithmeticCost<'a> {
    pub egraph: &'a EGraph<TensorLang, ShapeAnalysis>,
}

impl CostFunction<TensorLang> for ArithmeticCost<'_> {
    type Cost = u64;

    fn cost<C>(&mut self, node: &TensorLang, mut costs: C) -> u64
    where
        C: FnMut(Id) -> u64,
    {
        let output = |id: Id| match &self.egraph[self.egraph.find(id)].data {
            Shape::Known(shape) => Some(shape.as_slice()),
            _ => None,
        };
        let local = match (node.op(), node.children()) {
            (OpKind::Symbol, []) => 0,
            (OpKind::Add, [lhs, _]) => output(*lhs)
                .map(|shape| {
                    shape
                        .iter()
                        .fold(1_u64, |size, &dim| size.saturating_mul(dim as u64))
                })
                .unwrap_or(u64::MAX / 4),
            (OpKind::DotGeneral, [lhs, rhs]) => match (output(*lhs), output(*rhs)) {
                (Some(lhs_shape @ [batch, m, k]), Some(rhs_shape @ [_, _, n]))
                    if batched_dot_shape(lhs_shape, rhs_shape, node.attrs()).is_some() =>
                {
                    2_u64
                        .saturating_mul(*batch as u64)
                        .saturating_mul(*m as u64)
                        .saturating_mul(*k as u64)
                        .saturating_mul(*n as u64)
                }
                _ => u64::MAX / 4,
            },
            _ => u64::MAX / 4,
        };
        node.children()
            .iter()
            .fold(local, |sum, id| sum.saturating_add(costs(*id)))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorValue {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

pub fn example_values(shapes: &Shapes) -> HashMap<String, TensorValue> {
    ["X", "W", "A", "B"]
        .into_iter()
        .enumerate()
        .map(|(seed, name)| {
            let shape = shapes[name].clone();
            let size: usize = shape.iter().product();
            let data = (0..size)
                .map(|index| ((index * 17 + seed * 11) % 7) as i64 - 3)
                .collect();
            (name.into(), TensorValue { shape, data })
        })
        .collect()
}

pub fn evaluate(
    expr: &RecExpr<TensorLang>,
    inputs: &HashMap<String, TensorValue>,
) -> Result<TensorValue, String> {
    let mut values: Vec<TensorValue> = Vec::new();
    for node in expr.as_ref() {
        let value = match (node.op(), node.children()) {
            (OpKind::Symbol, []) => inputs
                .get(node.symbol_name().unwrap())
                .cloned()
                .ok_or_else(|| format!("missing tensor {}", node.symbol_name().unwrap()))?,
            (OpKind::Add, [lhs, rhs]) => {
                let lhs = &values[usize::from(*lhs)];
                let rhs = &values[usize::from(*rhs)];
                if lhs.shape != rhs.shape {
                    return Err("add shape mismatch".into());
                }
                TensorValue {
                    shape: lhs.shape.clone(),
                    data: lhs.data.iter().zip(&rhs.data).map(|(a, b)| a + b).collect(),
                }
            }
            (OpKind::DotGeneral, [lhs, rhs]) => {
                let lhs = &values[usize::from(*lhs)];
                let rhs = &values[usize::from(*rhs)];
                let shape = batched_dot_shape(&lhs.shape, &rhs.shape, node.attrs())
                    .ok_or("dot shape or attribute mismatch")?;
                let [batch, m, n] = shape.as_slice() else {
                    unreachable!()
                };
                let k = lhs.shape[2];
                let mut data = vec![0; batch * m * n];
                for b in 0..*batch {
                    for i in 0..*m {
                        for j in 0..*n {
                            for p in 0..k {
                                data[(b * m + i) * n + j] +=
                                    lhs.data[(b * m + i) * k + p] * rhs.data[(b * k + p) * n + j];
                            }
                        }
                    }
                }
                TensorValue { shape, data }
            }
            _ => return Err(format!("unsupported node: {node}")),
        };
        values.push(value);
    }
    values.pop().ok_or_else(|| "empty expression".into())
}

pub fn text_dump(egraph: &EGraph<TensorLang, ShapeAnalysis>) -> String {
    let mut classes: Vec<_> = egraph.classes().collect();
    classes.sort_by_key(|class| usize::from(class.id));
    let mut result = String::new();
    for class in classes {
        result.push_str(&format!("eclass {} {:?}\n", class.id, class.data));
        for node in &class.nodes {
            result.push_str(&format!("  {node} {:?}\n", node.children()));
        }
    }
    result
}

pub fn stopped_by_saturation(reason: &Option<StopReason>) -> bool {
    matches!(reason, Some(StopReason::Saturated))
}
