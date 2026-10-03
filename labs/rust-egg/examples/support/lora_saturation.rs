//! Shared shape, cost, and evaluation support for the LoRA example and tests.

use egg::{CostFunction, EGraph, Id, Language, RecExpr, StopReason};
use rust_egg::host::nodes::*;
pub use rust_egg::host::{
    DemoLoraFunctions, ShapeAnalysis, TensorBindings, batched_dot_shape, dot_attrs,
    infer_tensor_output, tensor_info,
};
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::pattern::TensorInfo;
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};
use std::collections::HashMap;

pub type Shapes = HashMap<String, Vec<u64>>;

pub fn example_shapes() -> Shapes {
    HashMap::from([
        ("X".into(), vec![2, 4, 64]),
        ("W".into(), vec![2, 64, 32]),
        ("A".into(), vec![2, 64, 4]),
        ("B".into(), vec![2, 4, 32]),
    ])
}

pub fn bindings_from_shapes(shapes: Shapes) -> TensorBindings {
    let mut bindings = TensorBindings::default();
    for (name, shape) in shapes {
        bindings
            .register_symbol(
                name,
                TensorInfo {
                    shape,
                    dtype: DType::I64,
                },
            )
            .unwrap();
    }
    bindings
}

fn dot(expr: &mut RecExpr<OpNode>, lhs: Id, rhs: Id, attrs: OpAttrs) -> Id {
    expr.add(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::DotGeneral),
            vec![lhs, rhs],
            attrs,
        )
        .unwrap(),
    )
}

pub fn original_expr(swapped_add: bool, inner_attrs: OpAttrs) -> RecExpr<OpNode> {
    let mut expr = RecExpr::default();
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| expr.add(symbol(name)));
    let ab = dot(&mut expr, a, b, inner_attrs);
    let (left, right) = if swapped_add { (ab, w) } else { (w, ab) };
    let sum = expr.add(binary(tensor_lang::Op::Add, left, right).unwrap());
    dot(&mut expr, x, sum, dot_attrs());
    expr
}

pub fn expected_expr() -> RecExpr<OpNode> {
    let mut expr = RecExpr::default();
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| expr.add(symbol(name)));
    let xw = dot(&mut expr, x, w, dot_attrs());
    let xa = dot(&mut expr, x, a, dot_attrs());
    let xab = dot(&mut expr, xa, b, dot_attrs());
    expr.add(binary(tensor_lang::Op::Add, xw, xab).unwrap());
    expr
}

pub fn input_graph(
    shapes: Shapes,
    swapped_add: bool,
    inner_attrs: OpAttrs,
) -> (EGraph<OpNode, ShapeAnalysis>, Id, RecExpr<OpNode>) {
    let expr = original_expr(swapped_add, inner_attrs);
    let mut egraph = EGraph::new(ShapeAnalysis::new(bindings_from_shapes(shapes)));
    let root = egraph.add_expr(&expr);
    egraph.rebuild();
    (egraph, root, expr)
}

pub struct ArithmeticCost<'a> {
    pub egraph: &'a EGraph<OpNode, ShapeAnalysis>,
}

impl CostFunction<OpNode> for ArithmeticCost<'_> {
    type Cost = u64;

    fn cost<C>(&mut self, node: &OpNode, mut costs: C) -> u64
    where
        C: FnMut(Id) -> u64,
    {
        let output = |id: Id| {
            self.egraph[self.egraph.find(id)]
                .data
                .info()
                .map(|info| info.shape.as_slice())
        };
        let local = match (node.op(), node.children()) {
            (Op::Input, []) => 0,
            (Op::TensorLang(tensor_lang::Op::Add), [lhs, _]) => output(*lhs)
                .map(|shape| {
                    shape
                        .iter()
                        .fold(1_u64, |size, &dim| size.saturating_mul(dim))
                })
                .unwrap_or(u64::MAX / 4),
            (Op::TensorLang(tensor_lang::Op::DotGeneral), [lhs, rhs]) => {
                match (output(*lhs), output(*rhs)) {
                    (Some(lhs_shape @ [batch, m, k]), Some(rhs_shape @ [_, _, n]))
                        if batched_dot_shape(lhs_shape, rhs_shape, node.attrs()).is_some() =>
                    {
                        2_u64
                            .saturating_mul(*batch)
                            .saturating_mul(*m)
                            .saturating_mul(*k)
                            .saturating_mul(*n)
                    }
                    _ => u64::MAX / 4,
                }
            }
            _ => u64::MAX / 4,
        };
        node.children()
            .iter()
            .fold(local, |sum, id| sum.saturating_add(costs(*id)))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorValue {
    pub shape: Vec<u64>,
    pub data: Vec<i64>,
}

pub fn example_values(shapes: &Shapes) -> HashMap<String, TensorValue> {
    ["X", "W", "A", "B"]
        .into_iter()
        .enumerate()
        .map(|(seed, name)| {
            let shape = shapes[name].clone();
            let size = shape
                .iter()
                .try_fold(1_usize, |size, &dim| {
                    size.checked_mul(usize::try_from(dim).ok()?)
                })
                .expect("demo tensor fits memory indexing");
            let data = (0..size)
                .map(|index| ((index * 17 + seed * 11) % 7) as i64 - 3)
                .collect();
            (name.into(), TensorValue { shape, data })
        })
        .collect()
}

pub fn evaluate(
    expr: &RecExpr<OpNode>,
    inputs: &HashMap<String, TensorValue>,
) -> Result<TensorValue, String> {
    let mut values: Vec<TensorValue> = Vec::new();
    for node in expr.as_ref() {
        let value = match (node.op(), node.children()) {
            (Op::Input, []) => inputs
                .get(symbol_name(&node).unwrap())
                .cloned()
                .ok_or_else(|| format!("missing tensor {}", symbol_name(&node).unwrap()))?,
            (Op::TensorLang(tensor_lang::Op::Add), [lhs, rhs]) => {
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
            (Op::TensorLang(tensor_lang::Op::DotGeneral), [lhs, rhs]) => {
                let lhs = &values[usize::from(*lhs)];
                let rhs = &values[usize::from(*rhs)];
                let shape = batched_dot_shape(&lhs.shape, &rhs.shape, node.attrs())
                    .ok_or("dot shape or attribute mismatch")?;
                let [batch, m, n] = shape.as_slice() else {
                    unreachable!()
                };
                let [batch, m, n, k] = [*batch, *m, *n, lhs.shape[2]].map(usize::try_from);
                let (batch, m, n, k) = (
                    batch.map_err(|_| "batch overflow")?,
                    m.map_err(|_| "dimension overflow")?,
                    n.map_err(|_| "dimension overflow")?,
                    k.map_err(|_| "dimension overflow")?,
                );
                let size = batch
                    .checked_mul(m)
                    .and_then(|x| x.checked_mul(n))
                    .ok_or("tensor size overflow")?;
                let mut data = vec![0; size];
                for b in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
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

pub fn text_dump(egraph: &EGraph<OpNode, ShapeAnalysis>) -> String {
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
