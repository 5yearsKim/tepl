//! Exercises the reusable LoRA rule against test tensor metadata and host functions.

use std::collections::HashMap;

use egg::{EGraph, Id, Rewrite, Var};
use rust_egg::ir::patterns::{AttrVar, InferredTensor, TensorInfo, matches_at};
use rust_egg::ir::rules::lora::{self, HostFunctions};
use rust_egg::ir::{OpAttrs, OpKind, TensorLang};

fn batched_dot_attrs() -> OpAttrs {
    OpAttrs::DotGeneral {
        lhs_contracting: vec![2],
        rhs_contracting: vec![1],
        lhs_batch: vec![0],
        rhs_batch: vec![0],
    }
}

fn batched_dot_shape(lhs: &[usize], rhs: &[usize], attrs: &OpAttrs) -> Option<Vec<usize>> {
    if lhs.len() != 3
        || rhs.len() != 3
        || *attrs != batched_dot_attrs()
        || lhs[0] != rhs[0]
        || lhs[2] != rhs[1]
    {
        return None;
    }
    Some(vec![lhs[0], lhs[1], rhs[2]])
}

struct TestFunctions;

impl HostFunctions for TestFunctions {
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
        // This test models exact arithmetic. A real implementation must check
        // its numerical semantics before permitting reassociation.
        Some(*outer == batched_dot_attrs() && *inner == batched_dot_attrs())
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

fn dot(lhs: Id, rhs: Id) -> TensorLang {
    TensorLang::new(OpKind::DotGeneral, vec![lhs, rhs], batched_dot_attrs()).unwrap()
}

struct Fixture {
    egraph: EGraph<TensorLang, ()>,
    root: Id,
    inputs: [Id; 4],
    shapes: HashMap<String, Vec<usize>>,
}

fn fixture(b_output: usize) -> Fixture {
    // X: [2, 3, 4], W: [2, 4, 5], A: [2, 4, 2], B: [2, 2, b_output].
    let shapes = HashMap::from([
        (String::from("X"), vec![2, 3, 4]),
        (String::from("W"), vec![2, 4, 5]),
        (String::from("A"), vec![2, 4, 2]),
        (String::from("B"), vec![2, 2, b_output]),
    ]);
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| egraph.add(TensorLang::symbol(name)));
    let ab = egraph.add(dot(a, b));
    let weights = egraph.add(TensorLang::binary(OpKind::Add, w, ab).unwrap());
    let root = egraph.add(dot(x, weights));
    egraph.rebuild();
    Fixture {
        egraph,
        root,
        inputs: [x, w, a, b],
        shapes,
    }
}

fn test_rule(shapes: HashMap<String, Vec<usize>>) -> Rewrite<TensorLang, ()> {
    let metadata = move |egraph: &EGraph<TensorLang, ()>, id: Id| {
        egraph[egraph.find(id)]
            .nodes
            .iter()
            .find_map(TensorLang::symbol_name)
            .and_then(|name| shapes.get(name))
            .map(|shape| TensorInfo {
                shape: shape.clone(),
            })
    };
    lora::rule_lora(metadata, TestFunctions).unwrap()
}

#[test]
fn lora_rule_matches_and_builds_rhs() {
    let Fixture {
        mut egraph,
        root,
        inputs: [x, w, a, b],
        shapes,
    } = fixture(5);
    let lhs = lora::pattern();
    let rule = test_rule(shapes);

    let captures = matches_at(&egraph, root, &lhs);
    assert_eq!(captures.len(), 1);
    assert_eq!(
        captures[0].attrs[&AttrVar::from("outer")],
        batched_dot_attrs()
    );
    assert_eq!(captures[0].tensors["?X".parse::<Var>().unwrap()], x);

    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    egraph.rebuild();

    let xw = egraph.lookup(dot(x, w)).expect("X @ W was inserted");
    let xa = egraph.lookup(dot(x, a)).expect("X @ A was inserted");
    let out = egraph.lookup(dot(xa, b)).expect("(X @ A) @ B was inserted");
    let rhs = egraph
        .lookup(TensorLang::binary(OpKind::Add, xw, out).unwrap())
        .expect("the RHS addition was inserted");
    assert_eq!(egraph.find(root), egraph.find(rhs));
}

#[test]
fn lora_rule_rejects_incompatible_shapes() {
    let Fixture {
        mut egraph,
        inputs: [x, w, _, _],
        shapes,
        ..
    } = fixture(6);
    let rule = test_rule(shapes);
    let before = egraph.total_size();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert!(rule.apply(&mut egraph, &found).is_empty());
    assert_eq!(egraph.total_size(), before);
    assert!(egraph.lookup(dot(x, w)).is_none());
}
