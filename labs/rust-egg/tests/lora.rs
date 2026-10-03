//! Exercises the reusable LoRA rule against test tensor metadata and host functions.

use rust_egg::host::nodes::*;
use rust_egg::ir::dialects::tensor_lang;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use egg::{EGraph, Id, Rewrite, Var};
use rust_egg::ir::pattern::{AttrVar, TensorInfo, matches_at};
use rust_egg::ir::rules::lora::rule_lora;
use rust_egg::ir::rules::lora::rule_lora::Functions;
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

fn batched_dot_attrs() -> OpAttrs {
    OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting: vec![2],
        rhs_contracting: vec![1],
        lhs_batch: vec![0],
        rhs_batch: vec![0],
    })
}

fn batched_dot_shape(lhs: &[u64], rhs: &[u64], attrs: &OpAttrs) -> Option<Vec<u64>> {
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

impl Functions for TestFunctions {
    fn broadcastable(&self, batch: &[u64], weight_batch: &[u64]) -> Option<bool> {
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

    fn infer_dot(&self, lhs: &TensorInfo, rhs: &TensorInfo, attrs: &OpAttrs) -> Option<OpAttrs> {
        batched_dot_shape(&lhs.shape, &rhs.shape, attrs)?;
        Some(attrs.clone())
    }
    fn infer_lora_out(
        &self,
        x: &TensorInfo,
        a: &TensorInfo,
        b: &TensorInfo,
        outer: &OpAttrs,
        inner: &OpAttrs,
    ) -> Option<OpAttrs> {
        if x.shape.len() != 3
            || a.shape.len() != 3
            || b.shape.len() != 3
            || *outer != batched_dot_attrs()
            || *inner != batched_dot_attrs()
            || x.shape[0] != a.shape[0]
            || a.shape[0] != b.shape[0]
            || x.shape[2] != a.shape[1]
            || a.shape[2] != b.shape[1]
        {
            return None;
        }
        Some(inner.clone())
    }
}

fn dot(lhs: Id, rhs: Id) -> OpNode {
    OpNode::from_parts(
        Op::TensorLang(tensor_lang::Op::DotGeneral),
        vec![lhs, rhs],
        batched_dot_attrs(),
    )
    .unwrap()
}

struct Fixture {
    egraph: EGraph<OpNode, ()>,
    root: Id,
    inputs: [Id; 4],
    shapes: HashMap<String, Vec<u64>>,
}

fn fixture(b_output: u64) -> Fixture {
    // X: [2, 3, 4], W: [2, 4, 5], A: [2, 4, 2], B: [2, 2, b_output].
    let shapes = HashMap::from([
        (String::from("X"), vec![2, 3, 4]),
        (String::from("W"), vec![2, 4, 5]),
        (String::from("A"), vec![2, 4, 2]),
        (String::from("B"), vec![2, 2, b_output]),
    ]);
    let mut egraph = EGraph::<OpNode, ()>::default();
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| egraph.add(symbol(name)));
    let ab = egraph.add(dot(a, b));
    let weights = egraph.add(binary(tensor_lang::Op::Add, w, ab).unwrap());
    let root = egraph.add(dot(x, weights));
    egraph.rebuild();
    Fixture {
        egraph,
        root,
        inputs: [x, w, a, b],
        shapes,
    }
}

fn test_rule(shapes: HashMap<String, Vec<u64>>) -> Rewrite<OpNode, ()> {
    let metadata = move |graph: &EGraph<OpNode, ()>, id: Id| {
        rust_egg::host::infer_eclass(graph, id, &|node| {
            Some(TensorInfo {
                dtype: DType::F32,
                shape: shapes.get(symbol_name(node)?)?.clone(),
            })
        })
    };
    rule_lora::build_rewrite(metadata, rust_egg::host::infer_tensor_output, TestFunctions).unwrap()
}

#[test]
fn lora_rule_matches_and_builds_rhs() {
    let Fixture {
        mut egraph,
        root,
        inputs: [x, w, a, b],
        shapes,
    } = fixture(5);
    let lhs = rule_lora::pattern();
    let rule = test_rule(shapes);

    let captures = matches_at(&egraph, root, &lhs);
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].attrs[&AttrVar::from("d0")], batched_dot_attrs());
    assert_eq!(captures[0].tensors["?c0".parse::<Var>().unwrap()], x);

    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    egraph.rebuild();

    let xw = egraph.lookup(dot(x, w)).expect("X @ W was inserted");
    let xa = egraph.lookup(dot(x, a)).expect("X @ A was inserted");
    let out = egraph.lookup(dot(xa, b)).expect("(X @ A) @ B was inserted");
    let rhs = egraph
        .lookup(binary(tensor_lang::Op::Add, xw, out).unwrap())
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

#[test]
fn lora_derivations_use_only_lhs_captures_and_return_descriptors() {
    struct TracingFunctions {
        calls: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Functions for TracingFunctions {
        fn broadcastable(&self, _: &[u64], _: &[u64]) -> Option<bool> {
            Some(true)
        }
        fn reassociable(
            &self,
            _: &TensorInfo,
            _: &TensorInfo,
            _: &TensorInfo,
            _: &OpAttrs,
            _: &OpAttrs,
        ) -> Option<bool> {
            Some(true)
        }
        fn infer_dot(
            &self,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            source: &OpAttrs,
        ) -> Option<OpAttrs> {
            assert_eq!(lhs.shape, [2, 3, 4]);
            assert_eq!(*source, batched_dot_attrs());
            let name = if rhs.shape == [2, 4, 5] {
                "xw"
            } else {
                assert_eq!(rhs.shape, [2, 4, 2]);
                "xa"
            };
            self.calls.lock().unwrap().push(name);
            Some(source.clone())
        }
        fn infer_lora_out(
            &self,
            x: &TensorInfo,
            a: &TensorInfo,
            b: &TensorInfo,
            outer: &OpAttrs,
            inner: &OpAttrs,
        ) -> Option<OpAttrs> {
            assert_eq!(x.shape, [2, 3, 4]);
            assert_eq!(a.shape, [2, 4, 2]);
            assert_eq!(b.shape, [2, 2, 5]);
            assert_eq!(*outer, batched_dot_attrs());
            assert_eq!(*inner, batched_dot_attrs());
            self.calls.lock().unwrap().push("out");
            Some(inner.clone())
        }
    }
    let Fixture {
        mut egraph, shapes, ..
    } = fixture(5);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let metadata = move |graph: &EGraph<OpNode, ()>, id: Id| {
        rust_egg::host::infer_eclass(graph, id, &|node| {
            Some(TensorInfo {
                dtype: DType::F32,
                shape: shapes.get(symbol_name(node)?)?.clone(),
            })
        })
    };
    let rule = rule_lora::build_rewrite(
        metadata,
        rust_egg::host::infer_tensor_output,
        TracingFunctions {
            calls: calls.clone(),
        },
    )
    .unwrap();
    let found = rule.search(&egraph);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    assert_eq!(*calls.lock().unwrap(), ["xw", "xa", "out"]);
}

// Compile the generated interface against the same implementation used by the
// handwritten reference rules, catching frontend/runtime signature drift.

#[test]
fn generated_host_interface_accepts_dtype_metadata() {
    let tensor = TensorInfo {
        shape: vec![2, 3, 4],
        dtype: DType::BF16,
    };
    let a = TensorInfo {
        shape: vec![2, 4, 2],
        dtype: DType::BF16,
    };
    let b = TensorInfo {
        shape: vec![2, 2, 5],
        dtype: DType::BF16,
    };
    let functions: &dyn Functions = &TestFunctions;
    assert_eq!(functions.broadcastable(&[2], &[2]), Some(true));
    assert!(
        functions
            .infer_dot(&tensor, &a, &batched_dot_attrs())
            .is_some()
    );
    assert!(
        functions
            .infer_lora_out(&tensor, &a, &b, &batched_dot_attrs(), &batched_dot_attrs())
            .is_some()
    );
    assert_eq!(
        functions.reassociable(&tensor, &a, &b, &batched_dot_attrs(), &batched_dot_attrs()),
        Some(true)
    );
}
