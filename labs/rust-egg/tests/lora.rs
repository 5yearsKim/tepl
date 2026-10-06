//! Exercises the reusable LoRA rule against test tensor metadata and host functions.

use rust_egg::host::nodes::*;
use rust_egg::host::{DemoLoraFunctions, dot_attrs};
use rust_egg::ir::dialects::tensor_lang;
use std::sync::{Arc, Mutex};

use egg::{EGraph, Id, Rewrite, Var};
use rust_egg::host::LoraAnalysis;
use rust_egg::ir::analysis::TensorBindingTable;
use rust_egg::ir::pattern::{AttrVar, TensorInfo, matches_at};
use rust_egg::ir::rules::lora::rule_lora;
use rust_egg::ir::rules::lora::rule_lora::Functions;
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

fn dot(lhs: Id, rhs: Id) -> OpNode {
    OpNode::from_parts(
        Op::TensorLang(tensor_lang::Op::DotGeneral),
        vec![lhs, rhs],
        dot_attrs(),
    )
    .unwrap()
}

struct Fixture {
    egraph: EGraph<OpNode, LoraAnalysis>,
    root: Id,
    inputs: [Id; 4],
}

fn fixture(b_output: u64) -> Fixture {
    // X: [2, 3, 4], W: [2, 4, 5], A: [2, 4, 2], B: [2, 2, b_output].
    let shapes = [
        ("X", vec![2, 3, 4]),
        ("W", vec![2, 4, 5]),
        ("A", vec![2, 4, 2]),
        ("B", vec![2, 2, b_output]),
    ];
    let mut bindings = TensorBindingTable::default();
    for (name, shape) in shapes {
        bindings
            .register_symbol(
                name,
                TensorInfo {
                    shape,
                    dtype: DType::F32,
                },
            )
            .unwrap();
    }
    let mut egraph = EGraph::new(LoraAnalysis::new(bindings));
    let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| egraph.add(symbol(name)));
    let ab = egraph.add(dot(a, b));
    let weights = egraph.add(binary(tensor_lang::Op::Add, w, ab).unwrap());
    let root = egraph.add(dot(x, weights));
    egraph.rebuild();
    Fixture {
        egraph,
        root,
        inputs: [x, w, a, b],
    }
}

fn test_rule() -> Rewrite<OpNode, LoraAnalysis> {
    rule_lora::build_rewrite(DemoLoraFunctions {
        allow_reassociation: true,
    })
    .unwrap()
}

#[test]
fn lora_rule_matches_and_builds_rhs() {
    let Fixture {
        mut egraph,
        root,
        inputs: [x, w, a, b],
    } = fixture(5);
    let lhs = rule_lora::pattern();
    let rule = test_rule();

    let captures = matches_at(&egraph, root, &lhs);
    assert_eq!(captures.len(), 1);
    assert_eq!(captures[0].attrs[&AttrVar::from("d0")], dot_attrs());
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
        ..
    } = fixture(6);
    let rule = test_rule();
    let before = egraph.total_size();
    let found = rule.search(&egraph);
    assert!(found.is_empty());
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
        fn is_broadcastable(&self, _: &[u64], _: &[u64]) -> Option<bool> {
            Some(true)
        }
        fn is_reassociable(
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
            assert_eq!(*source, dot_attrs());
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
            assert_eq!(*outer, dot_attrs());
            assert_eq!(*inner, dot_attrs());
            self.calls.lock().unwrap().push("out");
            Some(inner.clone())
        }
    }
    let Fixture { mut egraph, .. } = fixture(5);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let rule = rule_lora::build_rewrite(TracingFunctions {
        calls: calls.clone(),
    })
    .unwrap();
    let found = rule.search(&egraph);
    assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
    assert_eq!(*calls.lock().unwrap(), ["xw", "xa", "out"]);
}

// Exercise the example host functions with a non-default tensor dtype.

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
    let functions: &dyn Functions = &DemoLoraFunctions {
        allow_reassociation: true,
    };
    assert_eq!(functions.is_broadcastable(&[2], &[2]), Some(true));
    assert!(functions.infer_dot(&tensor, &a, &dot_attrs()).is_some());
    assert!(
        functions
            .infer_lora_out(&tensor, &a, &b, &dot_attrs(), &dot_attrs())
            .is_some()
    );
    assert_eq!(
        functions.is_reassociable(&tensor, &a, &b, &dot_attrs(), &dot_attrs()),
        Some(true)
    );
}
