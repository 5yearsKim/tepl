//! Express the LoRA rewrite using only the reusable tensor pattern API.

use std::collections::HashMap;

use egg::{EGraph, Id, Rewrite, Symbol, Var};
use rust_egg::ir::{OpAttrs, OpKind, TensorLang};
use rust_egg::tensor_pattern::{
    AttrExpr, AttrPattern, AttrVar, InferredTensor, MatchContext, TensorExpr, TensorInfo,
    TensorPattern, matches_at, tensor_rewrite,
};

#[path = "support/generated_host.rs"]
mod generated_host;
use generated_host::HostFunctions;

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
    shapes: HashMap<Symbol, Vec<usize>>,
}

fn fixture(b_output: usize) -> Fixture {
    // X: [2, 3, 4], W: [2, 4, 5], A: [2, 4, 2], B: [2, 2, b_output].
    let shapes = HashMap::from([
        (Symbol::from("X"), vec![2, 3, 4]),
        (Symbol::from("W"), vec![2, 4, 5]),
        (Symbol::from("A"), vec![2, 4, 2]),
        (Symbol::from("B"), vec![2, 2, b_output]),
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

fn lora_pattern(shapes: HashMap<Symbol, Vec<usize>>) -> (TensorPattern, Rewrite<TensorLang, ()>) {
    let [x, w, a, b] = ["?x", "?w", "?a", "?b"].map(|name| name.parse::<Var>().unwrap());
    let [outer, inner, xw, xa, out] = ["outer", "inner", "xw", "xa", "out"].map(AttrVar::from);

    // dot[@outer](X, add(W, dot[@inner](A, B)))
    let lhs = TensorPattern::op(
        OpKind::DotGeneral,
        AttrPattern::Bind(outer),
        vec![
            TensorPattern::Var(x),
            TensorPattern::op(
                OpKind::Add,
                AttrPattern::Exact(OpAttrs::None),
                vec![
                    TensorPattern::Var(w),
                    TensorPattern::op(
                        OpKind::DotGeneral,
                        AttrPattern::Bind(inner),
                        vec![TensorPattern::Var(a), TensorPattern::Var(b)],
                    ),
                ],
            ),
        ],
    );

    // add(dot[@xw](X, W), dot[@out](dot[@xa](X, A), B))
    let rhs = TensorExpr::op(
        OpKind::Add,
        AttrExpr::Exact(OpAttrs::None),
        vec![
            TensorExpr::op(
                OpKind::DotGeneral,
                AttrExpr::Derived(xw),
                vec![TensorExpr::Var(x), TensorExpr::Var(w)],
            ),
            TensorExpr::op(
                OpKind::DotGeneral,
                AttrExpr::Derived(out),
                vec![
                    TensorExpr::op(
                        OpKind::DotGeneral,
                        AttrExpr::Derived(xa),
                        vec![TensorExpr::Var(x), TensorExpr::Var(a)],
                    ),
                    TensorExpr::Var(b),
                ],
            ),
        ],
    );

    let metadata = move |egraph: &EGraph<TensorLang, ()>, id: Id| {
        egraph[egraph.find(id)]
            .nodes
            .iter()
            .find_map(TensorLang::symbol_name)
            .and_then(|name| shapes.get(&name))
            .map(|shape| TensorInfo {
                shape: shape.clone(),
            })
    };
    let host = TestFunctions;
    let rule = tensor_rewrite("lora-pattern", lhs.clone(), rhs, move |egraph, matched| {
        let ctx = MatchContext::new(egraph, matched, &metadata);
        check_and_derive(&ctx, &host, [x, w, a, b], [outer, inner, xw, xa, out])
    })
    .unwrap();
    (lhs, rule)
}

// This function is the shape of a generated callback for the LoRA DSL rule.
fn check_and_derive<M: rust_egg::tensor_pattern::TensorMetadata<()>>(
    ctx: &MatchContext<'_, (), M>,
    host: &impl HostFunctions,
    inputs: [Var; 4],
    attrs: [AttrVar; 5],
) -> Option<HashMap<AttrVar, OpAttrs>> {
    let [x, w, a, b] = inputs;
    let [outer, inner, xw_attr, xa_attr, out_attr] = attrs;
    let (x, w, a, b) = (
        ctx.tensor(x)?,
        ctx.tensor(w)?,
        ctx.tensor(a)?,
        ctx.tensor(b)?,
    );
    let (outer, inner) = (ctx.attrs(outer)?, ctx.attrs(inner)?);

    if !host.broadcastable(x.shape.get(..1)?, w.shape.get(..1)?)?
        || !host.reassociable(&x, &a, &b, outer, inner)?
    {
        return None;
    }
    if host.infer_dot(&a, &b, inner)?.output != w {
        return None;
    }

    let xw = host.infer_dot(&x, &w, outer)?;
    let xa = host.infer_dot(&x, &a, outer)?;
    // ?XA is described by xa.output before any RHS node is inserted.
    let out = host.infer_dot(&xa.output, &b, inner)?;
    if xw.output != out.output {
        return None;
    }

    Some(HashMap::from([
        (xw_attr, xw.attrs),
        (xa_attr, xa.attrs),
        (out_attr, out.attrs),
    ]))
}

#[test]
fn lora_rule_matches_and_builds_rhs() {
    let Fixture {
        mut egraph,
        root,
        inputs: [x, w, a, b],
        shapes,
    } = fixture(5);
    let (lhs, rule) = lora_pattern(shapes);

    let captures = matches_at(&egraph, root, &lhs);
    assert_eq!(captures.len(), 1);
    assert_eq!(
        captures[0].attrs[&AttrVar::from("outer")],
        batched_dot_attrs()
    );
    assert_eq!(captures[0].tensors["?x".parse::<Var>().unwrap()], x);

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
    let (_, rule) = lora_pattern(shapes);
    let before = egraph.total_size();
    let found = rule.search(&egraph);
    assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
    assert!(rule.apply(&mut egraph, &found).is_empty());
    assert_eq!(egraph.total_size(), before);
    assert!(egraph.lookup(dot(x, w)).is_none());
}
