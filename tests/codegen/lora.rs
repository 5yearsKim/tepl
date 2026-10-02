use egg::{EGraph, Id};
use std::collections::HashMap;
use tepl_generated::ir::dialects::tensor_lang;
use tepl_generated::ir::pattern::{OutputInference, TensorInfo};
use tepl_generated::ir::rules::lora::rule_lora;
use tepl_generated::ir::{DType, Op, OpAttrs, OpNode};

fn attrs() -> OpAttrs {
    OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        tepl_lhs_contracting: vec![2],
        tepl_rhs_contracting: vec![1],
        tepl_lhs_batch: vec![0],
        tepl_rhs_batch: vec![0],
    })
}
fn dot_shape(lhs: &[u64], rhs: &[u64], attributes: &OpAttrs) -> Option<Vec<u64>> {
    if lhs.len() != 3
        || rhs.len() != 3
        || *attributes != attrs()
        || lhs[0] != rhs[0]
        || lhs[2] != rhs[1]
    {
        return None;
    }
    Some(vec![lhs[0], lhs[1], rhs[2]])
}
struct Host {
    allowed: bool,
}
impl rule_lora::Functions for Host {
    fn tepl_broadcastable(&self, batch: &[u64], weight: &[u64]) -> Option<bool> {
        Some(batch == weight)
    }
    fn tepl_reassociable(
        &self,
        _: &TensorInfo,
        _: &TensorInfo,
        _: &TensorInfo,
        outer: &OpAttrs,
        inner: &OpAttrs,
    ) -> Option<bool> {
        Some(self.allowed && *outer == attrs() && *inner == attrs())
    }
    fn tepl_infer_dot(
        &self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        attributes: &OpAttrs,
    ) -> Option<OpAttrs> {
        dot_shape(&lhs.shape, &rhs.shape, attributes)?;
        Some(attributes.clone())
    }
    fn tepl_infer_lora_out(
        &self,
        x: &TensorInfo,
        a: &TensorInfo,
        b: &TensorInfo,
        outer: &OpAttrs,
        inner: &OpAttrs,
    ) -> Option<OpAttrs> {
        let xa = dot_shape(&x.shape, &a.shape, outer)?;
        dot_shape(&xa, &b.shape, inner)?;
        Some(inner.clone())
    }
}
struct Inference {
    reject_intermediate: bool,
}
impl OutputInference for Inference {
    fn infer_output(
        &self,
        op: Op,
        inputs: &[TensorInfo],
        attributes: &OpAttrs,
    ) -> Option<TensorInfo> {
        let [lhs, rhs] = inputs else {
            return None;
        };
        if lhs.dtype != rhs.dtype {
            return None;
        }
        let shape = match op {
            Op::TensorLang(tensor_lang::Op::Add) if lhs.shape == rhs.shape => lhs.shape.clone(),
            Op::TensorLang(tensor_lang::Op::DotGeneral) => {
                if self.reject_intermediate {
                    return None;
                }
                dot_shape(&lhs.shape, &rhs.shape, attributes)?
            }
            _ => return None,
        };
        Some(TensorInfo {
            shape,
            dtype: lhs.dtype,
        })
    }
}
fn dot(graph: &mut EGraph<OpNode, ()>, lhs: Id, rhs: Id) -> Id {
    graph.add(
        OpNode::from_parts(
            Op::TensorLang(tensor_lang::Op::DotGeneral),
            vec![lhs, rhs],
            attrs(),
        )
        .unwrap(),
    )
}
#[test]
fn generated_lora_checks_shapes_host_legality_and_every_intermediate_before_insertion() {
    for (b_columns, allowed, reject_intermediate, accepted) in [
        (5, true, false, true),
        (6, true, false, false),
        (5, false, false, false),
        (5, true, true, false),
    ] {
        let mut graph = EGraph::<OpNode, ()>::default();
        let [x, w, a, b] = ["X", "W", "A", "B"].map(|name| {
            graph.add(
                OpNode::from_parts(
                    Op::TensorLang(tensor_lang::Op::Symbol),
                    vec![],
                    OpAttrs::TensorLang(tensor_lang::OpAttrs::SymbolAttrs {
                        tepl_name: name.into(),
                    }),
                )
                .unwrap(),
            )
        });
        let ab = dot(&mut graph, a, b);
        let sum = graph.add(
            OpNode::from_parts(
                Op::TensorLang(tensor_lang::Op::Add),
                vec![w, ab],
                OpAttrs::None,
            )
            .unwrap(),
        );
        let root = dot(&mut graph, x, sum);
        let entries = HashMap::from([
            (x, vec![2, 3, 4]),
            (w, vec![2, 4, 5]),
            (a, vec![2, 4, 2]),
            (b, vec![2, 2, b_columns]),
            (root, vec![2, 3, 5]),
        ]);
        let metadata = move |graph: &EGraph<OpNode, ()>, id: Id| {
            entries
                .iter()
                .find(|(key, _)| graph.find(**key) == graph.find(id))
                .map(|(_, shape)| TensorInfo {
                    shape: shape.clone(),
                    dtype: DType::F32,
                })
        };
        let rule = rule_lora::build_rewrite(
            metadata,
            Inference {
                reject_intermediate,
            },
            Host { allowed },
        )
        .unwrap();
        graph.rebuild();
        let before = graph.total_size();
        let matches = rule.search(&graph);
        assert!(!matches.is_empty());
        assert_eq!(!rule.apply(&mut graph, &matches).is_empty(), accepted);
        if accepted {
            let xw = graph
                .lookup(
                    OpNode::from_parts(
                        Op::TensorLang(tensor_lang::Op::DotGeneral),
                        vec![x, w],
                        attrs(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let xa = graph
                .lookup(
                    OpNode::from_parts(
                        Op::TensorLang(tensor_lang::Op::DotGeneral),
                        vec![x, a],
                        attrs(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let xab = graph
                .lookup(
                    OpNode::from_parts(
                        Op::TensorLang(tensor_lang::Op::DotGeneral),
                        vec![xa, b],
                        attrs(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let rhs = graph
                .lookup(
                    OpNode::from_parts(
                        Op::TensorLang(tensor_lang::Op::Add),
                        vec![xw, xab],
                        OpAttrs::None,
                    )
                    .unwrap(),
                )
                .unwrap();
            assert_eq!(graph.find(rhs), graph.find(root));
        } else {
            assert_eq!(graph.total_size(), before);
        }
    }
}
