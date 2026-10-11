use egg::{EGraph, Id};
use tepl_generated::ir::analysis::{TensorAnalysis, TensorBindingTable};
use tepl_generated::ir::dialects::tensor_lang::{Op, OpAttrs};
use tepl_generated::ir::rules::semantic_general::*;
use tepl_generated::ir::{DType, OpNode, TensorInfo};
fn dot(lc: Vec<u64>, rc: Vec<u64>, lb: Vec<u64>, rb: Vec<u64>, a: Id, b: Id) -> OpNode {
    OpNode::new(
        Op::DotGeneral,
        OpAttrs::DotGeneralAttrs {
            lhs_contracting_dimensions: lc,
            rhs_contracting_dimensions: rc,
            lhs_batching_dimensions: lb,
            rhs_batching_dimensions: rb,
        },
        vec![a, b],
    )
    .unwrap()
}
fn graph(a: Vec<u64>, b: Vec<u64>) -> (EGraph<OpNode, TensorAnalysis>, Id, Id) {
    let mut bindings = TensorBindingTable::default();
    bindings
        .register_symbol(
            "a",
            TensorInfo {
                shape: a,
                dtype: DType::F32,
            },
        )
        .unwrap();
    bindings
        .register_symbol(
            "b",
            TensorInfo {
                shape: b,
                dtype: DType::F32,
            },
        )
        .unwrap();
    let mut g = EGraph::new(TensorAnalysis::new(bindings));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    (g, a, b)
}
#[test]
fn inverse_cycle_and_free_axis_guard() {
    let (mut g, a, b) = graph(vec![3, 2, 4], vec![3, 5]);
    let root = g.add(dot(vec![0], vec![0], vec![], vec![], a, b));
    g.rebuild();
    let rule = rule_cycle::build(()).unwrap();
    let matches = rule.search(&g);
    assert!(!rule.apply(&mut g, &matches).is_empty());
    g.rebuild();
    let t = g
        .lookup(
            OpNode::new(
                Op::Transpose,
                OpAttrs::TransposeAttrs {
                    permutation: vec![1, 2, 0],
                },
                vec![a],
            )
            .unwrap(),
        )
        .unwrap();
    let expected = g
        .lookup(dot(vec![2], vec![0], vec![], vec![], t, b))
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
    for shape in [vec![2, 4, 3], vec![2, 2, 3]] {
        let (mut g, a, b) = graph(shape, vec![3, 5]);
        g.add(dot(vec![2], vec![0], vec![], vec![], a, b));
        g.rebuild();
        let matches = rule.search(&g);
        assert!(rule.apply(&mut g, &matches).is_empty());
    }
}
#[test]
fn arbitrary_rank_and_multiple_axes() {
    for rank in [2usize, 3, 5, 12] {
        for extent in [0, 1, 3] {
            let mut dims = vec![2; rank];
            dims[rank - 1] = extent;
            let (mut g, a, b) = graph(dims, vec![extent, 5]);
            g.add(dot(vec![(rank - 1) as u64], vec![0], vec![], vec![], a, b));
            g.rebuild();
            let rule = rule_identity::build(()).unwrap();
            let found = rule.search(&g);
            assert!(!rule.apply(&mut g, &found).is_empty());
        }
    }
    let (mut g, a, b) = graph(vec![2, 3, 4, 5, 6], vec![2, 3, 5, 6, 7]);
    g.add(dot(vec![3, 4], vec![2, 3], vec![0, 1], vec![0, 1], a, b));
    g.rebuild();
    let rule = rule_identity::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
}
#[test]
fn algebraic_distribution_and_association() {
    let mut bindings = TensorBindingTable::default();
    for (name, shape) in [("a", vec![2, 3]), ("b", vec![2, 3]), ("c", vec![3, 4])] {
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
    let mut g = EGraph::new(TensorAnalysis::new(bindings));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    let c = g.add(OpNode::input("c"));
    let sum = g.add(OpNode::new(Op::Add, OpAttrs::None, vec![a, b]).unwrap());
    let root = g.add(dot(vec![1], vec![0], vec![], vec![], sum, c));
    g.rebuild();
    let rule = rule_distribute::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let ac = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], a, c))
        .unwrap();
    let bc = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], b, c))
        .unwrap();
    let expected = g
        .lookup(OpNode::new(Op::Add, OpAttrs::None, vec![ac, bc]).unwrap())
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
    let mut bindings = TensorBindingTable::default();
    for (name, shape) in [("a", vec![2, 3]), ("b", vec![3, 4]), ("c", vec![4, 5])] {
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
    let mut g = EGraph::new(TensorAnalysis::new(bindings));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    let c = g.add(OpNode::input("c"));
    let ab = g.add(dot(vec![1], vec![0], vec![], vec![], a, b));
    g.add(dot(vec![1], vec![0], vec![], vec![], ab, c));
    g.rebuild();
    let rule = rule_associate::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
}

#[test]
fn batched_association_and_lora() {
    for batch_rank in [0, 1, 3, 10] {
        let mut batch = vec![2; batch_rank];
        let a_shape = {
            let mut s = batch.clone();
            s.extend([3, 4]);
            s
        };
        let b_shape = {
            let mut s = batch.clone();
            s.extend([4, 5]);
            s
        };
        let c_shape = {
            let mut s = batch.clone();
            s.extend([5, 6]);
            s
        };
        let mut bindings = TensorBindingTable::default();
        for (name, shape) in [("a", a_shape), ("b", b_shape), ("c", c_shape)] {
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
        let mut g = EGraph::new(TensorAnalysis::new(bindings));
        let a = g.add(OpNode::input("a"));
        let b = g.add(OpNode::input("b"));
        let c = g.add(OpNode::input("c"));
        let axes: Vec<u64> = (0..batch_rank as u64).collect();
        let left = (batch_rank + 1) as u64;
        let right = batch_rank as u64;
        let ab = g.add(dot(
            vec![left],
            vec![right],
            axes.clone(),
            axes.clone(),
            a,
            b,
        ));
        let root = g.add(dot(
            vec![left],
            vec![right],
            axes.clone(),
            axes.clone(),
            ab,
            c,
        ));
        g.rebuild();
        assert_eq!(
            tepl_generated::ir::analysis::tensor_info(&g, root)
                .unwrap()
                .shape,
            {
                batch.extend([3, 6]);
                batch
            }
        );
        let rule = rule_associate::build(()).unwrap();
        let found = rule.search(&g);
        assert!(
            !rule.apply(&mut g, &found).is_empty(),
            "batch rank {batch_rank}"
        );
        g.rebuild();
        let bc = g
            .lookup(dot(
                vec![left],
                vec![right],
                axes.clone(),
                axes.clone(),
                b,
                c,
            ))
            .unwrap();
        let expected = g
            .lookup(dot(vec![left], vec![right], axes.clone(), axes, a, bc))
            .unwrap();
        assert_eq!(g.find(root), g.find(expected));
    }
    let mut bindings = TensorBindingTable::default();
    for (name, shape) in [
        ("x", vec![2, 3]),
        ("w", vec![3, 5]),
        ("a", vec![3, 4]),
        ("b", vec![4, 5]),
    ] {
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
    let mut g = EGraph::new(TensorAnalysis::new(bindings));
    let x = g.add(OpNode::input("x"));
    let w = g.add(OpNode::input("w"));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    let ab = g.add(dot(vec![1], vec![0], vec![], vec![], a, b));
    let sum = g.add(OpNode::new(Op::Add, OpAttrs::None, vec![w, ab]).unwrap());
    let root = g.add(dot(vec![1], vec![0], vec![], vec![], x, sum));
    g.rebuild();
    let partial = rule_lora_partial::build(()).unwrap();
    let mut partial_graph = g.clone();
    let found = partial.search(&partial_graph);
    assert!(!partial.apply(&mut partial_graph, &found).is_empty());
    let rule = rule_lora::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let xw = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], x, w))
        .unwrap();
    let xa = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], x, a))
        .unwrap();
    let out = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], xa, b))
        .unwrap();
    let expected = g
        .lookup(OpNode::new(Op::Add, OpAttrs::None, vec![xw, out]).unwrap())
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
}

#[test]
fn repeated_capture_symmetry_and_invalid_no_mutation() {
    let (mut g, a, _) = graph(vec![2], vec![2]);
    let root = g.add(dot(vec![], vec![], vec![], vec![], a, a));
    g.rebuild();
    let rule = rule_symmetric_product::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let expected = g
        .lookup(
            OpNode::new(
                Op::Transpose,
                OpAttrs::TransposeAttrs {
                    permutation: vec![1, 0],
                },
                vec![root],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
    let (mut g, a, b) = graph(vec![2], vec![2]);
    g.add(dot(vec![], vec![], vec![], vec![], a, b));
    g.rebuild();
    let rule = rule_invalid_drop_operand::build(()).unwrap();
    let found = rule.search(&g);
    let before = g.total_size();
    assert!(rule.apply(&mut g, &found).is_empty());
    assert_eq!(before, g.total_size());
}

#[test]
fn reordered_contract_and_reverse_reassociation() {
    let mut bindings = TensorBindingTable::default();
    for (name, shape) in [("a", vec![2, 3]), ("b", vec![4, 3]), ("c", vec![4, 5])] {
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
    let mut g = EGraph::new(TensorAnalysis::new(bindings));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    let c = g.add(OpNode::input("c"));
    let ab = g.add(dot(vec![1], vec![1], vec![], vec![], a, b));
    let root = g.add(dot(vec![1], vec![0], vec![], vec![], ab, c));
    g.rebuild();
    let rule = rule_associate::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let bc = g
        .lookup(dot(vec![0], vec![0], vec![], vec![], b, c))
        .unwrap();
    let expected = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], a, bc))
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
    // Verify the reverse direction on a fresh graph, so the replacement has
    // not already been inserted by the forward rewrite.
    let mut bindings = TensorBindingTable::default();
    for (name, shape) in [("a", vec![2, 3]), ("b", vec![4, 3]), ("c", vec![4, 5])] {
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
    let mut g = EGraph::new(TensorAnalysis::new(bindings));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    let c = g.add(OpNode::input("c"));
    let bc = g.add(dot(vec![0], vec![0], vec![], vec![], b, c));
    let root = g.add(dot(vec![1], vec![0], vec![], vec![], a, bc));
    g.rebuild();
    let reverse = rule_associate_reverse::build(()).unwrap();
    let found = reverse.search(&g);
    assert!(!reverse.apply(&mut g, &found).is_empty());
    g.rebuild();
    let ab = g
        .lookup(dot(vec![1], vec![1], vec![], vec![], a, b))
        .unwrap();
    let expected = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], ab, c))
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
}

#[test]
fn let_bound_reduction_is_fresh_at_each_use() {
    let (mut g, a, _) = graph(vec![2, 3], vec![2]);
    let root = g.add(
        OpNode::new(
            Op::SquaredSum,
            OpAttrs::Reduction { reduced: vec![1] },
            vec![a],
        )
        .unwrap(),
    );
    g.rebuild();
    let rule = rule_reused_reduction::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let fold = g
        .lookup(OpNode::new(Op::Fold, OpAttrs::Reduction { reduced: vec![1] }, vec![a]).unwrap())
        .unwrap();
    let expected = g
        .lookup(dot(vec![], vec![], vec![0], vec![0], fold, fold))
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
}

#[test]
fn bound_intermediate_keeps_its_semantics() {
    let (mut g, a, b) = graph(vec![2, 3], vec![3, 4]);
    let root = g.add(dot(vec![1], vec![0], vec![], vec![], a, b));
    g.rebuild();
    let rule = rule_bound_intermediate::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let expected = g
        .lookup(
            OpNode::new(
                Op::Transpose,
                OpAttrs::TransposeAttrs {
                    permutation: vec![0, 1],
                },
                vec![root],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
}
