use egg::{EGraph, Id};
use tepl_generated::ir::analysis::{TensorAnalysis, TensorBindingTable};
use tepl_generated::ir::dialects::tensor_lang::{Op, OpAttrs};
use tepl_generated::ir::rules::semantic::*;
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
fn arbitrary_rank_multiple_axes_and_empty_extents() {
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
fn typed_auto_overrides() {
    let (mut g, a, _) = graph(vec![2, 3], vec![3, 4]);
    let node = OpNode::new(
        Op::FieldCopy,
        OpAttrs::FieldOptions {
            label: Some("none".to_string()),
            flags: vec![vec![true, false]],
            weights: vec![vec![-1, 2]],
            mode: DType::F32,
            opaque: None,
            precision: vec![],
        },
        vec![a],
    )
    .unwrap();
    g.add(node);
    g.rebuild();
    let rule = rule_typed_fields::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
}

#[test]
fn factor_addition_into_a_single_contraction() {
    let mut inputs = TensorBindingTable::default();
    for (name, shape) in [("a", vec![2, 3]), ("b", vec![2, 3]), ("c", vec![3, 4])] {
        inputs
            .register_symbol(
                name,
                TensorInfo {
                    shape,
                    dtype: DType::F32,
                },
            )
            .unwrap();
    }
    let mut g = EGraph::new(TensorAnalysis::new(inputs));
    let a = g.add(OpNode::input("a"));
    let b = g.add(OpNode::input("b"));
    let c = g.add(OpNode::input("c"));
    let ac = g.add(dot(vec![1], vec![0], vec![], vec![], a, c));
    let bc = g.add(dot(vec![1], vec![0], vec![], vec![], b, c));
    let root = g.add(OpNode::new(Op::Add, OpAttrs::None, vec![ac, bc]).unwrap());
    g.rebuild();
    let rule = rule_factor::build(()).unwrap();
    let found = rule.search(&g);
    assert!(!rule.apply(&mut g, &found).is_empty());
    g.rebuild();
    let sum = g
        .lookup(OpNode::new(Op::Add, OpAttrs::None, vec![a, b]).unwrap())
        .unwrap();
    let expected = g
        .lookup(dot(vec![1], vec![0], vec![], vec![], sum, c))
        .unwrap();
    assert_eq!(g.find(root), g.find(expected));
}
