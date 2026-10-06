mod support;
use egg::{EGraph, Id, Language};
use rust_egg::ir::TensorInfo;
use rust_egg::ir::dialects::{scalar as s, tensor_lang as t};
use rust_egg::ir::rewriting::OutputInference;
use rust_egg::ir::rules::{lowering::rule_scalar_add, scalar::rule_commute_add};
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

struct Inference(bool);
impl OutputInference for Inference {
    fn infer_output(&self, op: Op, inputs: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
        if !self.0 || !matches!(op, Op::Scalar(s::Op::Add) | Op::TensorLang(t::Op::Add)) {
            return None;
        }
        let [lhs, rhs] = inputs else { return None };
        (lhs == rhs).then(|| lhs.clone())
    }
}
fn metadata(_: &EGraph<OpNode, support::TestAnalysis>, _: Id) -> Option<TensorInfo> {
    Some(TensorInfo {
        shape: vec![],
        dtype: DType::I32,
    })
}

#[test]
fn dialect_names_and_discriminants_remain_distinct() {
    let x = Id::from(0);
    let tensor = OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x, x]).unwrap();
    let scalar = OpNode::new(s::Op::Add, s::OpAttrs::None, vec![x, x]).unwrap();
    assert_ne!(tensor.discriminant(), scalar.discriminant());
    assert!(!tensor.matches(&scalar));
    assert_eq!(t::Op::from_name("add"), Some(t::Op::Add));
    assert_eq!(s::Op::from_name("add"), Some(s::Op::Add));
    assert_eq!(Op::from_name("Scalar.add"), Some(s::Op::Add.into()));
    assert_eq!(Op::from_name("add"), None);
    assert!(OpNode::new(s::Op::Add, s::OpAttrs::None, vec![x]).is_err());
    // A dynamic caller still gets runtime schema validation.
    assert!(
        OpNode::from_parts(
            s::Op::Add.into(),
            vec![x, x],
            OpAttrs::TensorLang(t::OpAttrs::None)
        )
        .is_err()
    );
}

#[test]
fn cross_dialect_lowering_and_scalar_rule_share_one_graph() {
    let mut graph = EGraph::<OpNode, support::TestAnalysis>::default();
    let x = graph.add(OpNode::literal("1", DType::I32).unwrap());
    let y = graph.add(OpNode::literal("2", DType::I32).unwrap());
    let root = graph.add(OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x, y]).unwrap());
    graph.rebuild();
    let scalar_rule = support::configure_rule(
        &mut graph,
        metadata,
        Inference(true),
        rule_commute_add::build(()),
    )
    .unwrap();
    assert!(scalar_rule.search(&graph).is_empty());
    let lowering = support::configure_rule(
        &mut graph,
        metadata,
        Inference(true),
        rule_scalar_add::build(()),
    )
    .unwrap();
    let matches = lowering.search(&graph);
    assert!(!lowering.apply(&mut graph, &matches).is_empty());
    graph.rebuild();
    let matches = scalar_rule.search(&graph);
    assert!(!scalar_rule.apply(&mut graph, &matches).is_empty());
    let swapped = graph
        .lookup(OpNode::new(s::Op::Add, s::OpAttrs::None, vec![y, x]).unwrap())
        .unwrap();
    assert_eq!(graph.find(root), graph.find(swapped));
    assert_eq!(lowering.name.to_string(), "lowering::scalar_add");
    assert_eq!(scalar_rule.name.to_string(), "scalar::commute_add");
}

#[test]
fn rejected_cross_dialect_inference_is_atomic() {
    let mut graph = EGraph::<OpNode, support::TestAnalysis>::default();
    let x = graph.add(OpNode::literal("1", DType::I32).unwrap());
    graph.add(OpNode::new(t::Op::Add, t::OpAttrs::None, vec![x, x]).unwrap());
    graph.rebuild();
    let before = graph.total_size();
    let rule = support::configure_rule(
        &mut graph,
        metadata,
        Inference(false),
        rule_scalar_add::build(()),
    )
    .unwrap();
    let matches = rule.search(&graph);
    assert!(rule.apply(&mut graph, &matches).is_empty());
    assert_eq!(graph.total_size(), before);
}
#[test]
fn shape_builtins_compose_in_generated_modules() {
    use rust_egg::ir::builtins::{BuiltinError, BuiltinResult, common, shape};

    fn transpose(input: &[u64], permutation: &[u64]) -> BuiltinResult<Vec<u64>> {
        common::ensure(
            shape::is_valid_axis_list(permutation, input.len()) && permutation.len() == input.len(),
            "invalid permutation",
        )?;
        common::gather(input, permutation)
    }

    assert_eq!(transpose(&[2, 3, 4], &[2, 0, 1]), Ok(vec![4, 2, 3]));
    assert_eq!(
        transpose(&[2, 3], &[0, 0]),
        Err(BuiltinError::Assertion("invalid permutation")),
    );
    assert_eq!(common::gather(&[2, 3], &[1, 1]), Ok(vec![3, 3]));
    assert_eq!(common::product(&[u64::MAX, 2]), Err(BuiltinError::Overflow));
}

#[test]
fn full_shape_builtins_support_reduction_and_concatenation() {
    use rust_egg::ir::builtins::{common, shape};

    // Reduction removes axis values, then gathers the remaining dimensions.
    let input = [2_u64, 3, 4, 5];
    let axes = common::range(common::len(&input)).unwrap();
    let remaining = common::exclude(&axes, &[1, 3]);
    assert_eq!(common::gather(&input, &remaining).unwrap(), vec![2, 4]);

    // Concatenation checks the other axes, sums the joining dimension, and
    // replaces that dimension in the first shape.
    let inputs = common::concat(&[&[vec![2_u64, 3]], &[vec![2, 5], vec![2, 1]]]).unwrap();
    let same_outer: Vec<_> = inputs.iter().map(|input| input[0] == 2).collect();
    assert!(common::all(&same_outer));
    let sizes: Vec<_> = inputs.iter().map(|input| input[1]).collect();
    assert_eq!(
        common::replace(&inputs[0], 1, common::sum(&sizes).unwrap()).unwrap(),
        vec![2, 9],
    );
    assert_eq!(common::floor_div(-7_i64, 3), Ok(-3));
    assert_eq!(common::ceil_div(-7_i64, 3), Ok(-2));
    assert_eq!(shape::broadcast_shape(&[0, 4], &[1, 4]), Ok(vec![0, 4]));
}
