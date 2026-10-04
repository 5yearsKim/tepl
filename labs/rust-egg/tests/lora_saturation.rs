//! End-to-end checks for LoRA equality saturation, extraction, and evaluation.

use rust_egg::ir::dialects::tensor_lang;
#[path = "../examples/support/lora_saturation.rs"]
mod support;

use std::time::Duration;

use egg::{EGraph, Extractor, Runner, StopReason};
use rust_egg::host::{LoraAnalysis, infer_lora_tensor_output, lora_tensor_info};
use rust_egg::ir::pattern::TensorInfo;
use rust_egg::ir::rules::{lora::rule_lora, simple::rule_commute_add};
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

use support::{
    ArithmeticCost, DemoLoraFunctions, dot_attrs, evaluate, example_shapes, example_values,
    expected_expr, input_graph, stopped_by_saturation, text_dump,
};

fn run_rules(
    egraph: EGraph<OpNode, LoraAnalysis>,
    allow_reassociation: bool,
) -> Runner<OpNode, LoraAnalysis> {
    let rules = [
        rule_lora::build_rewrite_with(
            lora_tensor_info,
            infer_lora_tensor_output,
            DemoLoraFunctions {
                allow_reassociation,
            },
        )
        .unwrap(),
        rule_commute_add::build_rewrite_with(lora_tensor_info, infer_lora_tensor_output, ())
            .unwrap(),
    ];
    Runner::<OpNode, LoraAnalysis>::new(LoraAnalysis::default())
        .with_egraph(egraph)
        .with_iter_limit(12)
        .with_node_limit(1_000)
        .with_time_limit(Duration::from_secs(5))
        .run(&rules)
}

#[test]
fn lora_saturates_extracts_cheaper_expression_and_preserves_values() {
    let shapes = example_shapes();
    let values = example_values(&shapes);
    let (egraph, root, original) = input_graph(shapes, false, dot_attrs());
    assert!(egraph.lookup_expr(&expected_expr()).is_none());

    let runner = run_rules(egraph, true);
    assert!(stopped_by_saturation(&runner.stop_reason));
    let alternative = runner.egraph.lookup_expr(&expected_expr()).unwrap();
    assert_eq!(runner.egraph.find(root), runner.egraph.find(alternative));
    assert_eq!(
        lora_tensor_info(&runner.egraph, root),
        Some(TensorInfo {
            shape: vec![2, 4, 32],
            dtype: DType::I64
        })
    );

    let (cost, best) = Extractor::new(
        &runner.egraph,
        ArithmeticCost {
            egraph: &runner.egraph,
        },
    )
    .find_best(root);
    assert_eq!(cost, 39_168);
    assert_eq!(
        best.as_ref().last().unwrap().op(),
        Op::TensorLang(tensor_lang::Op::Add)
    );
    assert!(cost < 69_632);
    assert_eq!(
        evaluate(&original, &values).unwrap(),
        evaluate(&best, &values).unwrap()
    );
    assert!(text_dump(&runner.egraph).contains("eclass"));
    let dot = runner.egraph.dot().to_string();
    assert!(dot.contains("digraph egraph"));
    assert!(dot.contains("dot_general"));

    let nodes = runner.egraph.total_number_of_nodes();
    let classes = runner.egraph.number_of_classes();
    let rerun = run_rules(runner.egraph, true);
    assert!(matches!(rerun.stop_reason, Some(StopReason::Saturated)));
    assert_eq!(rerun.egraph.total_number_of_nodes(), nodes);
    assert_eq!(rerun.egraph.number_of_classes(), classes);
}

#[test]
fn commutation_exposes_lora_match_in_a_later_iteration() {
    let (egraph, root, _) = input_graph(example_shapes(), true, dot_attrs());
    let runner = run_rules(egraph, true);
    assert!(stopped_by_saturation(&runner.stop_reason));
    let alternative = runner.egraph.lookup_expr(&expected_expr()).unwrap();
    assert_eq!(runner.egraph.find(root), runner.egraph.find(alternative));

    let count = |index: usize, rule: &str| {
        runner.iterations[index]
            .applied
            .iter()
            .find(|(name, _)| name.to_string() == rule)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    };
    assert_eq!(count(0, "lora::lora"), 0);
    assert!(count(0, "simple::commute_add") > 0);
    assert!(runner.iterations.iter().skip(1).any(|iteration| {
        iteration
            .applied
            .iter()
            .any(|(name, count)| name.to_string() == "lora::lora" && *count > 0)
    }));
}

fn dot_count(egraph: &EGraph<OpNode, LoraAnalysis>) -> usize {
    egraph
        .classes()
        .flat_map(|class| &class.nodes)
        .filter(|node| node.op() == Op::TensorLang(tensor_lang::Op::DotGeneral))
        .count()
}

#[test]
fn invalid_shape_unsupported_axes_and_reassociation_rejection_do_not_expand_lora() {
    let mut wrong_shape = example_shapes();
    wrong_shape.insert("B".into(), vec![2, 4, 31]);
    let wrong_axes = OpAttrs::TensorLang(tensor_lang::OpAttrs::DotGeneralAttrs {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![2],
        lhs_batching_dimensions: vec![0],
        rhs_batching_dimensions: vec![0],
        precision_config: vec![rust_egg::ir::types::Precision::Default; 2],
        algorithm: None,
    });
    for (shapes, inner_attrs, allow_reassociation) in [
        (wrong_shape, dot_attrs(), true),
        (example_shapes(), wrong_axes, true),
        (example_shapes(), dot_attrs(), false),
    ] {
        let (egraph, _, _) = input_graph(shapes, false, inner_attrs);
        let before = dot_count(&egraph);
        let runner = run_rules(egraph, allow_reassociation);
        assert!(stopped_by_saturation(&runner.stop_reason));
        assert!(runner.egraph.lookup_expr(&expected_expr()).is_none());
        assert_eq!(dot_count(&runner.egraph), before);
    }
}
