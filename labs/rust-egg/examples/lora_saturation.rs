//! Run LoRA equality saturation and write Graphviz files for inspection.

#[path = "support/lora_saturation.rs"]
mod support;

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use egg::{Extractor, Runner};

use rust_egg::ir::TensorLang;
use rust_egg::ir::rules::{rule_commute_add, rule_lora};

use support::{
    ArithmeticCost, DemoLoraFunctions, ShapeAnalysis, dot_attrs, evaluate, example_shapes,
    example_values, expected_expr, input_graph, stopped_by_saturation, tensor_info, text_dump,
};

fn main() -> Result<(), Box<dyn Error>> {
    let shapes = example_shapes();
    let inputs = example_values(&shapes);
    let (egraph, root, original) = input_graph(shapes, false, dot_attrs());
    let output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/lora");
    fs::create_dir_all(&output)?;

    egraph.dot().to_dot(output.join("before.dot"))?;
    println!("Before saturation:\n{}", text_dump(&egraph));

    let rules = [
        rule_lora::build_rewrite(
            tensor_info,
            DemoLoraFunctions {
                allow_reassociation: true,
            },
        )
        .unwrap(),
        rule_commute_add::build_rewrite().unwrap(),
    ];
    let runner = Runner::<TensorLang, ShapeAnalysis>::new(ShapeAnalysis::default())
        .with_egraph(egraph)
        .with_iter_limit(12)
        .with_node_limit(1_000)
        .with_time_limit(Duration::from_secs(5))
        .run(&rules);
    if !stopped_by_saturation(&runner.stop_reason) {
        return Err(format!("runner stopped early: {:?}", runner.stop_reason).into());
    }
    runner.egraph.dot().to_dot(output.join("after.dot"))?;
    println!("After saturation:\n{}", text_dump(&runner.egraph));
    runner.print_report();
    for (index, iteration) in runner.iterations.iter().enumerate() {
        println!("iteration {index}: applied {:?}", iteration.applied);
    }

    let expected = expected_expr();
    let alternative = runner
        .egraph
        .lookup_expr(&expected)
        .ok_or("LoRA alternative was not created")?;
    if runner.egraph.find(root) != runner.egraph.find(alternative) {
        return Err("original and LoRA alternative are in different e-classes".into());
    }
    let (cost, best) = Extractor::new(
        &runner.egraph,
        ArithmeticCost {
            egraph: &runner.egraph,
        },
    )
    .find_best(root);
    if evaluate(&original, &inputs)? != evaluate(&best, &inputs)? {
        return Err("extracted expression changed the tensor result".into());
    }
    println!("Original: {original}");
    println!("Extracted: {best}");
    println!("Estimated arithmetic operations: {cost}");
    println!("Graphviz files: {}", output.display());
    Ok(())
}
