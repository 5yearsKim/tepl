use egg::{EGraph, Id, Rewrite, Var};
use rust_egg::host::nodes::*;
use rust_egg::ir::analysis::TensorBindingTable;
use rust_egg::ir::dialects::tensor_lang;
use rust_egg::ir::pattern::{
    AttrExpr, TensorExpr, TensorInfo, TensorMetadata, TensorPattern, matches_at,
    tensor_rewrite_checked,
};
use rust_egg::ir::rules::basic::{
    rule_commute_f32, rule_commute_float_literal, rule_commute_same_dtype, rule_commute_scalar,
};
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};

fn info(shape: &[u64], dtype: DType) -> TensorInfo {
    TensorInfo {
        shape: shape.to_vec(),
        dtype,
    }
}

fn add_output(op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
    if let (
        Op::Literal,
        OpAttrs::Literal {
            dtype: Some(dtype), ..
        },
        [],
    ) = (op, attrs, operands)
    {
        return Some(info(&[], *dtype));
    }
    let [x, y] = operands else {
        return None;
    };
    if op != Op::TensorLang(tensor_lang::Op::Add)
        || *attrs != OpAttrs::None
        || x.dtype != y.dtype
        || x.dtype == DType::Bool
    {
        return None;
    }
    if x.shape == y.shape || y.shape.is_empty() {
        Some(x.clone())
    } else if x.shape.is_empty() {
        Some(y.clone())
    } else {
        None
    }
}

fn metadata(inputs: [TensorInfo; 2]) -> impl TensorMetadata<()> {
    move |graph: &EGraph<OpNode, ()>, id: Id| {
        // This fixture has exactly symbols, literals and one add root. Every
        // witness in an e-class must independently have the same known type.
        let mut result = None;
        for node in &graph[graph.find(id)].nodes {
            let value = match (node.op(), node.attrs()) {
                (Op::Input, OpAttrs::Input { name }) if name == "X" => inputs[0].clone(),
                (Op::Input, OpAttrs::Input { name }) if name == "Y" => inputs[1].clone(),
                (
                    Op::Literal,
                    OpAttrs::Literal {
                        dtype: Some(dtype), ..
                    },
                ) => info(&[], *dtype),
                (Op::TensorLang(tensor_lang::Op::Add), _) => add_output(
                    Op::TensorLang(tensor_lang::Op::Add),
                    &inputs,
                    &OpAttrs::None,
                )?,
                _ => return None,
            };
            if result.as_ref().is_some_and(|previous| previous != &value) {
                return None;
            }
            result = Some(value);
        }
        result
    }
}

struct Host;
impl rule_commute_same_dtype::Functions for Host {
    fn is_same_dtype(&self, x: &TensorInfo, y: &TensorInfo) -> Option<bool> {
        Some(x.dtype == y.dtype)
    }
}

fn graph() -> (EGraph<OpNode, ()>, [Id; 3]) {
    let mut graph = EGraph::default();
    let x = graph.add(symbol("X"));
    let y = graph.add(symbol("Y"));
    let root = graph.add(binary(tensor_lang::Op::Add, x, y).unwrap());
    graph.rebuild();
    (graph, [x, y, root])
}

fn apply(graph: &mut EGraph<OpNode, ()>, rule: &Rewrite<OpNode, ()>, accepted: bool) {
    let before = graph.total_number_of_nodes();
    let matches = rule.search(graph);
    assert!(!matches.is_empty());
    let changed = rule.apply(graph, &matches);
    assert_eq!(!changed.is_empty(), accepted);
    if !accepted {
        assert_eq!(graph.total_number_of_nodes(), before);
    }
}

#[test]
fn typed_declarations_reject_same_shaped_tensors_of_other_dtypes() {
    for dtype in DType::ALL {
        let (mut graph, _) = graph();
        let rule = rule_commute_f32::build_rewrite(
            metadata([info(&[4], dtype), info(&[4], dtype)]),
            add_output,
            (),
        )
        .unwrap();
        apply(&mut graph, &rule, dtype == DType::F32);
    }
}

#[test]
fn shape_only_declarations_and_generated_host_trait_use_concrete_dtype() {
    for dtype in [DType::BF16, DType::F32, DType::I64] {
        let (mut graph, _) = graph();
        let rule = rule_commute_same_dtype::build_rewrite(
            metadata([info(&[0], dtype), info(&[0], dtype)]),
            add_output,
            Host,
        )
        .unwrap();
        apply(&mut graph, &rule, true);
    }
    assert_eq!(
        rule_commute_same_dtype::Functions::is_same_dtype(
            &Host,
            &info(&[4], DType::F32),
            &info(&[4], DType::BF16)
        ),
        Some(false)
    );
    let (mut graph, _) = graph();
    let rule = rule_commute_same_dtype::build_rewrite(
        metadata([info(&[4], DType::F32), info(&[4], DType::BF16)]),
        add_output,
        Host,
    )
    .unwrap();
    apply(&mut graph, &rule, false);
}

#[test]
fn typed_scalar_is_rank_zero_and_typed_literal_is_exact() {
    for (shape, dtype, accepted) in [
        (vec![], DType::F32, true),
        (vec![1], DType::F32, false),
        (vec![], DType::BF16, false),
    ] {
        let (mut graph, _) = graph();
        let rule = rule_commute_scalar::build_rewrite(
            metadata([info(&[], DType::F32), info(&shape, dtype)]),
            add_output,
            (),
        )
        .unwrap();
        apply(&mut graph, &rule, accepted);
    }
    for dtype in [DType::F32, DType::BF16] {
        let mut graph = EGraph::default();
        let x = graph.add(symbol("X"));
        let value = graph.add(OpNode::literal("1.0", dtype).unwrap());
        graph.add(binary(tensor_lang::Op::Add, x, value).unwrap());
        graph.rebuild();
        let rule = rule_commute_float_literal::build_rewrite(
            metadata([info(&[], DType::F32), info(&[], dtype)]),
            add_output,
            (),
        )
        .unwrap();
        let matches = rule.search(&graph);
        assert_eq!(!matches.is_empty(), dtype == DType::F32);
        assert_eq!(
            !rule.apply(&mut graph, &matches).is_empty(),
            dtype == DType::F32
        );
    }
}

#[test]
fn dtype_participates_in_literal_identity_and_matching() {
    let mut graph = EGraph::<OpNode, ()>::default();
    let i32_id = graph.add(OpNode::literal("1", DType::I32).unwrap());
    let f32_id = graph.add(OpNode::literal("1", DType::F32).unwrap());
    graph.rebuild();
    assert_ne!(i32_id, f32_id);
    assert!(
        matches_at(
            &graph,
            f32_id,
            &TensorPattern::literal("1", Some(DType::I32))
        )
        .is_empty()
    );
    assert_eq!(
        matches_at(
            &graph,
            i32_id,
            &TensorPattern::literal("1", Some(DType::I32))
        )
        .len(),
        1
    );
}

#[test]
fn integer_and_boolean_literal_ranges_are_validated_without_rounding() {
    for (value, dtype) in [
        ("-128", DType::I8),
        ("127", DType::I8),
        ("255", DType::U8),
        ("-9223372036854775808", DType::I64),
        ("9223372036854775807", DType::I64),
        ("18446744073709551615", DType::U64),
        ("00001", DType::Bool),
    ] {
        assert!(OpNode::literal(value, dtype).is_ok(), "{value}:{dtype}");
    }
    for (value, dtype) in [
        ("-129", DType::I8),
        ("128", DType::I8),
        ("256", DType::U8),
        ("-1", DType::U64),
        ("18446744073709551616", DType::U64),
        ("1.0", DType::I32),
        ("2", DType::Bool),
        ("-0", DType::Bool),
    ] {
        assert!(OpNode::literal(value, dtype).is_err(), "{value}:{dtype}");
    }
    for dtype in DType::ALL {
        assert_eq!(dtype.name().parse::<DType>().unwrap(), dtype);
    }
    assert!("float32".parse::<DType>().is_err());
}

#[test]
fn input_registration_is_immutable_and_conflicting_eclass_metadata_is_unavailable() {
    let mut bindings = TensorBindingTable::default();
    let f32 = info(&[4], DType::F32);
    let bf16 = info(&[4], DType::BF16);
    bindings.register_symbol("X", f32.clone()).unwrap();
    bindings.register_symbol("X", f32.clone()).unwrap();
    assert!(bindings.register_symbol("X", bf16.clone()).is_err());
    assert!(
        bindings
            .register_symbol("X", info(&[5], DType::F32))
            .is_err()
    );
    assert_eq!(bindings.info(&symbol("X")), Some(&f32));

    let (mut graph, [x, y, _]) = graph();
    graph.union(x, y);
    graph.rebuild();
    assert!(metadata([f32, bf16]).info(&graph, x).is_none());
}

#[test]
fn output_dtype_shape_and_missing_metadata_rejections_leave_no_partial_rhs() {
    let var = "?X".parse::<Var>().unwrap();
    for actual in [
        None,
        Some(info(&[4], DType::BF16)),
        Some(info(&[5], DType::F32)),
    ] {
        let mut graph = EGraph::<OpNode, ()>::default();
        let root = graph.add(symbol("X"));
        graph.rebuild();
        let expected = info(&[4], DType::F32);
        let rule = tensor_rewrite_checked(
            "invalid_output",
            TensorPattern::Var(var),
            TensorExpr::op(
                Op::TensorLang(tensor_lang::Op::Negate),
                AttrExpr::Exact(OpAttrs::None),
                vec![TensorExpr::Var(var)],
            ),
            move |_: &EGraph<OpNode, ()>, _: Id| Some(expected.clone()),
            move |_: Op, _: &[TensorInfo], _: &OpAttrs| actual.clone(),
            |_, _| Some(Default::default()),
        )
        .unwrap();
        apply(&mut graph, &rule, false);
        assert!(
            graph
                .lookup(unary(tensor_lang::Op::Negate, root).unwrap())
                .is_none()
        );
    }
    let (mut graph, _) = graph();
    let rule =
        rule_commute_f32::build_rewrite(|_: &EGraph<OpNode, ()>, _: Id| None, add_output, ())
            .unwrap();
    apply(&mut graph, &rule, false);
}

#[test]
fn invalid_derived_descriptor_is_rejected_before_any_intermediate_insertion() {
    let mut graph = EGraph::<OpNode, ()>::default();
    graph.add(symbol("X"));
    graph.rebuild();
    let var = "?X".parse::<Var>().unwrap();
    let rhs = TensorExpr::op(
        Op::TensorLang(tensor_lang::Op::Transpose),
        AttrExpr::Derived("t".into()),
        vec![TensorExpr::op(
            Op::TensorLang(tensor_lang::Op::Negate),
            AttrExpr::Exact(OpAttrs::None),
            vec![TensorExpr::Var(var)],
        )],
    );
    let rule = tensor_rewrite_checked(
        "invalid_transpose",
        TensorPattern::Var(var),
        rhs,
        |_: &EGraph<OpNode, ()>, _: Id| Some(info(&[2, 2], DType::F32)),
        |op: Op, inputs: &[TensorInfo], attrs: &OpAttrs| {
            let [input] = inputs else {
                return None;
            };
            match (op, attrs) {
                (Op::TensorLang(tensor_lang::Op::Negate), OpAttrs::None) => Some(input.clone()),
                (
                    Op::TensorLang(tensor_lang::Op::Transpose),
                    OpAttrs::TensorLang(tensor_lang::OpAttrs::TransposeAttrs { permutation }),
                ) => {
                    let mut sorted = permutation.clone();
                    sorted.sort_unstable();
                    if sorted != (0..u64::try_from(input.shape.len()).unwrap()).collect::<Vec<_>>()
                    {
                        return None;
                    }
                    Some(TensorInfo {
                        shape: permutation
                            .iter()
                            .map(|&axis| input.shape[usize::try_from(axis).unwrap()])
                            .collect(),
                        dtype: input.dtype,
                    })
                }
                _ => None,
            }
        },
        |_, _| {
            Some(
                [(
                    "t".into(),
                    OpAttrs::TensorLang(tensor_lang::OpAttrs::TransposeAttrs {
                        permutation: vec![0, 0],
                    }),
                )]
                .into(),
            )
        },
    )
    .unwrap();
    apply(&mut graph, &rule, false);
}

#[test]
fn literal_root_output_checks_use_the_literal_dtype() {
    let mut graph = EGraph::<OpNode, ()>::default();
    graph.add(OpNode::literal("1", DType::I32).unwrap());
    graph.rebuild();
    let rule = tensor_rewrite_checked(
        "wrong_literal_type",
        TensorPattern::literal("1", Some(DType::I32)),
        TensorExpr::literal("1", Some(DType::F32)),
        |_: &EGraph<OpNode, ()>, _: Id| Some(info(&[], DType::I32)),
        add_output,
        |_, _| Some(Default::default()),
    )
    .unwrap();
    apply(&mut graph, &rule, false);
    assert!(
        graph
            .lookup(OpNode::literal("1", DType::F32).unwrap())
            .is_none()
    );
}

#[test]
fn unsupported_intermediate_is_rejected_even_when_the_final_operation_is_supported() {
    let mut graph = EGraph::<OpNode, ()>::default();
    graph.add(symbol("X"));
    graph.rebuild();
    let var = "?X".parse::<Var>().unwrap();
    let rhs = TensorExpr::op(
        Op::TensorLang(tensor_lang::Op::Add),
        AttrExpr::Exact(OpAttrs::None),
        vec![
            TensorExpr::op(
                Op::TensorLang(tensor_lang::Op::Exponential),
                AttrExpr::Exact(OpAttrs::None),
                vec![TensorExpr::Var(var)],
            ),
            TensorExpr::Var(var),
        ],
    );
    let rule = tensor_rewrite_checked(
        "unsupported_intermediate",
        TensorPattern::Var(var),
        rhs,
        |_: &EGraph<OpNode, ()>, _: Id| Some(info(&[4], DType::I32)),
        |op: Op, inputs: &[TensorInfo], _: &OpAttrs| {
            // This integer host supports add but does not support exp.
            (op == Op::TensorLang(tensor_lang::Op::Add)).then(|| inputs[0].clone())
        },
        |_, _| Some(Default::default()),
    )
    .unwrap();
    apply(&mut graph, &rule, false);
}

#[test]
fn literal_inference_can_reject_formats_but_cannot_override_explicit_types() {
    for actual in [
        None,
        Some(info(&[], DType::I32)),
        Some(info(&[1], DType::F32)),
    ] {
        let mut graph = EGraph::<OpNode, ()>::default();
        graph.add(symbol("X"));
        graph.rebuild();
        let var = "?X".parse::<Var>().unwrap();
        let expected = actual.clone().unwrap_or_else(|| info(&[], DType::F32));
        let rule = tensor_rewrite_checked(
            "invalid_literal_inference",
            TensorPattern::Var(var),
            TensorExpr::literal("1.0", Some(DType::F32)),
            move |_: &EGraph<OpNode, ()>, _: Id| Some(expected.clone()),
            move |_: Op, _: &[TensorInfo], _: &OpAttrs| actual.clone(),
            |_, _| Some(Default::default()),
        )
        .unwrap();
        apply(&mut graph, &rule, false);
    }
}
