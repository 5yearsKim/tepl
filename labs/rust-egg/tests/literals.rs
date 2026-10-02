use egg::{EGraph, Id, Language};
use rust_egg::ir::patterns::{TensorExpr, TensorInfo, TensorPattern, matches_at, tensor_rewrite};
use rust_egg::ir::rules::basic::{rule_commute_float_literal, rule_commute_integer_literal};
use rust_egg::ir::{DType, NodeError, OpAttrs, OpKind, TensorLang};

#[test]
fn literals_preserve_kind_spelling_and_precision() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let values = [
        "1",
        "1.0",
        "1.00",
        "-0.0",
        "-0.5",
        "0.0",
        "+1",
        "001",
        "999999999999999999999999",
        "1.000000000000000000000001",
    ];
    let ids: Vec<_> = values
        .iter()
        .map(|value| {
            let node = TensorLang::literal(*value, DType::F64).unwrap();
            assert!(node.children().is_empty());
            assert_eq!(node.to_string(), format!("{value}:f64"));
            egraph.add(node)
        })
        .collect();
    egraph.rebuild();
    for (i, value) in values.iter().enumerate() {
        let pattern = TensorPattern::literal(*value, DType::F64).unwrap();
        for (j, id) in ids.iter().enumerate() {
            assert_eq!(
                matches_at(&egraph, *id, &pattern).len(),
                usize::from(i == j)
            );
        }
    }
}

#[test]
fn invalid_literals_and_literal_nodes_are_rejected() {
    for value in [
        "", "+", "-", "1.", ".5", "1e3", "1.0f", "NaN", "inf", "1.2.3", " 1", "1 ",
    ] {
        assert_eq!(
            TensorLang::literal(value, DType::F32),
            Err(NodeError::Attributes {
                op: OpKind::Literal
            })
        );
        assert!(TensorPattern::literal(value, DType::F32).is_err());
        assert!(TensorExpr::literal(value, DType::F32).is_err());
    }
    assert!(TensorLang::new(OpKind::Literal, vec![], OpAttrs::None).is_err());
    assert!(
        TensorLang::new(
            OpKind::Literal,
            vec![0.into()],
            OpAttrs::Literal {
                value: "1".into(),
                dtype: DType::I32
            }
        )
        .is_err()
    );
}

// Operation semantics for the checked basic literal rules: same-dtype add
// with rank-zero literal broadcasting, plus literal format support.
fn literal_output(op: OpKind, inputs: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
    match (op, inputs, attrs) {
        (OpKind::Literal, [], OpAttrs::Literal { dtype, .. }) => Some(TensorInfo {
            shape: vec![],
            dtype: *dtype,
        }),
        (OpKind::Add, [x, y], OpAttrs::None) if x.dtype == y.dtype => {
            let shape = if x.shape == y.shape || y.shape.is_empty() {
                x.shape.clone()
            } else if x.shape.is_empty() {
                y.shape.clone()
            } else {
                return None;
            };
            Some(TensorInfo {
                shape,
                dtype: x.dtype,
            })
        }
        _ => None,
    }
}

fn literal_metadata(
    graph: &EGraph<TensorLang, ()>,
    id: Id,
    input_dtype: DType,
) -> Option<TensorInfo> {
    let mut result = None;
    for node in &graph[graph.find(id)].nodes {
        let info = if node.symbol_name() == Some("X") {
            TensorInfo {
                shape: vec![4],
                dtype: input_dtype,
            }
        } else {
            let inputs = node
                .children()
                .iter()
                .map(|&child| literal_metadata(graph, child, input_dtype))
                .collect::<Option<Vec<_>>>()?;
            literal_output(node.op(), &inputs, node.attrs())?
        };
        if result.as_ref().is_some_and(|previous| previous != &info) {
            return None;
        }
        result = Some(info);
    }
    result
}

#[test]
fn reference_rules_match_only_their_literal_and_construct_the_rhs() {
    for (value, dtype) in [("1", DType::I32), ("1.0", DType::F32)] {
        let metadata =
            move |graph: &EGraph<TensorLang, ()>, id: Id| literal_metadata(graph, id, dtype);
        let rule = if dtype == DType::I32 {
            rule_commute_integer_literal::build_rewrite(metadata, literal_output).unwrap()
        } else {
            rule_commute_float_literal::build_rewrite(metadata, literal_output).unwrap()
        };
        let mut egraph = EGraph::<TensorLang, ()>::default();
        let x = egraph.add(TensorLang::symbol("X"));
        let mut expected = None;
        for input in ["1", "1.0", "1.00", "-0.5", "2"] {
            let literal = egraph.add(
                TensorLang::literal(
                    input,
                    if input.contains('.') {
                        DType::F32
                    } else {
                        DType::I32
                    },
                )
                .unwrap(),
            );
            let root = egraph.add(TensorLang::binary(OpKind::Add, x, literal).unwrap());
            if input == value {
                expected = Some((root, literal));
            }
        }
        // A named constant with numeric-looking text is still a different node.
        let named = egraph.add(TensorLang::constant(value));
        egraph.add(TensorLang::binary(OpKind::Add, x, named).unwrap());
        egraph.rebuild();
        let found = rule.search(&egraph);
        assert_eq!(found.iter().map(|m| m.substs.len()).sum::<usize>(), 1);
        let (root, literal) = expected.unwrap();
        assert_eq!(found[0].eclass, egraph.find(root));
        assert_eq!(rule.apply(&mut egraph, &found).len(), 1);
        egraph.rebuild();
        let rhs = egraph
            .lookup(TensorLang::binary(OpKind::Add, literal, x).unwrap())
            .unwrap();
        assert_eq!(egraph.find(root), egraph.find(rhs));
    }
}

#[test]
fn literals_can_be_rewrite_roots_and_insert_new_values() {
    let mut egraph = EGraph::<TensorLang, ()>::default();
    let root = egraph.add(TensorLang::literal("1", DType::I32).unwrap());
    egraph.rebuild();
    // Structural fixture only: the test callback authorizes this substitution.
    let rule = tensor_rewrite(
        "literal_root",
        TensorPattern::literal("1", DType::I32).unwrap(),
        TensorExpr::literal("-2.5", DType::F32).unwrap(),
        |_, _| Some(Default::default()),
    )
    .unwrap();
    let found = rule.search(&egraph);
    assert_eq!(found.len(), 1);
    rule.apply(&mut egraph, &found);
    egraph.rebuild();
    let output = egraph
        .lookup(TensorLang::literal("-2.5", DType::F32).unwrap())
        .unwrap();
    assert_eq!(egraph.find(root), egraph.find(output));
}
