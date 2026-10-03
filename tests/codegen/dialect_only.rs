use tepl_generated::ir::OpNode;
use tepl_generated::ir::dialects::__::{Op, OpAttrs};

#[test]
fn keyword_fields_use_raw_identifiers_without_renaming_other_fields() {
    let attrs = OpAttrs::Tepl {
        self_: 7,
        r#type: vec!["tensor".into()],
        type_: 11,
        r#match: 13,
        r#gen: 17,
    };
    let OpAttrs::Tepl {
        self_,
        r#type,
        type_,
        r#match,
        r#gen,
    } = &attrs
    else {
        panic!("unexpected schema")
    };
    assert_eq!((*self_, *type_, *r#match, *r#gen), (7, 11, 13, 17));
    assert_eq!(r#type, &["tensor"]);
    assert!(OpNode::new(Op::Tepl1, attrs, vec![]).is_ok());
}
