use tepl_generated::components::generated::{
    OpNode,
    dialects::{egg, std},
};

fn main() {
    // The operation and its attributes must come from the same dialect.
    let _ = OpNode::new(
        std::Op::StdCopy,
        egg::OpAttrs::None,
        vec![::egg::Id::from(0)],
    );
}
