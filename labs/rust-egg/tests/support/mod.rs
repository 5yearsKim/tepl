use egg::{EGraph, Id};
use rust_egg::ir::pattern::TensorInfo;
use rust_egg::ir::{DType, Op, OpAttrs, OpNode};
// Structural fixtures give every operation and alternative the same scalar
// metadata. Separate semantic tests exercise real shape and dtype inference.
pub fn fixture_metadata(_: &EGraph<OpNode, ()>, _: Id) -> Option<TensorInfo> {
    Some(TensorInfo {
        shape: vec![],
        dtype: DType::F32,
    })
}
pub fn fixture_inference(_: Op, _: &[TensorInfo], _: &OpAttrs) -> Option<TensorInfo> {
    Some(TensorInfo {
        shape: vec![],
        dtype: DType::F32,
    })
}
