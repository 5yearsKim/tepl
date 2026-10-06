#![allow(dead_code)]
use egg::{EGraph, Id};
use tepl_generated::ir::pattern::TensorInfo;
use tepl_generated::ir::{DType, Op, OpAttrs, OpNode};
// Structural fixtures give every operation and alternative the same scalar
// metadata. Separate semantic tests exercise real shape and dtype inference.
pub fn fixture_metadata<N: egg::Analysis<OpNode>>(
    _: &EGraph<OpNode, N>,
    _: Id,
) -> Option<TensorInfo> {
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

use egg::{Analysis, DidMerge, Rewrite};
use std::sync::Arc;
use tepl_generated::ir::pattern::{OutputInference, RewriteAnalysis, TensorMetadata};

// A test analysis keeps controllable observations and output policies on the
// graph. Generated rules access them through the same interface as real analyses.
#[derive(Clone)]
pub struct TestAnalysis {
    reader: Arc<dyn TensorMetadata<TestAnalysis>>,
    inference: Arc<dyn OutputInference>,
}
impl Default for TestAnalysis {
    fn default() -> Self {
        Self {
            reader: Arc::new(fixture_metadata::<Self>),
            inference: Arc::new(fixture_inference),
        }
    }
}
impl Analysis<OpNode> for TestAnalysis {
    type Data = ();
    fn make(_: &mut EGraph<OpNode, Self>, _: &OpNode, _: Id) {}
    fn merge(&mut self, _: &mut (), _: ()) -> DidMerge {
        DidMerge(false, false)
    }
}
impl RewriteAnalysis for TestAnalysis {
    const HAS_TENSOR_INFO: bool = true;
    fn tensor_info(graph: &EGraph<OpNode, Self>, id: Id) -> Option<TensorInfo> {
        graph.analysis.reader.info(graph, id)
    }
    fn infer_output(&self, op: Op, operands: &[TensorInfo], attrs: &OpAttrs) -> Option<TensorInfo> {
        self.inference.infer_output(op, operands, attrs)
    }
    fn infer_literal(
        &self,
        value: &str,
        dtype: Option<DType>,
        expected: &TensorInfo,
    ) -> Option<DType> {
        self.inference.infer_literal(value, dtype, expected)
    }
}
pub fn configure_rule<M, I>(
    graph: &mut EGraph<OpNode, TestAnalysis>,
    reader: M,
    inference: I,
    rule: Result<Rewrite<OpNode, TestAnalysis>, String>,
) -> Result<Rewrite<OpNode, TestAnalysis>, String>
where
    M: TensorMetadata<TestAnalysis> + 'static,
    I: OutputInference + 'static,
{
    graph.analysis.reader = Arc::new(reader);
    graph.analysis.inference = Arc::new(inference);
    rule
}
