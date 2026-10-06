use super::super::DType;

/// Concrete shape and dtype shared by graph construction, analysis, and rewrites.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<u64>,
    pub dtype: DType,
}
