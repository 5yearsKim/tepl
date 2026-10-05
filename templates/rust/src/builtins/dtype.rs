//! Predicates over tensor element types; dtype programs perform inference.
use super::super::DType;
pub fn is_float(t: DType) -> bool {
    matches!(t, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
}
pub fn is_signed_integer(t: DType) -> bool {
    matches!(t, DType::I8 | DType::I16 | DType::I32 | DType::I64)
}
pub fn is_unsigned_integer(t: DType) -> bool {
    matches!(t, DType::U8 | DType::U16 | DType::U32 | DType::U64)
}
pub fn is_integer(t: DType) -> bool {
    is_signed_integer(t) || is_unsigned_integer(t)
}
pub fn is_numeric(t: DType) -> bool {
    is_integer(t) || is_float(t)
}
