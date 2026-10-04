//! Common dtype policies; operation dispatch is generated from checked TEPL.
use super::super::DType;
use super::Inference;

pub fn same(operands: &[DType]) -> Inference<DType> {
    let Some(&first) = operands.first() else {
        return Inference::Unknown;
    };
    if operands.iter().all(|&dtype| dtype == first) {
        Inference::Known(first)
    } else {
        Inference::Invalid("operand dtypes must match")
    }
}

pub fn same_numeric(operands: &[DType]) -> Inference<DType> {
    match same(operands) {
        Inference::Known(DType::Bool) => Inference::Invalid("numeric policy rejects bool"),
        result => result,
    }
}

pub fn same_float(operands: &[DType]) -> Inference<DType> {
    match same(operands) {
        Inference::Known(dtype)
            if matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) =>
        {
            Inference::Known(dtype)
        }
        Inference::Known(_) => Inference::Invalid("float policy requires floating-point operands"),
        result => result,
    }
}
