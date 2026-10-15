#pragma once
#include "../types.h"
namespace @TEPL_NAMESPACE@::builtins::dtype {
inline bool is_float(DType t) {
  return t == DType::F16 || t == DType::BF16 || t == DType::F32 ||
         t == DType::F64;
}
inline bool is_signed_integer(DType t) {
  return t == DType::I8 || t == DType::I16 || t == DType::I32 ||
         t == DType::I64;
}
inline bool is_unsigned_integer(DType t) {
  return t == DType::U8 || t == DType::U16 || t == DType::U32 ||
         t == DType::U64;
}
inline bool is_integer(DType t) {
  return is_signed_integer(t) || is_unsigned_integer(t);
}
inline bool is_numeric(DType t) { return is_integer(t) || is_float(t); }
}  // namespace @TEPL_NAMESPACE@::builtins::dtype
