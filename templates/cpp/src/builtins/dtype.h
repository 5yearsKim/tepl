#pragma once
#include "../analysis/inference.h"
#include "../types.h"

namespace @TEPL_NAMESPACE@::builtins::dtype {
inline analysis::Inference<DType> same(::std::span<const DType> operands) {
  if (operands.empty()) return analysis::Inference<DType>::unknown();
  for (auto d : operands)
    if (d != operands.front())
      return analysis::Inference<DType>::invalid("incompatible dtypes");
  return analysis::Inference<DType>::known(operands.front());
}
inline analysis::Inference<DType> same_numeric(
    ::std::span<const DType> operands) {
  for (auto d : operands)
    if (d == DType::Bool)
      return analysis::Inference<DType>::invalid("expected numeric dtype");
  return same(operands);
}
inline analysis::Inference<DType> same_float(
    ::std::span<const DType> operands) {
  for (auto d : operands)
    if (d < DType::F16)
      return analysis::Inference<DType>::invalid("expected floating dtype");
  return same(operands);
}
}  // namespace @TEPL_NAMESPACE@::builtins::dtype
