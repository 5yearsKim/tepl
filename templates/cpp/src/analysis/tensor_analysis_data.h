#pragma once
#include <eggc/analysis.hpp>

#include "inference.h"
#include "tensor_info.h"

namespace @TEPL_NAMESPACE@::analysis {
struct TensorAnalysisData {
  ::std::optional<TensorInfo> observed;
  bool conflict = false, has_unknown = false, has_invalid = false;
  bool operator==(const TensorAnalysisData&) const = default;
  static TensorAnalysisData from_inference(const Inference<TensorInfo>& value) {
    return {value.into_option(), false,
            value.kind == Inference<TensorInfo>::Kind::Unknown,
            value.kind == Inference<TensorInfo>::Kind::Invalid};
  }
  bool is_invalid() const { return conflict || has_invalid; }
  ::std::optional<TensorInfo> info() const {
    return has_unknown || is_invalid() ? ::std::nullopt : observed;
  }
  bool is_unknown() const { return !is_invalid() && !info(); }
  ::eggc::AnalysisMerge merge(const TensorAnalysisData& other) {
    auto previous = *this;
    conflict = conflict || other.conflict ||
               (observed && other.observed && observed != other.observed);
    if (conflict)
      observed.reset();
    else if (!observed)
      observed = other.observed;
    has_unknown |= other.has_unknown;
    has_invalid |= other.has_invalid;
    // A conflicting observation is accumulated evidence, not a rejected union.
    return previous == *this ? ::eggc::AnalysisMerge::Unchanged
                             : ::eggc::AnalysisMerge::Changed;
  }
};
}  // namespace @TEPL_NAMESPACE@::analysis
