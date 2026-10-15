#pragma once
#include "../types.h"

namespace @TEPL_NAMESPACE@::analysis {
// Shape and dtype shared by graph construction, analysis, and rewrites.
struct TensorInfo {
  ::std::vector<::std::uint64_t> shape;
  DType dtype;
  bool operator==(const TensorInfo&) const = default;
};
}  // namespace @TEPL_NAMESPACE@::analysis

namespace @TEPL_NAMESPACE@ {
using analysis::TensorInfo;
}  // namespace @TEPL_NAMESPACE@
