#pragma once
#include "../types.h"

namespace @TEPL_NAMESPACE@ {
struct TensorInfo {
  ::std::vector<::std::uint64_t> shape;
  DType dtype;
  bool operator==(const TensorInfo&) const = default;
};
namespace analysis {
using ::@TEPL_NAMESPACE@::TensorInfo;
}
namespace pattern {
using ::@TEPL_NAMESPACE@::TensorInfo;
}
}  // namespace @TEPL_NAMESPACE@
