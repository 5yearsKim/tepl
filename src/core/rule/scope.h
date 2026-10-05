#pragma once

#include <map>
#include <set>
#include <string>

#include "src/core/ids.h"

namespace tepl::core::detail {

// Graph and declaration checking introduce symbols. Expression checking only
// reads this scope; descriptor availability advances after each derivation.
struct RuleScope {
  std::map<std::string, CaptureId> captures;
  std::map<std::string, DimensionId> dimensions;
  std::map<std::string, DTypeVariableId> dtypes;
  std::map<std::string, DescriptorId> descriptors;
  std::set<std::size_t> available_descriptors;
};

}  // namespace tepl::core::detail
