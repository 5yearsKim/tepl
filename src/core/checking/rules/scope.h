#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>

#include "src/shared/catalog/ids.h"

namespace tepl::core::detail {

// Graph and declaration checking introduce symbols. Expression checking only
// reads this scope; descriptor availability advances after each derivation.
struct NodeBinding {
  OpId operation;
  std::optional<DescriptorId> descriptor;
  bool rhs = false;
};
struct RuleScope {
  std::map<std::string, NodeBinding> nodes;
  std::map<std::string, CaptureId> captures;
  std::map<std::string, DimensionId> dimensions;
  std::map<std::string, DTypeVariableId> dtypes;
  std::map<std::string, DescriptorId> attribute_definitions;
  std::set<std::size_t> available_descriptors;
};

}  // namespace tepl::core::detail
