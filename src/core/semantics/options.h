#pragma once
#include <string>
#include <utility>
namespace tepl::core {
enum class SemanticPolicy { kStrict, kAlgebraic };
struct SemanticOptions {
  SemanticPolicy semantic_policy = SemanticPolicy::kStrict;
};

namespace semantics {
struct InspectionOptions : SemanticOptions {
  bool inspect_semantic_rules = false;
  std::string semantic_rule;
  InspectionOptions() = default;
  InspectionOptions(SemanticPolicy p, bool inspect = false,
                    std::string rule = {})
      : SemanticOptions{p},
        inspect_semantic_rules(inspect),
        semantic_rule(std::move(rule)) {}
};
}  // namespace semantics
}  // namespace tepl::core
