#pragma once
#include <string>
#include <utility>
namespace tepl::core {
enum class SemanticPolicy { kStrict, kAlgebraic };
struct SemanticOptions {
  SemanticPolicy semantic_policy = SemanticPolicy::kStrict;
  // Explicitly permit match-time structural applicability checks. Declared
  // conditions retain their existing universal-coverage meaning by default.
  bool inferred_applicability = false;
};

namespace semantics {
struct InspectionOptions : SemanticOptions {
  bool inspect_semantic_rules = false;
  std::string semantic_rule;
  InspectionOptions() = default;
  InspectionOptions(SemanticPolicy p, bool inspect = false,
                    std::string rule = {}, bool inferred = false)
      : SemanticOptions{p, inferred},
        inspect_semantic_rules(inspect),
        semantic_rule(std::move(rule)) {}
};
}  // namespace semantics
}  // namespace tepl::core
