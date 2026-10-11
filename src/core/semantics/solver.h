#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/inference/semantic_ir.h"
namespace tepl::core::semantics {
using List = std::vector<std::uint64_t>;
enum class Status { Found, Invalid, Unknown };
struct Solution {
  Status status = Status::Unknown;
  std::vector<List> fields;
  std::string reason;
};
// Fixed metadata synthesis. Invalid means no witness for this metadata;
// it does not assert an arbitrary-rank coverage theorem.
Solution solve(const inference::SemanticPlan&, const std::vector<List>& shapes,
               const std::vector<List>& known, const std::vector<List>& pinned,
               std::size_t budget = 100000);
std::string encodePlan(const inference::SemanticPlan&);
std::string solveRequest(const std::string&);
}  // namespace tepl::core::semantics
