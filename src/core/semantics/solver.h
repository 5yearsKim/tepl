#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/inference/semantic_ir.h"
namespace tepl::core::semantics {
using List = std::vector<std::uint64_t>;
enum class Status { Found, Invalid, Unknown };
// Native measurements; not part of the generated application's wire format.
// Phase times can overlap: metadata_ms includes source legality solver checks.
struct SolveStatistics {
  double total_ms = 0, metadata_ms = 0, constraints_ms = 0;
  double z3_ms = 0, verification_ms = 0, ordered_ms = 0;
  std::size_t solver_calls = 0, candidates = 0, rejected_candidates = 0;
  bool constraint_attempted = false, constraint_fallback = false;
  bool preferred_attempted = false;
  std::string fallback_reason;
};
struct SolveOptions {
  bool semantic_constraints = true;
  bool preferred_candidate = true;
  unsigned random_seed = 0;
  // Zero means no candidate-count limit beyond the time/work budget.
  std::size_t max_candidates = 0;
};
struct Solution {
  Status status = Status::Unknown;
  std::vector<List> fields;
  std::string reason;
  // "constraints-z3" instantiates compile-time semantic matching obligations;
  // "ordered" constructs directly; "z3" uses metadata search and verification.
  std::string method = "z3";
  SolveStatistics statistics;
};
// Fixed metadata synthesis. Invalid means no witness for this metadata;
// it does not assert an arbitrary-rank coverage theorem.
Solution solve(const inference::SemanticPlan&, const std::vector<List>& shapes,
               const std::vector<List>& known,
               const std::vector<List>& pinned = {},
               std::size_t budget = 100000);
// Independent primitive/polynomial engine for differential checks and timing
// baselines. Generated applications use automatic dispatch through solve().
Solution solveGeneric(const inference::SemanticPlan&,
                      const std::vector<List>& shapes,
                      const std::vector<List>& known,
                      const std::vector<List>& pinned = {},
                      std::size_t budget = 100000,
                      const SolveOptions& options = {});
std::string encodePlan(const inference::SemanticPlan&);
std::string solveRequest(const std::string&);
}  // namespace tepl::core::semantics
