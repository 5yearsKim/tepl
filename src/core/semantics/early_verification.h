#pragma once
#include <string>
#include <vector>

#include "src/core/semantics/solver.h"

namespace tepl::core::semantics {
// All results concern concrete metadata and ideal polynomial arithmetic.
enum class VerificationStatus {
  Counterexample,
  NoCounterexample,
  Skipped,
  Unknown
};
struct SparseEntry {
  std::size_t capture = 0;
  List coordinates;
  std::uint64_t value = 0;
};
struct VerificationResult {
  VerificationStatus status = VerificationStatus::Unknown;
  std::size_t degree = 0;
  std::string reason;
  List output;
  std::vector<SparseEntry> entries;
  std::string lhs_value, rhs_value;
};
struct MetadataSample {
  // Found is a legal LHS sample; Invalid means this bounded domain is
  // exhausted.
  Status status = Status::Unknown;
  std::vector<List> known, node_shapes;
  std::string reason;
};
// RHS fields are NOT constraints on sample selection. Shared source descriptors
// remain shared, and blocked samples exclude only exact source assignments.
MetadataSample sampleMetadata(
    const inference::SemanticPlan&, const std::vector<List>& shapes,
    const std::vector<std::vector<List>>& blocked = {},
    std::size_t budget = 100000);
VerificationResult verifySparse(const inference::SemanticPlan&,
                                const std::vector<List>& shapes,
                                const std::vector<List>& known,
                                const std::vector<List>& pinned = {},
                                std::size_t budget = 100000,
                                std::size_t max_degree = 8);
// Separate versioned ABI; ordinary match-time inference is unchanged.
std::string verificationRequest(const std::string&);
}  // namespace tepl::core::semantics
