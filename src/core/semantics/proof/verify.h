#pragma once
#include "src/core/semantics/symbolic/normalize.h"
namespace tepl::core::semantics {
// Verify constructors using only declared conditions and LHS legality.
// Candidate guards are obligations, never assumptions.
RuleProof verify(const Plan&, const ConstructorCandidate&,
                 std::size_t max_work = 1000000);
}  // namespace tepl::core::semantics
