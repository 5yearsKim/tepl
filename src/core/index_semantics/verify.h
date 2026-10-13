#pragma once
#include "src/core/index_semantics/normalize.h"
namespace tepl::core::index_semantics {
// Verify constructors using only declared conditions and LHS legality.
// Candidate guards are obligations, never assumptions.
RuleProof verify(const Plan&, const RuleProof&, std::size_t max_work = 1000000);
}  // namespace tepl::core::index_semantics
