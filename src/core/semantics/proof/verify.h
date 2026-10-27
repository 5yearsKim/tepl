#pragma once
#include <chrono>

#include "src/core/semantics/ir.h"
namespace tepl::core::semantics {
// Verify constructors using only declared conditions and LHS legality.
// Candidate guards are obligations, never assumptions.
RuleProof verify(const Plan&, const ConstructorCandidate&,
                 std::size_t max_work = 1000000,
                 std::chrono::steady_clock::time_point deadline =
                     std::chrono::steady_clock::time_point::max());
}  // namespace tepl::core::semantics
