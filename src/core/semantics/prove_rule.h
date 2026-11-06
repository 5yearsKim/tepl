#pragma once

#include <chrono>

#include "src/core/semantics/ir.h"

namespace tepl::core::semantics {
// Optional benchmark accounting across all attempted strategies. Disabled in
// normal compilation; contexts and proof budgets are unchanged.
struct ProofTiming {
  double construction_ms = 0, verification_ms = 0;
  std::size_t construction_attempts = 0, verification_attempts = 0;
};

// Try synthesis strategies in order, independently verifying every candidate.
// The optional deadline is shared by all strategies and verification. Ordinary
// compiler calls retain the default work budget and have no wall-clock limit.
RuleProof proveRule(const Plan&, std::size_t max_work = 1000000,
                    ProofTiming* timing = nullptr,
                    std::chrono::steady_clock::time_point deadline =
                        std::chrono::steady_clock::time_point::max());

}  // namespace tepl::core::semantics
