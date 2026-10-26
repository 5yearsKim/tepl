#pragma once

#include "src/core/semantics/ir.h"

namespace tepl::core::semantics {
// Optional benchmark accounting across all attempted strategies. Disabled in
// normal compilation; contexts and proof budgets are unchanged.
struct ProofTiming {
  double construction_ms = 0, verification_ms = 0;
  std::size_t construction_attempts = 0, verification_attempts = 0;
};

// Try synthesis strategies in order, independently verifying every candidate.
RuleProof proveRule(const Plan&, std::size_t max_work = 1000000,
                    ProofTiming* timing = nullptr);

}  // namespace tepl::core::semantics
