#pragma once

#include "src/core/semantics/ir.h"

namespace tepl::core::semantics {

// Try synthesis strategies in order, independently verifying every candidate.
RuleProof proveRule(const Plan&, std::size_t max_work = 1000000);

}  // namespace tepl::core::semantics
