#pragma once

#include "src/core/semantics/ir.h"

namespace tepl::core::attr_synth {

enum class Strategy { Identity, Ordered, General };

// A constructed candidate still needs independent verification. Failure means
// this strategy could not construct a candidate, not that the rule is invalid.
// Steps, rewrites, and guard origins are diagnostics, never proof evidence.
struct Result {
  bool constructed = false;
  semantics::ConstructorCandidate candidate;
  std::string reason;
  std::vector<semantics::RuleProof::GuardOrigin> guard_origins;
  std::vector<std::string> steps;
  std::vector<semantics::RuleProof::RewriteStep> rewrites;
};

Result synthesize(const semantics::Plan&, Strategy,
                  std::size_t max_work = 1000000);

}  // namespace tepl::core::attr_synth
