#pragma once

#include <chrono>

#include "src/core/semantics/ir.h"
#include "src/inference/emission.h"

namespace tepl::core::attr_synth {

enum class Strategy { Identity, Ordered, General };

// A constructed candidate still needs independent verification. Failure means
// this strategy could not construct a candidate, not that the rule is invalid.
// Steps and guard origins are diagnostics, never proof evidence.
struct Result {
  bool constructed = false;
  // Guaranteed structural construction for admitted plans, before symbolic
  // normalization. Retained even if bounded expression specialization fails.
  std::optional<inference::Emission> metadata_program;
  semantics::SymbolPtr output_shape;
  semantics::ConstructorCandidate candidate;
  std::string reason;
  std::vector<semantics::RuleProof::GuardOrigin> guard_origins;
  std::vector<std::string> steps;
};

Result synthesize(const semantics::Plan&, Strategy,
                  std::size_t max_work = 1000000,
                  std::chrono::steady_clock::time_point deadline =
                      std::chrono::steady_clock::time_point::max());

}  // namespace tepl::core::attr_synth
