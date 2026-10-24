#pragma once
#include "src/core/semantics/symbolic/normalize.h"
namespace tepl::core::attr_synth {

RuleProof synthesize(const Plan&, std::size_t max_work = 1000000);
}  // namespace tepl::core::attr_synth
