#pragma once
#include "src/core/index_semantics/normalize.h"
namespace tepl::core::index_semantics {
RuleProof synthesize(const Plan&, std::size_t max_work = 1000000);
}  // namespace tepl::core::index_semantics
