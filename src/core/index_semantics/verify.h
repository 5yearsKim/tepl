#pragma once
#include "src/core/index_semantics/normalize.h"
namespace tepl::core::index_semantics {
RuleProof verify(const Plan&, const RuleProof&, std::size_t max_work = 1000000);
// Compiler-side verification-condition generation for a fixed constructor.
// Produces metadata obligations, then the public verifier checks that guard.
RuleProof completeGuard(const Plan&, const RuleProof&,
                        std::size_t max_work = 1000000);
}  // namespace tepl::core::index_semantics
