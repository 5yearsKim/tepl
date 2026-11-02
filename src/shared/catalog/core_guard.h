#pragma once
#include <functional>

#include "src/shared/catalog/completed.h"
namespace tepl::core::completed {
// Re-check the sufficient guard certificate from the actual completed rule.
// Only successful conjunctions, fixed capture ranks, and literal source fields
// contribute facts. Never inserts a runtime feasibility guard.
bool coreGuard(const RuleData&);
// Canonical same-dtype/numeric contracts; do not introduce a numeric premise.
bool coreTypes(const inference::Program&,
               const std::function<const metadata::Program*(std::size_t)>&);
}  // namespace tepl::core::completed
