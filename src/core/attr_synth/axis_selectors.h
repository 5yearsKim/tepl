#pragma once

#include <optional>

#include "src/core/semantics/symbolic/index_expr.h"

namespace tepl::core::attr_synth {
using semantics::Context;
using semantics::I;
using semantics::S;

using AxisAssignments = std::map<std::pair<std::size_t, std::size_t>, S>;
// Infer axis selectors (currently reverse dimensions) from scoped reads.
// All offsets and coefficients must already be supplied; no non-axis inference.
std::optional<AxisAssignments> matchAxisSelectors(Context&, const I& pattern,
                                                  const I& source);
}  // namespace tepl::core::attr_synth
