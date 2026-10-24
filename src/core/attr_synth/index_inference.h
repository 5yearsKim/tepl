#pragma once

#include <optional>

#include "src/core/semantics/symbolic/index_expr.h"

namespace tepl::core::attr_synth {
using semantics::Context;
using semantics::I;
using semantics::S;

using IndexAssignments = std::map<std::pair<std::size_t, std::size_t>, S>;
// Match scoped coordinate bases and solve whole, per-axis coefficient/offset
// holes. Returns no match for a different access. Never installs facts or
// attributes. Coupled/inverse equations and ambiguous basis correspondences are
// unsupported.
std::optional<IndexAssignments> matchIndexAttributes(Context&, const I& pattern,
                                                     const I& source);
}  // namespace tepl::core::attr_synth
