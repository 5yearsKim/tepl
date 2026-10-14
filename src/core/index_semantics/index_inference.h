#pragma once

#include <optional>

#include "src/core/index_semantics/index_expr.h"

namespace tepl::core::index_semantics {
using IndexAssignments = std::map<std::pair<std::size_t, std::size_t>, S>;
// Match scoped coordinate bases and solve whole, per-axis coefficient/offset
// holes. Returns no match for a different access. Never installs facts or
// attributes. Coupled/inverse equations and ambiguous basis correspondences are
// unsupported.
std::optional<IndexAssignments> matchIndexAttributes(Context&, const I& pattern,
                                                     const I& source);
}  // namespace tepl::core::index_semantics
