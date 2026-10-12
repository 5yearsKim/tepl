#pragma once
#include "src/core/index_semantics/graph.h"
namespace tepl::core::index_semantics {
// A sufficient block-routing guard, derived from source ports and requested RHS
// intermediate capture sets. Groups retain arbitrary lengths. No names or fixed
// tensor ranks participate. This is a conservative fallback, not a weakest
// guard.
std::vector<S> routingGuard(const Plan&, std::size_t max_work);
}  // namespace tepl::core::index_semantics
