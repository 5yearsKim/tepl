#pragma once
#include "src/core/semantics/graph.h"
namespace tepl::core::semantics {
bool equivalentValue(Context&, const E&, const E&);
E normalizeScalar(Context&, const E&);
void scalarProductLaws(Context&, std::vector<E>&);
}  // namespace tepl::core::semantics
