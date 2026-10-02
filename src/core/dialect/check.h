#pragma once

#include "src/core/dialect/symbols.h"

namespace tepl::core::detail {

struct AnalysisContext;

// Validate declarations, append checked IR, and retain names for scope
// building.
DialectRegistry checkDialects(AnalysisContext& context);

}  // namespace tepl::core::detail
