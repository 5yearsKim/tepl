#pragma once

#include "src/core/dialect/symbols.h"

namespace tepl::core::detail {

struct AnalysisContext;

DialectRegistry checkDialects(AnalysisContext& context);

}  // namespace tepl::core::detail
