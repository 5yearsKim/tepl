#pragma once

#include <optional>

#include "src/core/ir.h"

namespace tepl::core::detail {

struct AnalysisContext;
struct ExpandedRule;

std::optional<Rule> check(AnalysisContext& context, const ExpandedRule& input);

}  // namespace tepl::core::detail
