#pragma once

#include <optional>

#include "src/ast/ast.h"
#include "src/core/rule/expanded_rule.h"

namespace tepl::core::detail {

struct AnalysisContext;

std::optional<ExpandedRule> expand(AnalysisContext& context,
                                   const ast::Rule& rule);

}  // namespace tepl::core::detail
