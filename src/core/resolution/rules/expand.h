#pragma once

#include <optional>

#include "src/core/resolution/rules/expanded_rule.h"
#include "src/core/syntax/ast.h"

namespace tepl::core::detail {

struct ResolutionContext;

std::optional<ExpandedRule> expand(ResolutionContext& context,
                                   const syntax::Rule& rule);

}  // namespace tepl::core::detail
