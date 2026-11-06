#pragma once

#include "src/core/checking/ir.h"
#include "src/core/syntax/ast.h"

namespace tepl::core::detail {

struct RuleCheckContext;

void collectCaptures(RuleCheckContext& context, const syntax::GraphExpr& node);
PatternPtr checkPattern(RuleCheckContext& context,
                        const syntax::GraphExpr& node);
BuildExprPtr checkBuild(RuleCheckContext& context,
                        const syntax::GraphExpr& node);

}  // namespace tepl::core::detail
