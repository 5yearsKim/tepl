#pragma once

#include "src/ast/ast.h"
#include "src/core/ir.h"

namespace tepl::core::detail {

struct RuleCheckContext;

void collectCaptures(RuleCheckContext& context, const ast::GraphExpr& node);
PatternPtr checkPattern(RuleCheckContext& context, const ast::GraphExpr& node);
BuildExprPtr checkBuild(RuleCheckContext& context, const ast::GraphExpr& node);

}  // namespace tepl::core::detail
