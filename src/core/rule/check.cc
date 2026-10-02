#include "src/core/rule/check.h"

#include <utility>

#include "src/core/rule/check_context.h"
#include "src/core/rule/declarations.h"
#include "src/core/rule/descriptors.h"
#include "src/core/rule/expression.h"
#include "src/core/rule/graph.h"

namespace tepl::core::detail {

namespace {

void checkConditions(RuleCheckContext& context,
                     ExpressionChecker& expressions) {
  for (const auto& condition : context.input.conditions) {
    const auto& at = context.input.expr_origins.at(condition.get());
    if (auto checked = expressions.check(
            *condition, context.analysis.types.concrete({TypeKind::kBool}, at)))
      context.rule.conditions.push_back(std::move(checked));
  }
}

}  // namespace

std::optional<Rule> check(AnalysisContext& analysis,
                          const ExpandedRule& input) {
  RuleCheckContext context(analysis, input);
  const auto diagnostics_before = analysis.diagnostics.size();
  collectCaptures(context, *input.lhs);
  checkDeclarations(context);
  context.rule.lhs = checkPattern(context, *input.lhs);
  registerDerivations(context);
  // RHS uses constrain descriptor schemas before expression checking.
  context.rule.rhs = checkBuild(context, *input.rhs);
  ExpressionChecker expressions(analysis, input, context.rule, context.scope);
  checkConditions(context, expressions);
  checkDerivations(context, expressions);
  if (analysis.diagnostics.size() != diagnostics_before) return std::nullopt;
  analysis.types.unify(context.rule.lhs->type, context.rule.rhs->type,
                       input.origin);
  if (analysis.diagnostics.size() != diagnostics_before) return std::nullopt;
  return std::move(context.rule);
}

}  // namespace tepl::core::detail
