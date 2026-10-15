#include "src/core/rule/check.h"

#include <algorithm>
#include <functional>
#include <utility>

#include "src/core/index_semantics/check.h"
#include "src/core/rule/check_context.h"
#include "src/core/rule/declarations.h"
#include "src/core/rule/descriptors.h"
#include "src/core/rule/expression.h"
#include "src/core/rule/graph.h"

namespace tepl::core::detail {

namespace {

void checkConditions(RuleCheckContext& context,
                     ExpressionChecker& expressions) {
  for (std::size_t i = 0; i < context.input.conditions.size(); ++i) {
    if (std::find(context.input.condition_scopes.begin(),
                  context.input.condition_scopes.end(),
                  i) != context.input.condition_scopes.end())
      expressions.resetLocals();
    const auto& condition = context.input.conditions[i];
    const auto& at = context.input.expr_origins.at(condition.get());
    if (auto local = std::get_if<ast::LocalBinding>(&condition->value)) {
      if (auto checked = expressions.local(local->name, *local->value, at))
        context.rule.conditions.push_back(checked);
      continue;
    }
    if (auto checked = expressions.check(
            *condition, context.analysis.types.concrete({TypeKind::kBool}, at)))
      context.rule.conditions.push_back(std::move(checked));
  }
}

}  // namespace

std::optional<Rule> check(AnalysisContext& analysis,
                          const ExpandedRule& input) {
  RuleCheckContext context(analysis, input);
  context.rule.source_name = input.source_name;
  const auto diagnostics_before = analysis.diagnostics.size();
  collectCaptures(context, *input.lhs);
  checkDeclarations(context);
  context.rule.lhs = checkPattern(context, *input.lhs);
  registerDerivations(context);
  // RHS uses constrain descriptor schemas before expression checking.
  context.rule.rhs = checkBuild(context, *input.rhs);
  ExpressionChecker expressions(analysis, input, context.rule, context.scope);
  checkConditions(context, expressions);
  ExpressionChecker derivations(analysis, input, context.rule, context.scope,
                                builtins::Context::kDerive);
  checkDerivations(context, derivations);
  if (analysis.diagnostics.size() != diagnostics_before) return std::nullopt;
  analysis.types.unify(context.rule.lhs->type, context.rule.rhs->type,
                       input.origin);
  if (analysis.diagnostics.size() != diagnostics_before) return std::nullopt;
  if (!index_semantics::compile(analysis, context.rule)) {
    std::function<bool(const TypedExpr&)> hasGroup = [&](const TypedExpr& e) {
      if (std::holds_alternative<OutputAxisGroupRef>(e.value)) return true;
      bool found = false;
      forEachExpressionChild(
          e, [&](const TypedExpr& child) { found |= hasGroup(child); });
      return found;
    };
    for (const auto& e : context.rule.conditions)
      if (hasGroup(*e))
        analysis.report(
            e->origin,
            "output-axis group reference requires axes and semantics");
    if (!context.rule.attribute_holes.empty())
      analysis.report(
          input.origin,
          "auto requires operations with axes and semantics definitions");
  }
  if (analysis.diagnostics.size() != diagnostics_before) return std::nullopt;
  return std::move(context.rule);
}

}  // namespace tepl::core::detail
