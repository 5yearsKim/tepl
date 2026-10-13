#include "src/core/rule/check.h"

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
      if (auto b = std::get_if<BinaryExpr>(&e.value))
        return hasGroup(*b->lhs) || hasGroup(*b->rhs);
      if (auto u = std::get_if<UnaryExpr>(&e.value))
        return hasGroup(*u->operand);
      if (auto b = std::get_if<BuiltinCall>(&e.value))
        for (const auto& a : b->arguments)
          if (hasGroup(*a)) return true;
      if (auto h = std::get_if<HostCall>(&e.value))
        for (const auto& a : h->arguments)
          if (hasGroup(*a)) return true;
      return false;
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
