#include "src/core/shape/check.h"

#include <type_traits>
#include <utility>

#include "src/core/shape/expression.h"

namespace tepl::core::shape {

std::optional<Program> check(core::detail::AnalysisContext& analysis,
                             const Operation& operation,
                             const ast::ShapeDefinition& definition,
                             const std::string& source) {
  const auto before = analysis.diagnostics.size();
  detail::CheckContext context(analysis, operation, source);
  context.program.origin = context.origin(definition.span);
  if (definition.parameters.size() != operation.operands.size())
    context.report(
        context.program.origin,
        "shape parameter count must match operation operand signature");
  for (std::size_t i = 0; i < definition.parameters.size(); ++i) {
    const auto& parameter = definition.parameters[i];
    const auto origin = context.origin(parameter.span);
    if (i < operation.operands.size() &&
        parameter.variadic != operation.operands[i].variadic)
      context.report(
          origin,
          "shape variadic parameter must match operation variadic operand");
    const auto type = context.types.concrete(
        {Type::Kind::kInteger, parameter.variadic ? 2u : 1u}, origin);
    const auto symbol = context.symbol(parameter.name, type, origin);
    context.bind(parameter.name, symbol);
    context.program.parameters.push_back({symbol, i, parameter.variadic});
  }
  detail::ExpressionChecker expressions(context);
  for (const auto& statement : definition.statements) {
    const auto origin = context.origin(statement.span);
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, ast::ShapeLet>) {
            auto checked = expressions.check(*value.value);
            if (!checked) return;
            const auto symbol = context.symbol(value.name, checked->type,
                                               context.origin(value.name_span));
            context.bind(value.name, symbol);
            context.program.statements.push_back(
                {origin, Let{symbol, std::move(checked)}});
          } else {
            const auto boolean =
                context.types.concrete({Type::Kind::kBoolean}, origin);
            if (auto checked = expressions.check(*value.condition, boolean))
              context.program.statements.push_back(
                  {origin, Assert{std::move(checked)}});
          }
        },
        statement.value);
  }
  const auto shape = context.types.concrete(
      {Type::Kind::kInteger, 1}, context.origin(definition.result.span));
  context.program.result = {context.origin(definition.result.span),
                            expressions.check(*definition.result.value, shape)};
  if (analysis.diagnostics.size() != before) return std::nullopt;
  if (auto types = context.types.finish())
    context.program.types = std::move(*types);
  if (analysis.diagnostics.size() != before) return std::nullopt;
  return std::move(context.program);
}

}  // namespace tepl::core::shape
