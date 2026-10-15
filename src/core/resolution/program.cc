#include <functional>

#include "src/core/resolution/context.h"
#include "src/core/resolution/resolve.h"
#include "src/core/resolution/resolve_context.h"
#include "src/core/resolution/rules/expand.h"
#include "src/core/syntax/snapshot.h"
#include "src/shared/source/identity.h"
namespace tepl::core::resolution {
Result<ResolvedProgram> resolve(core::resolution::SourceProject input) {
  // Detach even manually constructed ASTs from externally retained aliases.
  for (auto& module : input.modules) module = syntax::snapshot(module);
  syntax::ParsedModule selected;
  selected.rules = std::move(input.declarations.rules);
  selected.dialects = std::move(input.declarations.dialects);
  selected = syntax::snapshot(selected);
  input.declarations.rules = std::move(selected.rules);
  input.declarations.dialects = std::move(selected.dialects);
  for (auto& rule : input.imported_rules) rule = syntax::snapshot(rule);
  ResolvedData data;
  auto sources =
      std::make_shared<core::resolution::SourceProject>(std::move(input));
  data.sources = sources;
  detail::ResolutionContext context(*data.sources);
  detail::resolve(context);
  if (context.diagnostics.empty())
    for (const auto& rule : data.sources->declarations.rules) {
      if (rule.is_abstract) continue;
      if (auto expanded = detail::expand(context, rule))
        data.rules.push_back(std::move(*expanded));
    }
  std::size_t next_concrete = 0;
  for (auto& graph : sources->declarations.graphs) {
    std::function<void(syntax::ConcreteExpr&)> visit = [&](auto& expression) {
      expression.resolution_id = next_concrete++;
      if (expression.kind == syntax::ConcreteExpr::Kind::kOperation)
        if (auto op = context.operation(
                expression.name, graph.source_name,
                detail::origin(graph.source_name, expression.span)))
          data.concrete_operations.emplace(expression.resolution_id, *op);
      for (auto& operand : expression.operands) visit(operand);
    };
    for (auto& binding : graph.bindings) visit(binding.expression);
    visit(graph.output);
  }
  data.source_names = context.output.sources;
  data.declarations = std::move(context.output.operations);
  for (const auto& [source, scope] : context.scopes)
    data.operations.emplace(source, scope.operations);
  Result<ResolvedProgram> result;
  result.diagnostics = std::move(context.diagnostics);
  if (result.diagnostics.empty())
    result.program = ResolvedProgram(std::move(data));
  return result;
}
}  // namespace tepl::core::resolution
