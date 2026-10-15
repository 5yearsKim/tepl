#include "src/core/checking/check.h"
#include "src/core/checking/context.h"
#include "src/core/checking/dialects/check.h"
#include "src/core/checking/graphs/check.h"
#include "src/core/checking/rules/check.h"
namespace tepl::core::checking {
Result<TypedProgram> check(const resolution::ResolvedProgram& input) {
  detail::CheckingContext context(input);
  detail::checkDialects(context);
  if (context.diagnostics.empty()) {
    for (const auto& rule : input.data().rules)
      if (auto checked = detail::check(context, rule))
        context.output.rules.push_back(std::move(*checked));
    detail::checkGraphs(context);
    if (context.diagnostics.empty())
      if (auto types = context.types.finish())
        context.output.types = std::move(*types);
  }
  Result<TypedProgram> result;
  result.diagnostics = std::move(context.diagnostics);
  if (result.diagnostics.empty())
    result.program = TypedProgram(std::move(context.output));
  return result;
}
}  // namespace tepl::core::checking
