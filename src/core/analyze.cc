#include "src/core/analyze.h"

#include <utility>

#include "src/core/analysis_context.h"

namespace tepl::core {

AnalysisResult analyze(const ast::Program& input) {
  detail::AnalysisContext context(input);
  detail::resolve(context);
  if (context.diagnostics.empty()) {
    for (const auto& rule : input.rules) {
      if (rule.is_abstract) continue;
      auto expanded = detail::expand(context, rule);
      if (!expanded) continue;
      if (auto checked = detail::check(context, *expanded))
        context.output.rules.push_back(std::move(*checked));
    }
    if (context.diagnostics.empty()) {
      if (auto types = context.types.finish())
        context.output.types = std::move(*types);
    }
  }
  AnalysisResult result;
  result.diagnostics = std::move(context.diagnostics);
  if (result.diagnostics.empty()) result.program = std::move(context.output);
  return result;
}

}  // namespace tepl::core
