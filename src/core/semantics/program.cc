#include "src/core/checking/program.h"

#include <functional>
#include <stdexcept>

#include "src/core/semantics/compilation_context.h"
#include "src/core/semantics/inspect.h"
#include "src/core/semantics/materialize.h"
#include "src/core/semantics/verify.h"
namespace tepl::core::semantics {
Result<InspectionData> inspect(const checking::TypedProgram& input,
                               const InspectionOptions& options) {
  InspectionData output;
  static_cast<ProgramTables&>(output) =
      static_cast<const ProgramTables&>(input.ir());
  for (const auto& rule : input.ir().rules) {
    InspectedRule inspected;
    static_cast<Rule&>(inspected) = rule;
    output.rules.push_back(std::move(inspected));
  }
  CompilationContext context{output, options, {}};
  for (auto& rule : output.rules) {
    if (compile(context, rule)) continue;
    if (rule.requiresSynthesis())
      context.report(
          rule.origin,
          "auto requires operations with axes and semantics definitions");
    std::function<bool(const TypedExpr&)> group = [&](const TypedExpr& e) {
      bool found = std::holds_alternative<OutputAxisGroupRef>(e.value);
      forEachExpressionChild(e, [&](const TypedExpr& c) { found |= group(c); });
      return found;
    };
    for (const auto& e : rule.conditions)
      if (group(*e))
        context.report(
            e->origin,
            "output-axis group reference requires axes and semantics");
  }
  Result<InspectionData> result;
  result.diagnostics = std::move(context.diagnostics);
  if (result.diagnostics.empty()) result.program = std::move(output);
  return result;
}
Result<VerifiedProgram> verify(const checking::TypedProgram& input,
                               const SemanticOptions& options) {
  auto inspected = inspect(input, {options.semantic_policy, false, {}});
  Result<VerifiedProgram> result;
  result.diagnostics = std::move(inspected.diagnostics);
  if (!inspected.program) return result;
  try {
    auto completed = materialize(std::move(*inspected.program));
    result.program =
        VerifiedProgram(std::move(completed.ir), options.semantic_policy,
                        std::move(completed.axes), std::move(completed.kinds));
  } catch (const std::exception& e) {
    result.diagnostics.push_back({{}, e.what(), {}});
  }
  return result;
}
}  // namespace tepl::core::semantics
