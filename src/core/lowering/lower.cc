#include "src/core/lowering/lower.h"

#include <stdexcept>
namespace tepl::core::lowering {
struct Builder {
  static executable::ExecutableProgram run(
      const semantics::VerifiedProgram& input) {
    const auto& checked = input.ir();
    executable::ExecutableProgram output;
    output.ir_.dialects = checked.dialects;
    output.ir_.sources = checked.sources;
    output.ir_.types = checked.types;
    for (const auto& operation : checked.operations) {
      executable::Operation lowered;
      static_cast<OperationSignature&>(lowered) =
          static_cast<const OperationSignature&>(operation);
      if (operation.semantic_definition)
        lowered.axis_contract = std::make_shared<const axis::ShapeContract>(
            *operation.semantic_definition);
      output.ir_.operations.push_back(std::move(lowered));
    }
    output.ir_.attribute_schemas = checked.attribute_schemas;
    output.ir_.graphs = checked.graphs;
    for (const auto& rule : checked.rules) {
      executable::Rule lowered;
      static_cast<completed::RuleData&>(lowered) = rule;
      output.ir_.rules.push_back(std::move(lowered));
      output.axes_.push_back(input.axes(rule.id.value));
    }
    output.policy_ =
        input.policy() == SemanticPolicy::kAlgebraic ? "algebraic" : "strict";
    executable::validate(output);
    return output;
  }
};
Result<executable::ExecutableProgram> lower(
    const semantics::VerifiedProgram& input) {
  Result<executable::ExecutableProgram> result;
  try {
    result.program = Builder::run(input);
  } catch (const std::exception& e) {
    result.diagnostics.push_back({{}, e.what(), {}});
  }
  return result;
}
}  // namespace tepl::core::lowering
