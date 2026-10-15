#pragma once

#include <vector>

#include "src/codegen/output.h"
#include "src/core/ir.h"

namespace tepl::codegen {

// Semantic dependencies shared by all language backends. IDs remain core IDs;
// syntax, ownership, and runtime calls belong to the selected backend.
struct ConditionPlan {
  std::vector<core::CaptureId> captures;
  std::vector<core::DimensionId> dimensions;
  std::vector<core::DescriptorId> descriptors;
  std::vector<core::DTypeVariableId> dtypes;
};

struct RulePlan {
  bool requires_tensor_info = false;
  // All conditions are pure. Readiness depends on bindings, never on eagerly
  // fetched metadata.
  std::vector<ConditionPlan> early_conditions;
};

std::vector<Diagnostic> unsupportedRuleExpressions(
    const core::Program& program);

RulePlan planRule(const core::Rule& rule);

}  // namespace tepl::codegen
