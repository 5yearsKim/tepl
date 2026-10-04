#pragma once

#include <vector>

#include "src/core/ir.h"

namespace tepl::codegen {

// Semantic dependencies shared by all language backends. IDs remain core IDs;
// syntax, ownership, and runtime calls belong to the selected backend.
struct ConditionPlan {
  std::vector<core::CaptureId> captures;
  std::vector<core::DimensionId> dimensions;
  std::vector<core::DescriptorId> descriptors;
};

struct RulePlan {
  std::vector<core::HostFunctionId> host_functions;
  // The ordered builtin-only prefix ends at the first condition containing a
  // host call. Readiness depends on bindings, never on eagerly fetched
  // metadata.
  std::vector<ConditionPlan> early_conditions;
  // Shape/dtype checks are unconditional. Expression-only metadata reads must
  // stay inside expression control flow to preserve short-circuit semantics.
  std::vector<core::CaptureId> constraint_captures;
  std::vector<core::CaptureId> metadata_captures;
};

RulePlan planRule(const core::Rule& rule);

}  // namespace tepl::codegen
