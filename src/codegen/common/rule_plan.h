#pragma once

#include <vector>

#include "src/core/ir.h"

namespace tepl::codegen {

// Semantic dependencies shared by all language backends. IDs remain core IDs;
// syntax, ownership, and runtime calls belong to the selected backend.
struct RulePlan {
  std::vector<core::HostFunctionId> host_functions;
  // Shape/dtype checks are unconditional. Expression-only metadata reads must
  // stay inside expression control flow to preserve short-circuit semantics.
  std::vector<core::CaptureId> constraint_captures;
  std::vector<core::CaptureId> metadata_captures;
};

RulePlan planRule(const core::Rule& rule);

}  // namespace tepl::codegen
