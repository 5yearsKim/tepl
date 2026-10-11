#pragma once

#include "src/core/inference/semantic_ir.h"

namespace tepl::core::inference {
// Expand and normalize primitive semantics at compile time. This never
// dispatches on operation names. Unsupported expressions retain a fallback.
ConstraintPlan compileConstraints(const SemanticPlan&);
}  // namespace tepl::core::inference
