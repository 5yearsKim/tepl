#pragma once
#include "src/core/semantics/inspection.h"
#include "src/core/semantics/options.h"
#include "src/shared/diagnostics/result.h"
namespace tepl::core::checking {
class TypedProgram;
}
namespace tepl::core::semantics {
// An inspection may retain unresolved proofs and cannot enter lowering.
Result<InspectionData> inspect(const checking::TypedProgram&,
                               const InspectionOptions& = {});
}  // namespace tepl::core::semantics
