#pragma once
#include "src/core/semantics/program.h"
namespace tepl::core::semantics {
Result<VerifiedProgram> verify(const checking::TypedProgram&,
                               const SemanticOptions& = {});
}
