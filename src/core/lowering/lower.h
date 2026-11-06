#pragma once
#include "src/core/semantics/program.h"
#include "src/executable/program.h"
namespace tepl::core::lowering {
Result<executable::ExecutableProgram> lower(const semantics::VerifiedProgram&);
}
