#pragma once
#include "src/codegen/options.h"
#include "src/codegen/output.h"
#include "src/executable/program.h"

namespace tepl::codegen {
// Generation is pure: validated executable IR in, Rust source files out.
GenerationResult generate(const executable::ExecutableProgram& program,
                          const Options& options = {});
}  // namespace tepl::codegen
