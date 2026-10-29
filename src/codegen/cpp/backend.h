#pragma once
#include "src/codegen/generate.h"
namespace tepl::codegen::cpp {
GenerationResult generate(const executable::ExecutableProgram&, const Options&);
}
