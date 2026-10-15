#pragma once
#include "src/codegen/generate.h"
namespace tepl::codegen::rust {
GenerationResult generate(const executable::ExecutableProgram&, const Options&);
}
