#pragma once

#include <string>

#include "src/codegen/rust/names.h"

namespace tepl::codegen::rust {
std::string emitShapeInference(const core::Program& program,
                               const Names& names);
std::string emitDTypeInference(const core::Program& program,
                               const Names& names);
}  // namespace tepl::codegen::rust
