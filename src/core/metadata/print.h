#pragma once

#include <ostream>

#include "src/core/metadata/ir.h"

namespace tepl::core {
struct Program;
}
namespace tepl::core::metadata {
void print(std::ostream& out, const Program& shape,
           const core::Program& program);
}  // namespace tepl::core::metadata
