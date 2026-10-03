#pragma once

#include <ostream>

#include "src/core/shape/ir.h"

namespace tepl::core {
struct Program;
}
namespace tepl::core::shape {
void print(std::ostream& out, const Program& shape,
           const core::Program& program);
}  // namespace tepl::core::shape
