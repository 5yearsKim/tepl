#pragma once

#include <ostream>

#include "src/shared/metadata/ir.h"

namespace tepl::core {
struct ProgramTables;
}
namespace tepl::core::metadata {
void print(std::ostream& out, const Program& shape,
           const core::ProgramTables& program);
}  // namespace tepl::core::metadata
