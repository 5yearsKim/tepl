#pragma once

#include <string>

#include "src/core/semantics/inspection.h"

namespace tepl::core {

std::string formatProgram(const semantics::InspectionData& program);

}  // namespace tepl::core
