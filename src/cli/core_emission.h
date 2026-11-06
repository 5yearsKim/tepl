#pragma once
#include <string>

#include "src/core/semantics/inspection.h"
namespace tepl {
std::string coreEmission(const core::semantics::InspectionData&,
                         const std::string&);
}
