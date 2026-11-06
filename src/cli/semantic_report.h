#pragma once
#include <ostream>
#include <string>

#include "src/core/semantics/inspection.h"
namespace tepl {
int checkSemanticRules(const core::semantics::InspectionData&,
                       const std::string& rule, std::ostream& output);
}  // namespace tepl
