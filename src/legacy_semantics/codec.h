#pragma once
#include <string>

#include "src/legacy_semantics/plan.h"

namespace tepl::core::semantics {
std::string encodePlan(const inference::SemanticPlan&);
}
