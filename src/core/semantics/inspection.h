#pragma once
#include "src/core/checking/ir.h"
#include "src/core/semantics/ir.h"
namespace tepl::core::semantics {
struct InspectedRule : Rule {
  std::shared_ptr<const Plan> semantic_plan;
};
struct InspectionData : ProgramTables {
  std::vector<InspectedRule> rules;
};
}  // namespace tepl::core::semantics
