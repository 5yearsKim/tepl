#pragma once
#include "src/core/checking/attribute_holes.h"
#include "src/core/checking/rule_ir.h"
#include "src/core/tables.h"
namespace tepl::core {
struct Rule : RuleData {
  std::vector<AttributeHole> attribute_holes;
};
struct ProgramData : ProgramTables {
  std::vector<Rule> rules;
};
}  // namespace tepl::core
