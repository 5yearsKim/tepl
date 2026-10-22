#pragma once
#include <map>

#include "src/core/checking/attribute_holes.h"
#include "src/core/checking/rule_ir.h"
#include "src/core/tables.h"
namespace tepl::core {
struct Rule : RuleData {
  std::vector<AttributeHole> attribute_holes;
  // Source spellings keep suggested rules usable with qualified imports.
  std::map<std::size_t, std::string> operation_spellings;
};
struct ProgramData : ProgramTables {
  std::vector<Rule> rules;
};
}  // namespace tepl::core
