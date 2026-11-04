#pragma once
#include <map>

#include "src/core/checking/rule_ir.h"
#include "src/core/tables.h"
namespace tepl::core {
struct Rule : RuleData {
  bool requiresSynthesis() const {
    for (const auto& d : descriptors)
      if (d.kind == Descriptor::Kind::kInferred) return true;
    return false;
  }
  // Source spellings preserve qualified operation names in diagnostics.
  std::map<std::size_t, std::string> operation_spellings;
};
struct ProgramData : ProgramTables {
  std::vector<Rule> rules;
};
}  // namespace tepl::core
