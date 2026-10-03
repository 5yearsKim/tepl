#pragma once

#include <string>
#include <vector>

#include "src/core/ir.h"

namespace tepl::codegen {

// Source identities and membership shared by every language backend. Names in
// paths are source names; each backend maps them to its own identifiers.
struct DialectPlan {
  std::string name;
  std::string source;
  std::vector<core::OpId> operations;
  std::vector<core::AttributeSchemaId> schemas;
};
struct ModulePlan {
  std::string source;
  std::vector<std::string> path;
  std::vector<core::RuleId> rules;
};
struct ProjectPlan {
  std::vector<DialectPlan> dialects;
  std::vector<ModulePlan> modules;
};

// Requires checked core. Rejects rule sources outside rules_root and ambiguous
// source module paths. Target-specific naming validation belongs to backends.
ProjectPlan planProject(const core::Program& program,
                        const std::string& rules_root = {});

}  // namespace tepl::codegen
