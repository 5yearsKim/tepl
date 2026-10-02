#pragma once

#include <string>
#include <unordered_map>

#include "src/core/dialect/symbols.h"
#include "src/core/rule/resolve.h"

namespace tepl::core::detail {

struct AnalysisContext;

// Visible names in one source file. Template pointers borrow the input AST
// for the duration of analysis; they never enter the checked IR.
struct FileScope {
  std::unordered_map<std::string, OpId> operations;
  std::unordered_map<std::string, const ast::Rule*> templates;
};

void buildFileScopes(AnalysisContext& context, const DialectRegistry& dialects,
                     const RuleRegistry& rules);

}  // namespace tepl::core::detail
