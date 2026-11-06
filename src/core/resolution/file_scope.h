#pragma once

#include <string>
#include <unordered_map>

#include "src/core/resolution/dialects/symbols.h"
#include "src/core/resolution/rules/resolve.h"

namespace tepl::core::detail {

struct ResolutionContext;

// Visible names in one source file. Template pointers borrow the input AST
// for the duration of analysis; they never enter the checked IR.
struct FileScope {
  std::unordered_map<std::string, OpId> operations;
  std::unordered_map<std::string, const syntax::Rule*> templates;
};

void buildFileScopes(ResolutionContext& context,
                     const DialectRegistry& dialects,
                     const RuleRegistry& rules);

}  // namespace tepl::core::detail
