#pragma once

#include <string>
#include <vector>

#include "src/ast/ast.h"

namespace tepl {

struct SemanticDiagnostic {
  ast::SourceSpan span;
  std::string message;
};

// Checks the subset supported by the first structural rewriter: graph
// variables and operators without attributes, declarations, or conditions.
std::vector<SemanticDiagnostic> validateSimpleRule(const ast::Rule& rule);

}  // namespace tepl
