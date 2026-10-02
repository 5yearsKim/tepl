#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/ast/ast.h"
#include "src/core/types.h"

namespace tepl {

using DType = core::DType;

std::optional<DType> resolveDType(std::string_view name);

struct SemanticDiagnostic {
  ast::SourceSpan span;
  std::string message;
};

// Validate annotations and local tensor declarations. Unexpanded inherited
// rules validate annotations but defer capture checks until expansion.
std::vector<SemanticDiagnostic> validateTensorTypes(const ast::Rule& rule);

// Checks the subset supported by the first structural rewriter: graph
// variables, numeric literals, and operators without attributes, declarations,
// or conditions.
std::vector<SemanticDiagnostic> validateSimpleRule(const ast::Rule& rule);

}  // namespace tepl
