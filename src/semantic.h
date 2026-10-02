#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/ast/ast.h"

namespace tepl {

enum class DType {
  kBool,
  kI8,
  kI16,
  kI32,
  kI64,
  kU8,
  kU16,
  kU32,
  kU64,
  kF16,
  kBF16,
  kF32,
  kF64,
};

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
