#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/ast/ast.h"

namespace tepl {

struct Diagnostic {
  std::size_t line;
  std::size_t column;
  std::string message;
  std::string source_name;
};

struct ParseResult {
  std::vector<Diagnostic> diagnostics;
  std::size_t rule_count = 0;
  std::string tree;
  std::optional<ast::Program> program;

  bool ok() const { return diagnostics.empty(); }
};

// Coordinates are one-based. Failed parses expose neither tree nor AST.
ParseResult parse(std::string_view source,
                  std::string_view source_name = "<input>");

}  // namespace tepl
