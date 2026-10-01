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

// Load dialect declarations from imports relative to program.source_name.
// Appends imported dialects to the AST and reports missing, invalid, or cyclic
// imports at the corresponding import statement.
std::vector<Diagnostic> resolveImports(ast::Program& program);

// Check rule operation names, arities, and descriptor use against the loaded
// dialect declarations. Programs without a dialect keep the old syntax-only
// behavior.
std::vector<Diagnostic> validateDialectUses(const ast::Program& program);

}  // namespace tepl
