#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/syntax/ast.h"
#include "src/shared/diagnostics/diagnostic.h"

namespace tepl {

struct ParseResult {
  std::vector<Diagnostic> diagnostics;
  std::size_t rule_count = 0;
  std::string tree;
  std::optional<syntax::ParsedModule> program;

  bool ok() const { return program.has_value() && diagnostics.empty(); }
};

// Coordinates are one-based. Failed parses expose neither tree nor AST.
ParseResult parse(std::string_view source,
                  std::string_view source_name = "<input>");

ParseResult parse(const SourceFile&);

}  // namespace tepl
