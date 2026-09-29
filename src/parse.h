#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tepl {

struct Diagnostic {
  std::size_t line;
  std::size_t column;
  std::string message;
};

struct ParseResult {
  std::vector<Diagnostic> diagnostics;
  std::size_t rule_count = 0;
  std::string tree;

  bool ok() const { return diagnostics.empty(); }
};

// Coordinates are one-based. Failed parses never expose a recovered parse tree.
ParseResult Parse(std::string_view source);

}  // namespace tepl
