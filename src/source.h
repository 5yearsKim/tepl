#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tepl {

// One-based lines and Unicode code-point columns. Spans are half-open.
struct SourcePosition {
  std::size_t line = 1;
  std::size_t column = 1;
};

struct SourceSpan {
  SourcePosition begin;
  SourcePosition end;
};

struct SourceLocation {
  std::string source_name;
  SourceSpan span;
};

struct SourceOrigin {
  SourceLocation definition;
  std::vector<SourceLocation> expansions;
};

}  // namespace tepl
