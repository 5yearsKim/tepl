#pragma once
#include <string>
#include <vector>

#include "src/shared/source/source.h"
namespace tepl {
struct Diagnostic {
  SourceOrigin origin;
  std::string message;
  std::vector<SourceLocation> related;
  Diagnostic(SourceOrigin at, std::string text,
             std::vector<SourceLocation> notes = {})
      : origin(std::move(at)),
        message(std::move(text)),
        related(std::move(notes)) {}
  Diagnostic(std::size_t line, std::size_t column, std::string text,
             std::string source = {})
      : origin{{std::move(source), {{line, column}, {line, column}}}, {}},
        message(std::move(text)) {}
};
namespace core {
using ::tepl::Diagnostic;
}
}  // namespace tepl
