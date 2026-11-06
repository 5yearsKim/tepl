#pragma once
#include <optional>
#include <vector>

#include "src/shared/diagnostics/diagnostic.h"
namespace tepl::core {
template <class T>
struct Result {
  std::optional<T> program;
  std::vector<Diagnostic> diagnostics;
  bool ok() const { return program.has_value() && diagnostics.empty(); }
};
}  // namespace tepl::core
