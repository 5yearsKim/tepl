#pragma once

#include <string>
#include <string_view>

namespace tepl::codegen::cpp {
// Namespace validation uses the same identifier policy as declarations.
void validateNamespace(std::string_view name);
inline std::string parentInclude(std::string_view path) {
  std::string result = "../";
  for (char c : path)
    if (c == '/') result += "../";
  return result;
}
}  // namespace tepl::codegen::cpp
