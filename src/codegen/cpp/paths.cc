#include "src/codegen/cpp/paths.h"

#include "src/codegen/cpp/names.h"

namespace tepl::codegen::cpp {
void validateNamespace(std::string_view name) {
  if (name.empty())
    throw std::invalid_argument("C++ namespace cannot be empty");
  if (name == "std" || name.starts_with("std::"))
    throw std::invalid_argument(
        "generated C++ cannot declare types in namespace std");
  while (true) {
    const auto split = name.find("::");
    // Namespace spelling is preserved, independently of module casing.
    Identifier::make(name.substr(0, split), NameKind::kField, {});
    if (split == std::string_view::npos) break;
    name.remove_prefix(split + 2);
  }
}
}  // namespace tepl::codegen::cpp
