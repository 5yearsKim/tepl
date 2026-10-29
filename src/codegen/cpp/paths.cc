#include "src/codegen/cpp/paths.h"

#include "src/codegen/cpp/names.h"

namespace tepl::codegen::cpp {
void validateNamespace(std::string_view name) {
  if (name.empty())
    throw std::invalid_argument("C++ namespace cannot be empty");
  if (name == "eggc" || name.starts_with("eggc::") || name == "std" ||
      name.starts_with("std::") || name == "tepl_runtime" ||
      name.starts_with("tepl_runtime::"))
    throw std::invalid_argument(
        "generated C++ namespace conflicts with a runtime namespace");
  while (true) {
    const auto split = name.find("::");
    // Namespace spelling is preserved, independently of module casing.
    const auto part = name.substr(0, split);
    if (Identifier::make(part, NameKind::kField, {}).token != part)
      throw std::invalid_argument("C++ namespace contains a keyword");
    if (split == std::string_view::npos) break;
    name.remove_prefix(split + 2);
  }
}
}  // namespace tepl::codegen::cpp
