#pragma once

#include <string>
#include <string_view>

namespace tepl::codegen::rust::paths {
// root is relative to the module containing the reference. Keeping generated
// paths relative allows the output to be relocated in the consuming crate.
inline std::string within(std::string_view root, std::string_view member) {
  return std::string(root) + "::" + std::string(member);
}
inline std::string parent(std::string_view path) {
  return "super::" + std::string(path);
}
inline std::string external(std::string_view crate, std::string_view member) {
  return "::" + std::string(crate) + "::" + std::string(member);
}
}  // namespace tepl::codegen::rust::paths
