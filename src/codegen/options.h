#pragma once

#include <string>
#include <string_view>

namespace tepl::codegen {

enum class Target { kRust, kCpp, kPython };
std::string_view targetName(Target target);

struct Options {
  Target target = Target::kRust;
  // Rule source paths are made relative to this directory, when provided.
  std::string rules_root;
  std::string package_name = "tepl_generated";
};

}  // namespace tepl::codegen
