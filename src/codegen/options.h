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
  // Concrete graph source paths are made relative to this directory.
  std::string graphs_root;
};

}  // namespace tepl::codegen
