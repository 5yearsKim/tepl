#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "src/core/ir.h"

namespace tepl::program {
// Versioned executable snapshot. Construction and loading validate the same
// contract. No AST, import resolver, or synthesis engine is needed to run it.
class Program {
 public:
  const core::Program& ir() const { return ir_; }
  const std::vector<core::index_semantics::SymbolPtr>& axes(
      std::size_t rule) const {
    return axes_.at(rule);
  }
  const std::string& policy() const { return policy_; }

 private:
  core::Program ir_;
  std::vector<std::vector<core::index_semantics::SymbolPtr>> axes_;
  std::string policy_ = "strict";
  friend Program compile(const core::Program&);
  friend Program load(std::string_view);
  friend std::string serialize(const Program&);
  friend void validate(const Program&);
};
// Throws std::invalid_argument with a diagnostic on invalid input.
Program compile(const core::Program& checked);
Program load(std::string_view json);
std::string serialize(const Program&);
std::string_view schema();
void validate(const Program&);
void validateGraph(const core::Program&, const core::ConcreteGraph&);
}  // namespace tepl::program
