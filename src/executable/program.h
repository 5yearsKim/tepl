#pragma once
#include <string>
#include <string_view>

#include "src/executable/ir.h"
namespace tepl::core::lowering {
struct Builder;
}
namespace tepl::executable {
// Validated runtime snapshot. Its representation has no inference requests or
// proof plans.
class ExecutableProgram {
 public:
  const ProgramData& ir() const { return ir_; }
  const std::vector<core::completed::SymbolPtr>& axes(std::size_t rule) const {
    return axes_.at(rule);
  }
  const inference::Emission& core(std::size_t rule) const {
    return cores_.at(rule);
  }
  const std::string& policy() const { return policy_; }

 private:
  ExecutableProgram() = default;
  ProgramData ir_;
  std::vector<std::vector<core::completed::SymbolPtr>> axes_;
  std::string policy_ = "strict";
  std::map<std::size_t, inference::Emission> cores_;
  friend struct core::lowering::Builder;
  friend ExecutableProgram load(std::string_view);
  friend std::string serialize(const ExecutableProgram&);
  friend void validate(const ExecutableProgram&);
};
ExecutableProgram load(std::string_view);
std::string serialize(const ExecutableProgram&);
std::string_view schema();
void validate(const ExecutableProgram&);
void validateGraph(const ProgramData&, const core::ConcreteGraph&);
}  // namespace tepl::executable
