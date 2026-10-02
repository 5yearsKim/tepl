#include "src/codegen/common/rule_plan.h"

#include <set>
#include <type_traits>

namespace tepl::codegen {
namespace {
class Planner {
 public:
  RulePlan run(const core::Rule& rule) {
    for (const auto& constraint : rule.constraints) {
      captures_.insert(constraint.capture.value);
      constrained_.insert(constraint.capture.value);
    }
    for (const auto& condition : rule.conditions) expression(*condition);
    for (const auto& derivation : rule.derivations)
      expression(*derivation.value);
    RulePlan plan;
    for (auto id : constrained_)
      plan.constraint_captures.push_back(core::CaptureId{id});
    for (auto id : hosts_)
      plan.host_functions.push_back(core::HostFunctionId{id});
    for (auto id : captures_)
      plan.metadata_captures.push_back(core::CaptureId{id});
    return plan;
  }

 private:
  void expression(const core::TypedExpr& expr) {
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, core::CaptureRef>)
            captures_.insert(value.capture.value);
          else if constexpr (std::is_same_v<T, core::HostCall>) {
            hosts_.insert(value.function.value);
            for (const auto& argument : value.arguments) expression(*argument);
          } else if constexpr (std::is_same_v<T, core::UnaryExpr>)
            expression(*value.operand);
          else if constexpr (std::is_same_v<T, core::BinaryExpr>) {
            expression(*value.lhs);
            expression(*value.rhs);
          }
        },
        expr.value);
  }
  std::set<std::size_t> hosts_, captures_, constrained_;
};
}  // namespace
RulePlan planRule(const core::Rule& rule) { return Planner().run(rule); }
}  // namespace tepl::codegen
