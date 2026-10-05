#include "src/codegen/common/rule_plan.h"

#include <set>
#include <type_traits>

namespace tepl::codegen {
namespace {
struct Dependencies {
  std::set<std::size_t> captures, dimensions, descriptors, hosts, dtypes;
};
void collect(const core::TypedExpr& expr, Dependencies& deps) {
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, core::CaptureRef>)
          deps.captures.insert(value.capture.value);
        else if constexpr (std::is_same_v<T, core::DimensionRef>)
          deps.dimensions.insert(value.dimension.value);
        else if constexpr (std::is_same_v<T, core::DTypeVariableRef>)
          deps.dtypes.insert(value.variable.value);
        else if constexpr (std::is_same_v<T, core::DescriptorFieldRef>)
          deps.descriptors.insert(value.descriptor.value);
        else if constexpr (std::is_same_v<T, core::DescriptorRef>)
          deps.descriptors.insert(value.descriptor.value);
        else if constexpr (std::is_same_v<T, core::HostCall> ||
                           std::is_same_v<T, core::BuiltinCall>) {
          if constexpr (std::is_same_v<T, core::HostCall>)
            deps.hosts.insert(value.function.value);
          for (const auto& argument : value.arguments) collect(*argument, deps);
        } else if constexpr (std::is_same_v<T, core::UnaryExpr>)
          collect(*value.operand, deps);
        else if constexpr (std::is_same_v<T, core::BinaryExpr>) {
          collect(*value.lhs, deps);
          collect(*value.rhs, deps);
        }
      },
      expr.value);
}
template <class Id>
std::vector<Id> ids(const std::set<std::size_t>& values) {
  std::vector<Id> result;
  for (auto value : values) result.push_back(Id{value});
  return result;
}
}  // namespace

RulePlan planRule(const core::Rule& rule) {
  RulePlan plan;
  std::set<std::size_t> hosts;
  bool early = true;
  for (const auto& condition : rule.conditions) {
    Dependencies deps;
    collect(*condition, deps);
    early = early && deps.hosts.empty();
    if (early)
      plan.early_conditions.push_back(
          {ids<core::CaptureId>(deps.captures),
           ids<core::DimensionId>(deps.dimensions),
           ids<core::DescriptorId>(deps.descriptors),
           ids<core::DTypeVariableId>(deps.dtypes)});
    hosts.insert(deps.hosts.begin(), deps.hosts.end());
  }
  for (const auto& derivation : rule.derivations) {
    Dependencies deps;
    collect(*derivation.value, deps);
    hosts.insert(deps.hosts.begin(), deps.hosts.end());
  }
  plan.host_functions = ids<core::HostFunctionId>(hosts);
  return plan;
}
}  // namespace tepl::codegen
