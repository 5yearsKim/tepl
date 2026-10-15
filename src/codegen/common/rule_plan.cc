#include "src/codegen/common/rule_plan.h"

#include <functional>
#include <set>
#include <type_traits>

namespace tepl::codegen {
namespace {
struct Dependencies {
  std::set<std::size_t> captures, dimensions, descriptors, dtypes;
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
        else if constexpr (std::is_same_v<T, core::BuiltinCall>) {
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

std::vector<Diagnostic> unsupportedRuleExpressions(
    const core::Program& program) {
  std::vector<Diagnostic> diagnostics;
  for (const auto& rule : program.rules) {
    std::function<bool(const core::TypedExpr&)> unsupported =
        [&](const core::TypedExpr& e) {
          const auto& type = program.types.at(e.type.value);
          if (type.list_depth || type.optional ||
              type.kind == core::TypeKind::kString ||
              type.kind == core::TypeKind::kPrecision ||
              type.kind == core::TypeKind::kOpaque)
            return true;
          return std::visit(
              [&](const auto& v) -> bool {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, core::BuiltinCall>) {
                  for (const auto& a : v.arguments)
                    if (unsupported(*a)) return true;
                  return false;
                } else if constexpr (std::is_same_v<T, core::UnaryExpr>)
                  return unsupported(*v.operand);
                else if constexpr (std::is_same_v<T, core::BinaryExpr>)
                  return unsupported(*v.lhs) || unsupported(*v.rhs);
                else
                  return !(std::is_same_v<T, core::CaptureRef> ||
                           std::is_same_v<T, core::DimensionRef> ||
                           std::is_same_v<T, core::DescriptorRef> ||
                           std::is_same_v<T, core::DescriptorFieldRef> ||
                           std::is_same_v<T, core::NumericConstant> ||
                           std::is_same_v<T, bool> ||
                           std::is_same_v<T, core::DType> ||
                           std::is_same_v<T, core::DTypeVariableRef>);
              },
              e.value);
        };
    for (const auto& e : rule.conditions)
      if (unsupported(*e))
        diagnostics.push_back(
            {e->origin,
             "code generation for extended rule expressions is not "
             "implemented; use tepl check to validate this rule"});
    for (const auto& d : rule.derivations)
      if (unsupported(*d.value))
        diagnostics.push_back(
            {d.origin,
             "code generation for descriptor constructors and extended rule "
             "expressions is not implemented; use tepl check to validate this "
             "rule"});
  }
  return diagnostics;
}

RulePlan planRule(const core::Rule& rule) {
  RulePlan plan;
  plan.requires_tensor_info =
      !rule.constraints.empty() || rule.inference.has_value();
  for (const auto& condition : rule.conditions) {
    Dependencies deps;
    collect(*condition, deps);
    plan.requires_tensor_info |= !deps.captures.empty() ||
                                 !deps.dimensions.empty() ||
                                 !deps.dtypes.empty();
    plan.early_conditions.push_back({ids<core::CaptureId>(deps.captures),
                                     ids<core::DimensionId>(deps.dimensions),
                                     ids<core::DescriptorId>(deps.descriptors),
                                     ids<core::DTypeVariableId>(deps.dtypes)});
  }
  for (const auto& derivation : rule.derivations) {
    Dependencies deps;
    collect(*derivation.value, deps);
    plan.requires_tensor_info |= !deps.captures.empty() ||
                                 !deps.dimensions.empty() ||
                                 !deps.dtypes.empty();
  }
  return plan;
}
}  // namespace tepl::codegen
