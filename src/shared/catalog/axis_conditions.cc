#include "src/shared/catalog/axis_conditions.h"

#include <set>
#include <stdexcept>
namespace tepl::core::completed {
using B = builtins::Builtin;
bool usesAxisQueries(const TypedExpr& e) {
  bool result = false;
  if (const auto* b = std::get_if<BuiltinCall>(&e.value))
    result = builtins::isAxisQuery(b->builtin);
  forEachExpressionChild(
      e, [&](const auto& child) { result |= usesAxisQueries(child); });
  return result;
}
WhereSchedule scheduleWhere(const RuleData& r) {
  WhereSchedule out;
  std::set<std::size_t> dependent;
  std::function<bool(const TypedExpr&)> after = [&](const TypedExpr& e) {
    bool result = false;
    std::visit(
        [&](const auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, BuiltinCall>)
            result = builtins::isReplacementAxisQuery(v.builtin) ||
                     (v.builtin == B::kAxisFeasible && r.inference &&
                      r.inference->mode == inference::Mode::Validate);
          else if constexpr (std::is_same_v<T, LocalRef>)
            result = dependent.contains(v.local.value);
          else if constexpr (std::is_same_v<T, DescriptorRef> ||
                             std::is_same_v<T, DescriptorFieldRef>)
            result = v.descriptor.value < r.descriptors.size() &&
                     r.descriptors[v.descriptor.value].kind ==
                         Descriptor::Kind::kDerived;
        },
        e.value);
    forEachExpressionChild(e,
                           [&](const auto& child) { result |= after(child); });
    if (auto* l = std::get_if<LocalLet>(&e.value); l && result)
      dependent.insert(l->local.value);
    return result;
  };
  std::function<void(const TypedExprPtr&)> append = [&](const auto& e) {
    if (auto* b = std::get_if<BinaryExpr>(&e->value);
        b && b->op == BinaryOp::kLogicalAnd) {
      append(b->lhs);
      append(b->rhs);
    } else
      (after(*e) ? out.replacement : out.source).push_back(e);
  };
  for (const auto& c : r.conditions) append(c);
  return out;
}
void validateAxisConditions(const RuleData& r) {
  for (const auto& c : r.conditions)
    if (usesAxisQueries(*c))
      throw std::invalid_argument(
          "runtime axis queries are unsupported in executable conditions");
}
}  // namespace tepl::core::completed
