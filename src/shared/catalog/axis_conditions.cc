#include "src/shared/catalog/axis_conditions.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace tepl::core::completed {
namespace {
using B = builtins::Builtin;
class Formula {
 public:
  SourceOrigin origin;
  std::function<TypeId(Type)> type;
  template <class T>
  TypedExprPtr make(Type t, T v) {
    return std::make_shared<TypedExpr>(
        TypedExpr{origin, type(t), std::move(v)});
  }
  TypedExprPtr call(B b, std::vector<TypedExprPtr> args = {}) {
    auto t = Type{TypeKind::kBool};
    if (builtins::signature(b).result == builtins::SignatureType::kAxisList)
      t = listType(logicalAxisType());
    if (builtins::signature(b).result == builtins::SignatureType::kIntegerList)
      t = Type{TypeKind::kU64List};
    if (builtins::signature(b).result == builtins::SignatureType::kAxisSet)
      t = logicalAxisSetType();
    if (b == B::kConcat) t = listType(logicalAxisType());
    return make(t, BuiltinCall{b, std::move(args)});
  }
  TypedExprPtr query(B b, std::size_t node) {
    return call(b, {make(Type{TypeKind::kU64},
                         NumericConstant{std::to_string(node), false})});
  }
  TypedExprPtr equal(TypedExprPtr a, TypedExprPtr b) {
    return make(Type{TypeKind::kBool},
                BinaryExpr{BinaryOp::kEqual, std::move(a), std::move(b)});
  }
};
TypedExprPtr requiredExpression(const inference::Program& p, std::size_t node) {
  if (!p.components.empty())
    throw std::invalid_argument("axis requirements need a component scope");
  Formula f{{}, [](Type) { return TypeId{0}; }};
  if (node == p.replacement.back().node) return f.call(B::kAxisSourceOrder);
  for (const auto& parent : p.replacement)
    for (std::size_t i = 0; i < parent.inputs.size(); ++i)
      if (p.replacement[parent.inputs[i]].node == node) {
        if (parent.kind == inference::Kind::Permutation)
          return f.make(listType(logicalAxisType()), ListExpr{});
        if (parent.kind == inference::Kind::Summation)
          return requiredExpression(p, parent.node);
        return f.call(B::kAxisFilter,
                      {requiredExpression(p, parent.node),
                       f.query(i == 0 ? B::kAxisLeftFree : B::kAxisRightFree,
                               parent.node)});
      }
  throw std::invalid_argument("unknown axis requirement node");
}
// Canonical syntactic recognition deliberately does not claim a complete
// implication prover. Local names and conjunction grouping are irrelevant.
using Locals = std::map<std::size_t, std::string>;
std::string key(const TypedExpr& e, const Locals& locals,
                const inference::Program* plan = nullptr) {
  return std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, NumericConstant>)
          return "#" + v.spelling;
        else if constexpr (std::is_same_v<T, LocalRef>) {
          auto it = locals.find(v.local.value);
          return it == locals.end() ? "?" : it->second;
        } else if constexpr (std::is_same_v<T, BuiltinCall>) {
          auto builtin = v.builtin;
          auto arguments = v.arguments;
          if (plan && arguments.size() == 1) {
            if (auto* n = std::get_if<NumericConstant>(&arguments[0]->value)) {
              auto node = static_cast<std::size_t>(std::stoull(n->spelling));
              if (builtin == B::kAxisRequired)
                return key(*requiredExpression(*plan, node), locals, plan);
              if (builtin == B::kAxisSourceNodeOrder &&
                  node == plan->source.back().node) {
                builtin = B::kAxisSourceOrder;
                arguments.clear();
              }
              if (builtin == B::kAxisRhsNodeOrder &&
                  node == plan->replacement.back().node) {
                builtin = B::kAxisRhsOrder;
                arguments.clear();
              }
              if (builtin == B::kAxisInputOrder)
                for (const auto& input : plan->replacement)
                  if (input.node == node && input.capture) {
                    Formula f{{}, [](Type) { return TypeId{0}; }};
                    return key(*f.query(B::kAxisCaptureOrder, *input.capture),
                               locals, plan);
                  }
            }
          }
          std::string result(builtins::builtinName(builtin));
          result += "(";
          for (const auto& a : arguments) result += key(*a, locals, plan) + ",";
          return result + ")";
        } else if constexpr (std::is_same_v<T, BinaryExpr>)
          return std::string(spelling(v.op)) + "(" + key(*v.lhs, locals, plan) +
                 "," + key(*v.rhs, locals, plan) + ")";
        else if constexpr (std::is_same_v<T, ListExpr>) {
          std::string result = "[";
          for (const auto& a : v.elements)
            result += key(*a, locals, plan) + ",";
          return result + "]";
        } else
          return "?";
      },
      e.value);
}
std::string format(const TypedExpr& e) {
  return std::visit(
      [&](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, NumericConstant>)
          return v.spelling;
        else if constexpr (std::is_same_v<T, BuiltinCall>) {
          std::string result(builtins::builtinName(v.builtin));
          result += "(";
          for (std::size_t i = 0; i < v.arguments.size(); ++i) {
            if (i) result += ", ";
            result += format(*v.arguments[i]);
          }
          return result + ")";
        } else if constexpr (std::is_same_v<T, BinaryExpr>)
          return "(" + format(*v.lhs) + " " + std::string(spelling(v.op)) +
                 " " + format(*v.rhs) + ")";
        else
          throw std::logic_error("invalid generated axis condition");
      },
      e.value);
}
}  // namespace
std::vector<TypedExprPtr> axisConditions(
    const inference::Program& p, const SourceOrigin& origin,
    const std::function<TypeId(Type)>& type) {
  Formula f{origin, type};
  std::vector<TypedExprPtr> result{f.call(B::kAxisDomain)};
  if (!p.components.empty()) {
    result.push_back(f.call(B::kAxisFeasible));
    return result;
  }
  if (p.mode == inference::Mode::Validate) {
    result.push_back(f.call(B::kAxisRhsLegal));
    result.push_back(
        f.equal(f.call(B::kAxisSourcePartition), f.call(B::kAxisRhsPartition)));
    result.push_back(
        f.equal(f.call(B::kAxisSourceOrder), f.call(B::kAxisRhsOrder)));
    return result;
  }
  if (std::any_of(
          p.source.begin(), p.source.end(),
          [](const auto& n) { return n.kind == inference::Kind::Summation; }) ||
      std::any_of(
          p.replacement.begin(), p.replacement.end(),
          [](const auto& n) { return n.kind == inference::Kind::Summation; })) {
    result.push_back(f.call(B::kAxisFeasible));
    return result;
  }
  for (const auto& n : p.replacement) {
    auto q = f.query(B::kAxisRequired, n.node);
    if (n.kind == inference::Kind::Input)
      result.push_back(f.call(B::kAxisSubsequence,
                              {q, f.query(B::kAxisInputOrder, n.node)}));
    if (n.kind == inference::Kind::Contraction) {
      std::vector<TypedExprPtr> blocks;
      for (auto role : {B::kAxisBatch, B::kAxisLeftFree, B::kAxisRightFree})
        blocks.push_back(f.call(B::kAxisFilter, {q, f.query(role, n.node)}));
      result.push_back(f.equal(q, f.call(B::kConcat, std::move(blocks))));
    }
  }
  return result;
}
std::string axisWhere(const inference::Program& p) {
  auto conditions = axisConditions(p, {}, [](Type) { return TypeId{0}; });
  std::string result = "where {\n";
  for (std::size_t i = 0; i < conditions.size(); ++i)
    result += "  " + format(*conditions[i]) +
              (i + 1 < conditions.size() ? " &&\n" : "\n");
  return result + "}";
}
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
  if (!r.inference) {
    for (const auto& c : r.conditions)
      if (usesAxisQueries(*c))
        throw std::invalid_argument(
            "logical axis queries require an admitted synthesis plan");
    return;
  }
  Locals locals;
  std::set<std::string> conjuncts;
  std::function<void(const TypedExpr&)> collect = [&](const TypedExpr& e) {
    if (auto* b = std::get_if<BinaryExpr>(&e.value);
        b && b->op == BinaryOp::kLogicalAnd) {
      collect(*b->lhs);
      collect(*b->rhs);
    } else
      conjuncts.insert(key(e, locals, &*r.inference));
  };
  for (const auto& e : r.conditions) {
    if (auto* l = std::get_if<LocalLet>(&e->value))
      locals[l->local.value] = key(*l->value, locals, &*r.inference);
    else
      collect(*e);
  }
  for (const auto& required :
       axisConditions(*r.inference, {}, [](Type) { return TypeId{0}; }))
    if (!conjuncts.contains(key(*required, {}, &*r.inference)))
      throw std::invalid_argument(
          "missing axis feasibility condition in rule '" + r.name +
          "': " + format(*required) + ". Suggested " + axisWhere(*r.inference));
}
}  // namespace tepl::core::completed
